#!/usr/bin/env python3
"""Pure-pursuit controller for AutoDRIVE RoboRacer (speed + crash-safety version).

Steering, path estimation and lookahead are unchanged from the speed-focused v4
program. The changes here are safeguards around the speed planner and the
output stage, so the car keeps its pace on open stretches but no longer carries
that pace into places it cannot see through:

  1. Visible-distance rule: when the estimated path ends because the walls
     stop being visible (not because the sensor range ran out), the planner
     must be able to slow to `end_speed` by the end of that path.
  2. Crash-proof loop: an error inside one scan callback no longer kills the
     node (a dead node leaves the car driving on its last command). The car
     falls back to the gap fallback or brakes instead.
  3. Bad-number guard: a NaN/inf steering or throttle value is never published
     (it would otherwise be clamped to full lock / full throttle).
  4. New 'balanced' profile (default): a little more braking margin than 'fast'.
     Switch back with:  --ros-args -p profile:=fast

The controller builds a short, local estimate of the track centreline from LiDAR
returns, chooses a speed- and curvature-dependent point ahead, and steers toward it
using the pure-pursuit curvature equation. It does not require a hand-written
track map or waypoint file.

ROS interface kept compatible with the existing wall_follow_node:
  Subscribes: /autodrive/roboracer_1/lidar (sensor_msgs/LaserScan)
              /autodrive/roboracer_1/odom  (nav_msgs/Odometry, when available)
              /autodrive/roboracer_1/speed  (std_msgs/Float32, optional fallback)
  Publishes:  /autodrive/roboracer_1/steering_command (std_msgs/Float32, [-1, 1])
              /autodrive/roboracer_1/throttle_command (std_msgs/Float32, [-1, 1])

Run with the default fast profile:
  ros2 run autodrive_roboracer wall_follow_node

For an even more aggressive experiment, only after testing fast:
  ros2 run autodrive_roboracer wall_follow_node --ros-args -p profile:=max

The pure-pursuit target remains smooth; a bounded curvature-preview term starts
steering into an upcoming bend before the target point itself has moved far from
straight ahead. This speed-focused experiment is intended to measure pace separately from collision fixes. It is not a guarantee of zero crashes or a
globally optimal lap time; validate it against the simulator.
"""

import itertools
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

MAX_STEER_RAD = 0.5236  # AutoDRIVE's steering limit is approximately +/-30 degrees.
WHEELBASE_M = 0.324      # RoboRacer vehicle wheelbase from the AutoDRIVE technical guide.
LIDAR_OFFSET_X_M = 0.2733  # LiDAR is this far in front of the rear axle.


PROFILES = {
    # Start here if the car has not yet completed a clean lap.
    'safe': {
        'max_throttle': 0.48,
        'straight_throttle': 0.68,
        'max_brake': 0.55,
        'brake_decel': 4.5,
        'max_lateral_accel': 7.0,
        'lookahead_base': 0.80,
        'lookahead_speed_gain': 0.10,
    },
    # Default. Between 'safe' and 'fast': slightly lower corner speed, earlier and
    # firmer braking. Same steering and lookahead as 'fast'.
    'balanced': {
    'max_throttle': 0.75,
    'straight_throttle': 1.0,
    'max_brake': 0.90,
    'brake_decel': 8.0,
    'brake_gain': 0.45,
    'reaction_time': 0.28,
    'max_lateral_accel': 14.0,
    'lookahead_base': 0.45,
    'lookahead_speed_gain': 0.55,
    },
    # Speed-focused fast profile: modestly higher straight and corner speeds than v4.
    # Steering, lookahead, path estimation, braking timing, and smoothing remain v4.
    'fast': {
        'max_throttle': 0.66,          # v4: 0.62
        'straight_throttle': 0.96,     # v4: 0.90
        'max_brake': 0.70,
        'brake_decel': 5.5,
        'max_lateral_accel': 10.5,     # v4: 10.0; small corner-speed increase
        'lookahead_base': 0.65,
        'lookahead_speed_gain': 0.12,
    },
    # Try only after fast is completing laps without hitting the track boundary.
    'max': {
        'max_throttle': 0.72,
        'straight_throttle': 1.00,
        'max_brake': 0.85,
        'brake_decel': 6.5,
        'max_lateral_accel': 13.0,
        'lookahead_base': 0.58,
        'lookahead_speed_gain': 0.105,
    },
}


def clamp(value, low, high):
    return max(low, min(high, value))


def smooth_median(values, window=3):
    """Small median filter; suppresses noisy LiDAR centreline samples."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = values.copy()
    for i in range(len(values)):
        out[i] = float(np.median(values[max(0, i - half):min(len(values), i + half + 1)]))
    return out


def cluster_lateral_points(y_values, gap):
    """Group nearby lateral LiDAR returns into candidate track-boundary clusters."""
    ys = np.sort(np.asarray(y_values, dtype=float))
    if len(ys) == 0:
        return []
    split_at = np.where(np.diff(ys) > gap)[0] + 1
    groups = np.split(ys, split_at)
    # A one-beam blip is too fragile to trust as a wall.
    return [group for group in groups if len(group) >= 2]


class WallFollowNode(Node):
    """The ROS node is still named wall_follow_node to preserve the package entry point."""

    def __init__(self):
        super().__init__('wall_follow_node')

        self.declare_parameter('profile', 'balanced')
        profile_name = str(self.get_parameter('profile').value)
        preset = PROFILES.get(profile_name)
        if preset is None:
            self.get_logger().warning(f"Unknown profile '{profile_name}'; using 'balanced'.")
            profile_name, preset = 'balanced', PROFILES['balanced']
        self.profile = profile_name

        def dp(name, default):
            self.declare_parameter(name, preset.get(name, default))
            return self.get_parameter(name).value

        # Longitudinal speed / steering behaviour.
        dp('min_throttle', 0.08)
        dp('max_throttle', 0.62)
        dp('straight_throttle', 0.90)
        dp('speed_per_throttle', 24.0)  # retained from the user's previous HUD-based tuning
        dp('max_brake', 0.70)
        dp('brake_decel', 5.5)
        dp('brake_gain', 0.28)
        dp('brake_margin', 0.45)
        dp('reaction_time', 0.10)
        dp('brake_dist', 0.35)
        dp('max_lateral_accel', 10.0)
        dp('min_curve_speed', 1.8)
        # Visible-distance rule: if the path ends before the sensor range does, be
        # able to slow to this speed by the path's end (same units as the speeds above).
        dp('end_speed', 3.0)
        dp('end_view_margin', 0.75)  # m short of the full view that counts as "path was cut off"

        # Pure-pursuit geometry.
        dp('wheelbase', WHEELBASE_M)
        dp('max_steer_rad', MAX_STEER_RAD)
        dp('lookahead_base', 0.65)
        dp('lookahead_speed_gain', 0.12)
        dp('lookahead_min', 0.75)
        dp('lookahead_max', 2.70)
        dp('lidar_offset_x', LIDAR_OFFSET_X_M)
        dp('steer_smoothing', 0.22)
        dp('steer_rate_limit', 5.0)  # normalized steering units per second
        # Anticipatory steering: use a bounded estimate of the bend ahead, but fade
        # it out as pure pursuit starts tracking that bend to prevent double-turning.
        dp('turn_preview_distance', 2.5)  # metres ahead of rear axle
        dp('turn_preview_span', 1.15)     # half-width of the curvature sampling window
        dp('turn_preview_gain', 0.38)     # feed-forward fraction before pursuit catches up
        dp('turn_preview_min_curvature', 0.045)

        # LiDAR-to-centreline estimation.
        dp('range_cap', 9.0)  # compatibility with the previous launch command
        dp('max_path_range', 8.5)
        dp('path_step', 0.25)
        dp('path_bin_half_width', 0.26)
        dp('track_width', 1.15)
        dp('min_track_width', 0.55)
        dp('max_track_width', 3.50)
        dp('max_boundary_offset', 3.25)
        dp('boundary_min_offset', 0.12)
        dp('boundary_cluster_gap', 0.34)
        dp('min_path_points', 6)
        dp('min_path_length', 2.0)
        dp('max_interp_gap_bins', 4)

        # Obstacles and fallback behaviour.
        dp('car_half_width', 0.135)
        dp('obstacle_half_width', 0.23)
        dp('obstacle_safety_distance', 0.55)
        dp('fallback_speed', 2.8)
        dp('gap_fov_deg', 80.0)
        dp('gap_min_range', 0.85)
        dp('bubble', 0.36)
        dp('gap_gain', 1.5)  # compatibility with the previous launch command
        dp('slow_dist', 6.0)  # compatibility with the previous launch command
        dp('log_period', 1.0)

        self.pub_throttle = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.pub_steer = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.sub_lidar = self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.on_scan, 10)
        # Odometry is preferred over estimating speed from the throttle command.
        # The Float32 speed topic is kept as a fallback for bridge versions that provide it.
        self.sub_odom = self.create_subscription(
            Odometry, '/autodrive/roboracer_1/odom', self.on_odom, 10)
        self.sub_speed = self.create_subscription(
            Float32, '/autodrive/roboracer_1/speed', self.on_speed, 10)

        self.prev_scan_time = None
        self.last_log_time = None
        self.last_throttle = 0.0
        self.last_steer = 0.0
        self.v_est = 0.0
        self.v_meas = 0.0
        self.v_meas_time = None
        self.v_meas_source = ''
        self.scan_count = 0
        self.fallback_count = 0
        self.view_limit = 0.0       # how far the last centreline search could see (m)
        self.error_count = 0

        self.get_logger().info(
            f"LiDAR pure-pursuit controller started (profile '{self.profile}'). "
            "It estimates the track centreline locally; no track map is required.")

    # ------------------------------------------------------------------
    # Speed feedback
    # ------------------------------------------------------------------
    def on_odom(self, msg):
        speed = math.hypot(float(msg.twist.twist.linear.x),
                           float(msg.twist.twist.linear.y))
        self._store_speed(speed, 'odom')

    def on_speed(self, msg):
        self._store_speed(abs(float(msg.data)), 'speed-topic')

    def _store_speed(self, speed, source):
        now = self.get_clock().now().nanoseconds * 1e-9
        if math.isfinite(speed):
            self.v_meas = max(0.0, speed)
            self.v_meas_time = now
            self.v_meas_source = source

    def current_speed(self, now, dt):
        """Use live simulator speed feedback, or a simple command-based estimate."""
        p = self.get_parameter
        k = float(p('speed_per_throttle').value)
        if self.last_throttle < 0.0:
            # A rough braking model keeps fallback speed estimation from staying high forever.
            self.v_est = max(0.0, self.v_est - float(p('brake_decel').value) * dt)
        else:
            target_est = max(0.0, self.last_throttle) * k
            alpha = 1.0 - math.exp(-dt / 0.45)
            self.v_est += alpha * (target_est - self.v_est)

        if self.v_meas_time is not None and now - self.v_meas_time < 0.5:
            return self.v_meas, self.v_meas_source
        return self.v_est, 'estimated'

    # ------------------------------------------------------------------
    # LiDAR and local track centreline
    # ------------------------------------------------------------------
    @staticmethod
    def scan_xy(scan, lidar_offset_x):
        """Convert scan beams into points in a frame whose origin is the rear axle."""
        ranges = np.asarray(scan.ranges, dtype=float)
        if len(ranges) == 0 or abs(float(scan.angle_increment)) < 1e-9:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        angles = float(scan.angle_min) + np.arange(len(ranges)) * float(scan.angle_increment)
        finite = np.isfinite(ranges)
        hit = finite & (ranges >= float(scan.range_min)) & (ranges < float(scan.range_max) - 0.035)
        r = np.where(finite, ranges, float(scan.range_max))
        r = np.clip(r, float(scan.range_min), float(scan.range_max))
        x = r * np.cos(angles) + lidar_offset_x
        y = r * np.sin(angles)
        return x, y, angles, hit

    def estimate_centerline(self, scan, x_all, y_all, angles, hit):
        """Estimate a sequence of local centreline points from track-boundary returns.

        At each forward slice, lateral point clusters are treated as candidate walls.
        If two plausible walls are visible, their midpoint is the centreline. If one
        wall is visible, the previous centreline estimate and expected track width are
        used to estimate the other side. The estimate is deliberately local: it is
        rebuilt from the newest scan rather than accumulated into a drifting map.
        """
        p = self.get_parameter
        step = max(0.12, float(p('path_step').value))
        range_limit = min(float(p('range_cap').value),
                          float(p('max_path_range').value),
                          float(scan.range_max) - 0.35)
        self.view_limit = range_limit
        if range_limit < 2.0:
            return None

        # Only forward returns can be used as upcoming path points. Keep wall returns
        # away from the far left/right edge of the sensor field where projection is noisy.
        valid = (hit & (x_all > 0.25) & (x_all < range_limit)
                 & (np.abs(angles) < math.radians(84.0))
                 & (np.abs(y_all) >= float(p('boundary_min_offset').value))
                 & (np.abs(y_all) <= float(p('max_boundary_offset').value)))
        if int(np.count_nonzero(valid)) < 8:
            return None

        px = x_all[valid]
        py = y_all[valid]
        x_samples = np.arange(0.40, range_limit, step)
        raw_centres = np.full(len(x_samples), np.nan, dtype=float)
        expected_width = float(p('track_width').value)
        min_width = float(p('min_track_width').value)
        max_width = float(p('max_track_width').value)
        half_bin = max(float(p('path_bin_half_width').value), 0.5 * step)
        cluster_gap = float(p('boundary_cluster_gap').value)
        predicted_center = 0.0
        previous_center = None
        previous_x = None

        for i, x_sample in enumerate(x_samples):
            in_slice = np.abs(px - x_sample) <= half_bin
            ys = py[in_slice]
            clusters = cluster_lateral_points(ys, cluster_gap)
            cluster_centres = [float(np.median(group)) for group in clusters]
            chosen_center = None

            # Select the pair whose spacing resembles a track and whose midpoint changes
            # least from the preceding slice. This works even when both walls are on the
            # same side of the car in a tight turn.
            pair_candidates = []
            for a, b in itertools.combinations(cluster_centres, 2):
                width = abs(b - a)
                if min_width <= width <= max_width:
                    midpoint = 0.5 * (a + b)
                    score = abs(midpoint - predicted_center) + 0.30 * abs(width - expected_width)
                    pair_candidates.append((score, midpoint))
            if pair_candidates:
                pair_candidates.sort(key=lambda item: item[0])
                chosen_center = pair_candidates[0][1]
            elif cluster_centres:
                # One wall only: decide which side it most likely occupies relative to
                # the previous centre estimate, then offset by half the expected width.
                wall_y = min(cluster_centres, key=lambda value: abs(value - predicted_center))
                if wall_y >= predicted_center:
                    chosen_center = wall_y - 0.5 * expected_width
                else:
                    chosen_center = wall_y + 0.5 * expected_width

            if chosen_center is not None and math.isfinite(chosen_center):
                # Reject single-slice leaps caused by a stray beam/obstacle.
                if previous_center is not None:
                    dx = max(x_sample - previous_x, step)
                    max_change = 1.05 * (dx / step)
                    if abs(chosen_center - previous_center) > max_change:
                        # Retain the previous trend rather than following a one-bin outlier.
                        chosen_center = previous_center + math.copysign(
                            max_change, chosen_center - previous_center)
                raw_centres[i] = chosen_center
                if previous_center is not None and previous_x is not None:
                    slope = (chosen_center - previous_center) / max(x_sample - previous_x, step)
                    predicted_center = chosen_center + clamp(slope, -1.8, 1.8) * step
                else:
                    predicted_center = chosen_center
                previous_center = chosen_center
                previous_x = x_sample

        valid_idx = np.flatnonzero(np.isfinite(raw_centres))
        if len(valid_idx) < int(p('min_path_points').value):
            return None

        # Use the contiguous near-field run; don't invent a path across large areas where
        # no wall geometry was observed. Short gaps (up to ~1 m) are interpolated.
        max_gap_bins = max(1, int(p('max_interp_gap_bins').value))
        end_pos = 0
        for j in range(1, len(valid_idx)):
            if valid_idx[j] - valid_idx[j - 1] > max_gap_bins:
                break
            end_pos = j
        usable_idx = valid_idx[:end_pos + 1]
        if len(usable_idx) < int(p('min_path_points').value):
            return None

        first, last = int(usable_idx[0]), int(usable_idx[-1])
        if x_samples[last] - x_samples[first] < float(p('min_path_length').value):
            return None
        sample_x = x_samples[first:last + 1]
        measured_y = raw_centres[first:last + 1]
        measured_mask = np.isfinite(measured_y)
        if int(np.count_nonzero(measured_mask)) < int(p('min_path_points').value):
            return None
        centre_y = np.interp(sample_x, sample_x[measured_mask], measured_y[measured_mask])
        centre_y = smooth_median(centre_y, window=3)
        # Prevent an isolated bad estimate from producing an extreme lookahead point.
        centre_y = np.clip(centre_y, -3.0, 3.0)
        path = np.column_stack((sample_x, centre_y))
        return path

    # ------------------------------------------------------------------
    # Pure pursuit and curvature-based speed planning
    # ------------------------------------------------------------------
    @staticmethod
    def path_curvature(path):
        """Signed polyline curvature using points about 0.5 m apart."""
        n = len(path)
        curvature = np.zeros(n, dtype=float)
        if n < 5:
            return curvature
        stride = 2
        for i in range(stride, n - stride):
            a = path[i] - path[i - stride]
            b = path[i + stride] - path[i]
            c = path[i + stride] - path[i - stride]
            la = float(np.linalg.norm(a))
            lb = float(np.linalg.norm(b))
            lc = float(np.linalg.norm(c))
            denom = la * lb * lc
            if denom > 1e-5:
                cross = a[0] * b[1] - a[1] * b[0]
                curvature[i] = 2.0 * cross / denom
        # End samples have incomplete geometry; copy the nearest valid estimate.
        curvature[:stride] = curvature[stride]
        curvature[-stride:] = curvature[-stride - 1]
        # Median filtering prevents individual imperfect wall returns from causing braking spikes.
        curvature = smooth_median(curvature, window=7)
        return np.clip(curvature, -2.5, 2.5)

    def select_lookahead(self, path, speed, curvature):
        """Select a speed-based lookahead with only a mild tight-turn adjustment.

        The previous revision divided lookahead by (1 + 0.85 * curvature). On a
        sharp/noisy scan that regularly pushed the target down to the minimum,
        causing large, twitchy steering commands. Here curvature can shorten the
        nominal target by at most 10%; it can never force a very near target.
        """
        p = self.get_parameter
        base = float(p('lookahead_base').value)
        speed_gain = float(p('lookahead_speed_gain').value)
        distance = clamp(base + speed_gain * max(speed, 0.0),
                         float(p('lookahead_min').value),
                         float(p('lookahead_max').value))

        forward = np.linalg.norm(path, axis=1)
        relevant = np.flatnonzero(
            (forward >= 0.45) & (forward <= max(2.5, distance + 0.8)))
        if len(relevant):
            bend_strength = float(np.percentile(np.abs(curvature[relevant]), 75))
            bend_fraction = clamp(bend_strength / 1.2, 0.0, 1.0)
            distance *= 1.0 - 0.10 * bend_fraction
        distance = clamp(distance, float(p('lookahead_min').value),
                         float(p('lookahead_max').value))

        # Interpolate the exact lookahead-circle crossing instead of jumping between
        # 0.25 m path bins. This avoids small steering changes as the chosen beam/bin
        # flips between adjacent samples from scan to scan.
        euclidean = np.linalg.norm(path, axis=1)
        eligible = np.flatnonzero(euclidean >= distance)
        if len(eligible) == 0:
            target = path[-1].copy()
        else:
            idx = int(eligible[0])
            if idx == 0:
                target = path[0].copy()
            else:
                r0, r1 = float(euclidean[idx - 1]), float(euclidean[idx])
                denom = r1 - r0
                frac = 1.0 if abs(denom) < 1e-6 else clamp((distance - r0) / denom, 0.0, 1.0)
                target = path[idx - 1] + frac * (path[idx] - path[idx - 1])
        actual_distance = max(float(np.linalg.norm(target)), 0.1)
        return target, actual_distance

    def preview_curvature(self, path, curvature):
        """Estimate the direction/strength of a bend roughly 1-3 m ahead.

        Use a distance-weighted average instead of the single largest curvature
        sample. A sign-coherence check rejects inconsistent/noisy estimates, which
        otherwise could make the controller twitch left and right.
        """
        p = self.get_parameter
        preview_distance = float(p('turn_preview_distance').value)
        span = max(0.35, float(p('turn_preview_span').value))
        xs = path[:, 0]
        mask = (xs >= max(0.55, preview_distance - span)) & (xs <= preview_distance + span)
        if int(np.count_nonzero(mask)) < 4:
            return 0.0

        local_x = xs[mask]
        local_k = np.clip(np.asarray(curvature[mask], dtype=float), -1.8, 1.8)
        sigma = max(0.3, span * 0.65)
        weights = np.exp(-0.5 * np.square((local_x - preview_distance) / sigma))
        weight_sum = float(np.sum(weights))
        if weight_sum < 1e-8:
            return 0.0
        mean_k = float(np.sum(weights * local_k) / weight_sum)
        abs_mean = abs(mean_k)
        abs_weighted = float(np.sum(weights * np.abs(local_k)) / weight_sum)
        coherence = abs_mean / max(abs_weighted, 1e-6)

        if abs_mean < float(p('turn_preview_min_curvature').value) or coherence < 0.35:
            return 0.0
        return clamp(mean_k, -1.5, 1.5)

    def smooth_steering(self, raw_steer, dt):
        """Filter steering and limit per-second changes, consistently in every mode."""
        p = self.get_parameter
        raw_steer = clamp(float(raw_steer), -1.0, 1.0)
        previous = float(self.last_steer)
        smoothing = clamp(float(p('steer_smoothing').value), 0.0, 0.95)
        filtered = (1.0 - smoothing) * raw_steer + smoothing * previous
        rate = max(float(p('steer_rate_limit').value), 0.1)
        max_step = rate * clamp(float(dt), 0.005, 0.20)
        return previous + clamp(filtered - previous, -max_step, max_step)

    def planned_speed(self, path, curvature, speed):
        """Curvature speed caps plus backward braking-distance planning."""
        p = self.get_parameter
        k_throttle = max(float(p('speed_per_throttle').value), 1e-3)
        normal_top = float(p('max_throttle').value) * k_throttle
        straight_top = max(float(p('straight_throttle').value),
                           float(p('max_throttle').value)) * k_throttle
        lateral_accel = max(float(p('max_lateral_accel').value), 0.5)
        brake_decel = max(float(p('brake_decel').value), 0.5)

        # Distance along the estimated centreline from the rear axle to each sample.
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        along_path = np.concatenate(([float(np.linalg.norm(path[0]))],
                                     float(np.linalg.norm(path[0]))
                                     + np.cumsum(segment_lengths)))

        abs_k = np.abs(curvature)
        # Smoothly allow the higher straight throttle only when the path is actually straight.
        straight_fraction = np.exp(-np.square(abs_k / 0.075))
        top_at_point = normal_top + (straight_top - normal_top) * straight_fraction
        v_curve = np.sqrt(lateral_accel / np.maximum(abs_k, 0.02))
        curve_floor = float(p('min_curve_speed').value)
        speed_cap = np.minimum(top_at_point, np.maximum(v_curve, curve_floor))

        # Visible-distance rule. If the path stops well short of the sensor range, the
        # walls ended there (a corner, a wall ahead) and we do not know what comes next.
        # Require the car to be able to slow to end_speed by that point; the backward
        # braking calculation below turns this into an earlier, smoother slowdown.
        if float(path[-1, 0]) < self.view_limit - float(p('end_view_margin').value):
            speed_cap[-1] = min(float(speed_cap[-1]), float(p('end_speed').value))

        # Looking farther ahead, calculate the maximum speed from which the car could
        # still brake to each upcoming curve's limit. This moves braking before the bend.
        buffer = float(p('brake_dist').value) + speed * float(p('reaction_time').value)
        available = np.maximum(along_path - buffer, 0.0)
        allowed_now = np.sqrt(np.square(speed_cap) + 2.0 * brake_decel * available)
        target_speed = min(float(np.min(allowed_now)), straight_top)
        return max(0.0, target_speed), along_path, speed_cap

    def nearest_path_obstacle(self, x_all, y_all, hit, path):
        """Find LiDAR returns inside a narrow corridor around the estimated centreline."""
        if len(path) < 2:
            return None
        p = self.get_parameter
        x_min, x_max = float(path[0, 0]), float(path[-1, 0])
        valid = hit & (x_all > max(0.1, x_min - 0.25)) & (x_all < x_max)
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            return None
        centre_at_x = np.interp(x_all[indices], path[:, 0], path[:, 1])
        on_path = np.abs(y_all[indices] - centre_at_x) <= float(p('obstacle_half_width').value)
        candidates = indices[on_path]
        if len(candidates) == 0:
            return None
        return max(0.0, float(np.min(x_all[candidates])))

    # ------------------------------------------------------------------
    # Safety fallback: reactive gap steering if a stable centreline cannot be found.
    # ------------------------------------------------------------------
    def gap_fallback(self, scan):
        p = self.get_parameter
        ranges = np.asarray(scan.ranges, dtype=float)
        angles = float(scan.angle_min) + np.arange(len(ranges)) * float(scan.angle_increment)
        safe_ranges = np.where(np.isfinite(ranges), ranges, float(scan.range_max))
        safe_ranges = np.clip(safe_ranges, float(scan.range_min), float(scan.range_max))
        in_fov = np.abs(angles) <= math.radians(float(p('gap_fov_deg').value))
        ids = np.flatnonzero(in_fov)
        if len(ids) < 5:
            return 0.0, 0.0

        fov_ranges = safe_ranges[ids]
        fov_angles = angles[ids]
        smoothed = np.convolve(fov_ranges, np.ones(7) / 7.0, mode='same')
        min_idx = int(np.argmin(smoothed))
        min_distance = max(float(smoothed[min_idx]), 0.10)
        bubble_angle = math.asin(min(1.0, float(p('bubble').value) / min_distance))
        blocked = np.abs(fov_angles - fov_angles[min_idx]) <= bubble_angle
        free = (smoothed > float(p('gap_min_range').value)) & (~blocked)

        transitions = np.diff(np.concatenate(([0], free.astype(np.int8), [0])))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        if len(starts) == 0:
            turn_away = -1.0 if fov_angles[min_idx] > 0 else 1.0
            return turn_away, float(p('fallback_speed').value) * 0.45

        # Prefer a wide free sector; break ties by distance from obstacles and favour a
        # modest steering angle so a single distant return doesn't cause a violent swerve.
        best_score = -float('inf')
        target_angle = 0.0
        for start, end in zip(starts, ends):
            width = int(end - start)
            if width < 2:
                continue
            mid = (int(start) + int(end) - 1) // 2
            farthest = int(start) + int(np.argmax(smoothed[start:end]))
            target_idx = (mid + farthest) // 2
            score = width + 0.12 * float(smoothed[target_idx]) - 2.0 * abs(float(fov_angles[target_idx]))
            if score > best_score:
                best_score = score
                target_angle = float(fov_angles[target_idx])
        steer = clamp(target_angle / max(float(p('max_steer_rad').value), 1e-3)
                      * float(p('gap_gain').value), -1.0, 1.0)
        # Keep the speed low whenever the centreline inference is unreliable.
        fallback_speed = float(p('fallback_speed').value)
        if min_distance < 1.0:
            fallback_speed = min(fallback_speed, 1.5)
        return steer, fallback_speed

    # ------------------------------------------------------------------
    # Output and main scan callback
    # ------------------------------------------------------------------
    def publish_commands(self, steer, throttle):
        # clamp() turns NaN into +1.0 (full lock / full throttle), so never let one through.
        if not (math.isfinite(float(steer)) and math.isfinite(float(throttle))):
            steer = self.last_steer
            throttle = -float(self.get_parameter('max_brake').value)
        steer = clamp(float(steer), -1.0, 1.0)
        throttle = clamp(float(throttle), -1.0, 1.0)
        self.pub_steer.publish(Float32(data=steer))
        self.pub_throttle.publish(Float32(data=throttle))
        self.last_steer = steer
        self.last_throttle = throttle

    def on_scan(self, scan):
        """Run the controller; if anything goes wrong, brake instead of crashing the node."""
        try:
            self.process_scan(scan)
        except Exception as exc:  # noqa: BLE001 - a dead node leaves the car on its last command
            self.error_count += 1
            if self.error_count <= 5 or self.error_count % 50 == 0:
                self.get_logger().error(f'scan processing failed ({exc!r}); braking '
                                        f'(failure #{self.error_count})')
            try:
                steer, _ = self.gap_fallback(scan)
                steer = self.smooth_steering(steer, 0.03)
            except Exception:  # noqa: BLE001
                steer = self.last_steer
            self.publish_commands(steer, -float(self.get_parameter('max_brake').value) * 0.6)

    def process_scan(self, scan):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_scan_time is None:
            dt = 0.025
        else:
            dt = clamp(now - self.prev_scan_time, 0.005, 0.20)
        self.prev_scan_time = now
        self.scan_count += 1
        p = self.get_parameter

        if len(scan.ranges) < 10 or abs(float(scan.angle_increment)) < 1e-9:
            self.publish_commands(0.0, -float(p('max_brake').value))
            return

        speed, speed_source = self.current_speed(now, dt)
        x_all, y_all, angles, hit = self.scan_xy(scan, float(p('lidar_offset_x').value))
        path = self.estimate_centerline(scan, x_all, y_all, angles, hit)

        if path is None or len(path) < 5:
            self.fallback_count += 1
            raw_steer, fallback_speed = self.gap_fallback(scan)
            steer = self.smooth_steering(raw_steer, dt)
            target_speed = min(fallback_speed, float(p('fallback_speed').value))
            if speed > target_speed + float(p('brake_margin').value) and speed > 1.0:
                excess = speed - target_speed
                throttle = -min(float(p('max_brake').value),
                                max(0.18, float(p('brake_gain').value) * excess))
            else:
                throttle = clamp(target_speed / max(float(p('speed_per_throttle').value), 1e-3),
                                 float(p('min_throttle').value),
                                 float(p('max_throttle').value))
            self.publish_commands(steer, throttle)
            self.maybe_log(now, mode='gap-fallback', speed=speed, speed_source=speed_source,
                           target_speed=target_speed, target=None, curvature=0.0,
                           path_points=0, obstacle=None)
            return

        curvature = self.path_curvature(path)
        target, lookahead_distance = self.select_lookahead(path, speed, curvature)
        target_x, target_y = float(target[0]), float(target[1])

        # Pure pursuit: curvature = 2*y / distance^2; delta = atan(wheelbase*curvature).
        # Add a bounded curvature preview so steering starts before the pure-pursuit
        # target has moved deep into the corner. The preview fades as PP catches up,
        # and is ignored when its sign contradicts a clear PP turn (often a noisy path).
        target_distance_sq = max(target_x * target_x + target_y * target_y, 1e-4)
        pp_curvature = 2.0 * target_y / target_distance_sq
        upcoming_curvature = self.preview_curvature(path, curvature)
        preview_gain = clamp(float(p('turn_preview_gain').value), 0.0, 0.6)
        if abs(pp_curvature) > 0.10 and pp_curvature * upcoming_curvature < 0.0:
            preview_term = 0.0
        else:
            # Taper the preview smoothly to zero once pursuit is already steering
            # decisively. That avoids adding the same corner curvature twice.
            fade = clamp(1.0 - abs(pp_curvature) / 0.65, 0.0, 1.0)
            preview_term = preview_gain * fade * upcoming_curvature
        commanded_curvature = clamp(pp_curvature + preview_term, -2.5, 2.5)
        steering_angle = math.atan(float(p('wheelbase').value) * commanded_curvature)
        raw_steer = clamp(steering_angle / max(float(p('max_steer_rad').value), 1e-3), -1.0, 1.0)
        steer = self.smooth_steering(raw_steer, dt)

        target_speed, along_path, speed_caps = self.planned_speed(path, curvature, speed)
        obstacle_distance = self.nearest_path_obstacle(x_all, y_all, hit, path)
        if obstacle_distance is not None:
            # Stop before a detected object that lies in the vehicle's projected path.
            available = max(obstacle_distance
                            - float(p('obstacle_safety_distance').value)
                            - speed * float(p('reaction_time').value), 0.0)
            obstacle_speed = math.sqrt(2.0 * max(float(p('brake_decel').value), 0.5) * available)
            target_speed = min(target_speed, obstacle_speed)

        target_speed = max(0.0, target_speed)
        speed_error = speed - target_speed
        if speed_error > float(p('brake_margin').value) and speed > 0.8:
            brake = float(p('brake_gain').value) * speed_error
            # Use strong braking if the planned stop/turn speed is far below current speed.
            if speed_error > 2.0 or target_speed < 1.0:
                brake = max(brake, 0.75 * float(p('max_brake').value))
            throttle = -clamp(brake, 0.0, float(p('max_brake').value))
        elif target_speed < 0.7:
            throttle = 0.0
        else:
            k_throttle = max(float(p('speed_per_throttle').value), 1e-3)
            throttle = target_speed / k_throttle
            # Small acceleration nudge while below the planned speed; it fades away near target.
            if target_speed - speed > 0.8:
                throttle += 0.035
            throttle = clamp(throttle, float(p('min_throttle').value),
                             max(float(p('straight_throttle').value),
                                 float(p('max_throttle').value)))

        self.publish_commands(steer, throttle)
        self.maybe_log(now, mode='pure-pursuit', speed=speed, speed_source=speed_source,
                       target_speed=target_speed, target=(target_x, target_y),
                       curvature=commanded_curvature, path_points=len(path),
                       obstacle=obstacle_distance, preview=upcoming_curvature)

    def maybe_log(self, now, mode, speed, speed_source, target_speed,
                  target, curvature, path_points, obstacle, preview=0.0):
        period = float(self.get_parameter('log_period').value)
        if period <= 0.0:
            return
        if self.last_log_time is not None and now - self.last_log_time < period:
            return
        self.last_log_time = now
        if target is None:
            target_text = 'n/a'
        else:
            target_text = f'({target[0]:.2f},{target[1]:+.2f}) m'
        obstacle_text = 'none' if obstacle is None else f'{obstacle:.2f} m'
        self.get_logger().info(
            f'{mode} | v={speed:.2f} m/s ({speed_source}) target_v={target_speed:.2f} | '
            f'target={target_text} kappa={curvature:+.3f} 1/m '
            f'preview_k={preview:+.3f} 1/m | '
            f'path_points={path_points} obstacle={obstacle_text} | '
            f'steer={self.last_steer:+.3f} throttle={self.last_throttle:+.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()