import time
from pygame.math import Vector2

class RewardSystem:
    def __init__(self):
        self.rewards = {
            'speed_multiplier': 0.45,
            'lap_completion_base': 130,
            'lap_completion_time_penalty': -0.5,
            'lap_completion_min': 30,
            'checkpoint_progress': 5.0,
            'death_penalty_0_checkpoints': -50,
            'death_penalty_50_checkpoints': -10,
            'timeout_penalty': -30,
            'certainty_penalty': -0.1
        }

        self.last_checkpoint_distance = float('inf')
        self.closest_checkpoint_distance = float('inf')
        self.checkpoints_passed_total = 0
        self.last_checkpoint_time = time.time()
        self.timeout_duration = 7.0  # seconds
        self.just_passed_checkpoint = False

    def reset_episode(self):
        self.last_checkpoint_distance = float('inf')
        self.closest_checkpoint_distance = float('inf')
        self.checkpoints_passed_total = 0
        self.last_checkpoint_time = time.time()
        self.just_passed_checkpoint = False

    def reset_after_death(self):
        """Keep total checkpoint count but reset per-episode distance tracking."""
        self.last_checkpoint_distance = float('inf')
        self.closest_checkpoint_distance = float('inf')
        self.last_checkpoint_time = time.time()
        self.just_passed_checkpoint = False

    def checkpoint_passed(self):
        self.checkpoints_passed_total += 1
        self.last_checkpoint_time = time.time()
        self.just_passed_checkpoint = True
        self.last_checkpoint_distance = float('inf')
        self.closest_checkpoint_distance = float('inf')

    def calculate_reward(self, car, track, cp_passed, prob_distribution, lap_time=None, died=False):
        total_reward = 0.0

        total_reward += car.vel * self.rewards['speed_multiplier']

        if lap_time is not None:
            lap_reward = max(
                self.rewards['lap_completion_min'],
                self.rewards['lap_completion_base'] + self.rewards['lap_completion_time_penalty'] * lap_time
            )
            total_reward += lap_reward

        if not died and track.checkpoints:
            total_reward += self._calculate_checkpoint_progress_reward(car, track, cp_passed)

        if died:
            total_reward += self._calculate_death_penalty()

        # penalise being very confident about one turning direction vs the other
        if max([prob_distribution[4], prob_distribution[2], prob_distribution[6]]) - max([prob_distribution[3], prob_distribution[5], prob_distribution[7]]) > 0.4:
            total_reward += self.rewards['certainty_penalty']

        self.just_passed_checkpoint = False
        return total_reward

    def _calculate_checkpoint_progress_reward(self, car, track, cp_passed):
        try:
            next_checkpoint_idx = cp_passed.index(False)
            checkpoint_line = track.checkpoints[next_checkpoint_idx]

            car_pos = Vector2(car.x, car.y)
            checkpoint_start = Vector2(checkpoint_line[0])
            checkpoint_end = Vector2(checkpoint_line[1])
            current_distance = self._distance_to_line_segment(car_pos, checkpoint_start, checkpoint_end)

            # skip the frame right after passing a checkpoint to avoid distance-reset artifacts
            if self.just_passed_checkpoint:
                self.last_checkpoint_distance = current_distance
                self.closest_checkpoint_distance = current_distance
                return 0.0

            reward = 0.0
            if current_distance < self.closest_checkpoint_distance:
                if self.closest_checkpoint_distance == float('inf'):
                    self.closest_checkpoint_distance = current_distance
                else:
                    improvement = self.closest_checkpoint_distance - current_distance
                    reward = improvement * self.rewards['checkpoint_progress'] / 100.0
                    self.closest_checkpoint_distance = current_distance
            return reward

        except ValueError:
            return 0.0

    def _distance_to_line_segment(self, point, line_start, line_end):
        if line_start == line_end:
            return (point - line_start).length()
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len_sq = line_vec.length_squared()
        if line_len_sq == 0:
            return (point - line_start).length()
        t = max(0, min(1, point_vec.dot(line_vec) / line_len_sq))
        projection = line_start + t * line_vec
        return (point - projection).length()

    def _calculate_death_penalty(self):
        # linearly interpolate penalty: harsh early on, lighter once the AI knows the track
        if self.checkpoints_passed_total == 0:
            return self.rewards['death_penalty_0_checkpoints']
        elif self.checkpoints_passed_total >= 50:
            return self.rewards['death_penalty_50_checkpoints']
        else:
            ratio = self.checkpoints_passed_total / 50.0
            p0 = self.rewards['death_penalty_0_checkpoints']
            p50 = self.rewards['death_penalty_50_checkpoints']
            return p0 + ratio * (p50 - p0)

    def update_reward_parameter(self, parameter_name, value):
        if parameter_name in self.rewards:
            self.rewards[parameter_name] = value
        else:
            print(f"Warning: '{parameter_name}' not found in rewards dict")

    def get_reward_parameters(self):
        return self.rewards.copy()
