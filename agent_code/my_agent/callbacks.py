from collections import defaultdict, deque
import os
import pickle
import random

import numpy as np

from .features import (
    FEATURE_NAMES,
    LEGACY_FEATURE_COUNT,
    STAGE2_FEATURE_COUNT,
    STAGE3_FEATURE_COUNT,
    MOVE_DELTAS,
    bomb_enemy_count,
    bomb_escape_route_count,
    danger_map,
    legal_actions,
    local_exit_count,
    nearby_enemy_count,
    potential_enemy_bomb_risk_count,
    nearest_attack_target_info,
    nearest_coin_info,
    nearest_crate_target_info,
    nearest_enemy_approach_info,
    state_to_features,
)
from .model import new_model, q_values


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = 'my-saved-model.pt'


def load_or_migrate_model(candidate, logger):
    if not isinstance(candidate, dict) or 'weights' not in candidate:
        return None

    weights = candidate['weights']

    # Already Stage 4.
    if weights.shape == (len(ACTIONS), len(FEATURE_NAMES)):
        candidate['stage'] = 4
        candidate['variant'] = 'v2'
        return candidate

    # Stage 3 -> Stage 4. Preserve all 50 learned features and initialize
    # the 14 new multiplayer-survival weights to zero.
    if weights.shape == (len(ACTIONS), STAGE3_FEATURE_COUNT):
        migrated = new_model(len(ACTIONS), len(FEATURE_NAMES))
        migrated['weights'][:, :STAGE3_FEATURE_COUNT] = weights
        migrated['episodes'] = int(candidate.get('episodes', 0))
        migrated['epsilon'] = max(float(candidate.get('epsilon', 1.0)), 0.28)
        migrated['stage'] = 4
        migrated['variant'] = 'v2'

        logger.info(
            f"Migrated Stage-3 checkpoint ({STAGE3_FEATURE_COUNT} features) "
            f"to Stage 4 ({len(FEATURE_NAMES)} features)."
        )
        return migrated

    # Stage 2 -> Stage 4 fallback.
    if weights.shape == (len(ACTIONS), STAGE2_FEATURE_COUNT):
        migrated = new_model(len(ACTIONS), len(FEATURE_NAMES))
        migrated['weights'][:, :STAGE2_FEATURE_COUNT] = weights
        migrated['episodes'] = int(candidate.get('episodes', 0))
        migrated['epsilon'] = max(float(candidate.get('epsilon', 1.0)), 0.35)
        migrated['stage'] = 4
        migrated['variant'] = 'v2'

        logger.info(
            f"Migrated Stage-2 checkpoint ({STAGE2_FEATURE_COUNT} features) "
            f"directly to Stage 4 ({len(FEATURE_NAMES)} features)."
        )
        return migrated

    # Stage 1 -> Stage 4 fallback. Normally do not use this path.
    if weights.shape == (len(ACTIONS), LEGACY_FEATURE_COUNT):
        migrated = new_model(len(ACTIONS), len(FEATURE_NAMES))
        for action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
            i = ACTIONS.index(action)
            migrated['weights'][i, :LEGACY_FEATURE_COUNT] = (
                weights[i, :LEGACY_FEATURE_COUNT]
            )

        migrated['weights'][ACTIONS.index('WAIT')] = 0.0
        migrated['weights'][ACTIONS.index('BOMB')] = 0.0
        migrated['episodes'] = int(candidate.get('episodes', 0))
        migrated['epsilon'] = max(float(candidate.get('epsilon', 1.0)), 0.45)
        migrated['stage'] = 4
        migrated['variant'] = 'v2'

        logger.info(
            f"Migrated Stage-1 checkpoint ({LEGACY_FEATURE_COUNT} features) "
            f"directly to Stage 4 ({len(FEATURE_NAMES)} features)."
        )
        return migrated

    return None


def setup(self):
    self.model = None

    if os.path.isfile(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as file:
                candidate = pickle.load(file)
            self.model = load_or_migrate_model(candidate, self.logger)
            if self.model is not None:
                self.logger.info(
                    f"Loaded model after {self.model.get('episodes', 0)} training rounds."
                )
        except Exception as exception:
            self.logger.warning(f"Could not load checkpoint: {exception}")

    if self.model is None:
        self.model = new_model(len(ACTIONS), len(FEATURE_NAMES))
        self.model['stage'] = 4
        self.model['variant'] = 'v2'
        self.logger.info("Created fresh Stage-4 v2 Approximate Q-Learning model.")

    self.epsilon = float(self.model.get('epsilon', 1.0)) if self.train else 0.0

    self.recent_positions = deque(maxlen=24)
    self.position_visit_count = defaultdict(int)
    self.last_round_seen = None


def _action_toward_step(current_position, first_step):
    if first_step is None or tuple(first_step) == tuple(current_position):
        return None

    dx = first_step[0] - current_position[0]
    dy = first_step[1] - current_position[1]
    mapping = {
        (0, -1): 'UP',
        (1, 0): 'RIGHT',
        (0, 1): 'DOWN',
        (-1, 0): 'LEFT',
    }
    return mapping.get((dx, dy))


def apply_progress_bonus(game_state, valid, masked_values):
    """
    Stage-4 v2 objective priority:
      1. safe attack opportunity / attack position
      2. visible coin
      3. approach opponent if no attack square is reachable yet
      4. safe crate-bomb position

    v2 restores some of the Stage-3 offensive pressure that was lost in the
    first Stage-4 survival-heavy version. These remain SOFT Q-value bonuses.
    """
    current_position = tuple(game_state['self'][3])

    # 1) Combat first while an opponent is alive.
    if game_state['others']:
        attack_step, attack_distance = nearest_attack_target_info(game_state)

        if attack_distance == 0 and 'BOMB' in valid and bomb_enemy_count(game_state) > 0:
            masked_values[ACTIONS.index('BOMB')] += 20.0
            return masked_values

        attack_action = _action_toward_step(current_position, attack_step)
        if attack_action in valid:
            masked_values[ACTIONS.index(attack_action)] += 16.0
            return masked_values

    # 2) Visible coin.
    coin_step, coin_distance = nearest_coin_info(game_state)
    if coin_distance is not None:
        target_action = _action_toward_step(current_position, coin_step)
        if target_action in valid:
            masked_values[ACTIONS.index(target_action)] += 10.0
        return masked_values

    # 3) Approach the opponent if no safe attack square is reachable yet.
    if game_state['others']:
        approach_step, approach_distance = nearest_enemy_approach_info(game_state)
        if approach_distance is not None:
            target_action = _action_toward_step(current_position, approach_step)
            if target_action in valid:
                masked_values[ACTIONS.index(target_action)] += 7.0
                return masked_values

    # 4) Continue useful crate clearing.
    crate_step, crate_distance = nearest_crate_target_info(game_state)
    if crate_distance == 0:
        if 'BOMB' in valid:
            masked_values[ACTIONS.index('BOMB')] += 10.0
        return masked_values

    target_action = _action_toward_step(current_position, crate_step)
    if target_action in valid:
        masked_values[ACTIONS.index(target_action)] += 8.0

    return masked_values


def apply_survival_adjustments(game_state, valid, masked_values):
    """
    Balanced Stage-4 v2 survival preferences.

    Known explosions are still handled as hard constraints by legal_actions().
    These are deliberately softer than v1 so a SINGLE nearby opponent does not
    automatically push the agent away from useful attacking positions.
    """
    current_position = tuple(game_state['self'][3])
    danger = danger_map(game_state)
    urgent_known_danger = (
        float(danger[current_position]) <= 2
        or game_state['explosion_map'][current_position] > 0
    )

    nearby_count = nearby_enemy_count(game_state, radius=3)
    bomb_busy = len(game_state['bombs']) > 0

    for action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        if action not in valid:
            continue

        dx, dy = MOVE_DELTAS[action]
        next_position = (
            current_position[0] + dx,
            current_position[1] + dy,
        )
        i = ACTIONS.index(action)

        exits = local_exit_count(game_state, next_position)
        enemy_risk = potential_enemy_bomb_risk_count(
            game_state,
            next_position,
        )

        # v1 punished any one-exit tile whenever even one enemy was nearby.
        # v2 reserves the strong penalty for genuinely pressured states.
        high_pressure = (
            bomb_busy
            or nearby_count >= 2
            or enemy_risk >= 2
        )

        if exits <= 1:
            if high_pressure:
                masked_values[i] -= 5.0
            elif nearby_count > 0:
                masked_values[i] -= 1.5
        elif exits >= 3:
            masked_values[i] += 0.75
        elif exits == 2:
            masked_values[i] += 0.25

        # Potential enemy bombs are uncertain. Keep them as a soft warning,
        # not a reason to abandon every attack lane.
        if not urgent_known_danger:
            masked_values[i] -= 1.5 * enemy_risk

    # Redundant bomb escapes are still good, but are not over-rewarded.
    if 'BOMB' in valid:
        routes = bomb_escape_route_count(game_state)
        bomb_i = ACTIONS.index('BOMB')
        if routes >= 3:
            masked_values[bomb_i] += 2.0
        elif routes == 2:
            masked_values[bomb_i] += 1.0
        elif routes == 1:
            masked_values[bomb_i] += 0.25

    return masked_values


def apply_loop_penalties(self, game_state, valid, masked_values):
    """Suppress repeated paths while leaving emergency escape untouched."""
    current_position = tuple(game_state['self'][3])
    danger = danger_map(game_state)

    urgent_danger = (
        float(danger[current_position]) <= 2
        or game_state['explosion_map'][current_position] > 0
    )
    if urgent_danger:
        return masked_values

    history = list(self.recent_positions)

    for action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        if action not in valid:
            continue

        dx, dy = MOVE_DELTAS[action]
        next_position = (current_position[0] + dx, current_position[1] + dy)
        action_index = ACTIONS.index(action)
        visit_count = self.position_visit_count[next_position]
        recent_count = history.count(next_position)

        masked_values[action_index] -= min(4.0 * visit_count, 20.0)
        masked_values[action_index] -= min(2.0 * recent_count, 8.0)

        if len(history) >= 2 and next_position == history[-2]:
            masked_values[action_index] -= 8.0

    return masked_values


def choose_exploration_action(self, game_state, valid):
    current_position = tuple(game_state['self'][3])
    enemies_hit = bomb_enemy_count(game_state)
    routes = bomb_escape_route_count(game_state) if 'BOMB' in valid else 0

    weights = []
    for action in valid:
        if action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
            dx, dy = MOVE_DELTAS[action]
            next_position = (
                current_position[0] + dx,
                current_position[1] + dy,
            )
            visits = self.position_visit_count[next_position]
            exits = local_exit_count(game_state, next_position)
            enemy_risk = potential_enemy_bomb_risk_count(
                game_state,
                next_position,
            )

            weight = 1.0 / (1.0 + 2.0 * visits)
            weight *= 0.75 + 0.15 * exits
            weight /= 1.0 + 0.75 * enemy_risk
            weights.append(max(weight, 0.03))

        elif action == 'BOMB':
            # Restore attack exploration while keeping crate-only bombing lower.
            weight = 1.20 if enemies_hit > 0 else 0.45
            if routes >= 2:
                weight *= 1.15
            weights.append(weight)

        else:  # WAIT
            weights.append(0.03)

    return random.choices(valid, weights=weights, k=1)[0]


def act(self, game_state: dict) -> str:
    current_round = game_state['round']
    if self.last_round_seen != current_round:
        self.recent_positions.clear()
        self.position_visit_count.clear()
        self.last_round_seen = current_round

    current_position = tuple(game_state['self'][3])
    self.position_visit_count[current_position] += 1

    if not self.recent_positions or self.recent_positions[-1] != current_position:
        self.recent_positions.append(current_position)

    valid = legal_actions(game_state)

    if self.train and random.random() < self.epsilon:
        return choose_exploration_action(self, game_state, valid)

    features = state_to_features(game_state)
    values = q_values(self.model, features)

    masked_values = np.full(len(ACTIONS), -np.inf, dtype=np.float32)
    for action in valid:
        i = ACTIONS.index(action)
        masked_values[i] = values[i]

    masked_values = apply_progress_bonus(game_state, valid, masked_values)
    masked_values = apply_survival_adjustments(game_state, valid, masked_values)
    masked_values = apply_loop_penalties(self, game_state, valid, masked_values)

    best_value = np.max(masked_values)
    best_indices = np.flatnonzero(np.isclose(masked_values, best_value))

    movement_indices = [
        i for i in best_indices
        if ACTIONS[i] in {'UP', 'RIGHT', 'DOWN', 'LEFT'}
    ]

    if movement_indices:
        best_index = random.choice(movement_indices)
    else:
        best_index = int(random.choice(best_indices.tolist()))

    return ACTIONS[best_index]
