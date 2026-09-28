from collections import deque
import pickle
from typing import List

import numpy as np
import events as e

from .callbacks import ACTIONS, MODEL_FILE
from .features import (
    _state_with_self_position,
    bomb_crate_count,
    bomb_enemy_count,
    bomb_escape_route_count,
    can_escape_after_bomb,
    danger_map,
    legal_actions,
    local_exit_count,
    nearby_enemy_count,
    potential_enemy_bomb_risk_count,
    nearest_attack_distance,
    nearest_coin_distance,
    nearest_crate_distance,
    state_to_features,
)
from .model import q_values, update_q


ALPHA = 0.03
GAMMA = 0.95
EPSILON_MIN = 0.04
EPSILON_DECAY = 0.998

MOVED_TOWARD_COIN = 'MOVED_TOWARD_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
MOVED_TOWARD_CRATE = 'MOVED_TOWARD_CRATE'
MOVED_AWAY_FROM_CRATE = 'MOVED_AWAY_FROM_CRATE'
MOVED_TOWARD_ATTACK = 'MOVED_TOWARD_ATTACK'
MOVED_AWAY_FROM_ATTACK = 'MOVED_AWAY_FROM_ATTACK'
ESCAPED_DANGER = 'ESCAPED_DANGER'
ENTERED_DANGER = 'ENTERED_DANGER'
GOOD_BOMB = 'GOOD_BOMB'
GOOD_ATTACK_BOMB = 'GOOD_ATTACK_BOMB'
USELESS_BOMB = 'USELESS_BOMB'
OSCILLATION = 'OSCILLATION'
REVISITED_TILE = 'REVISITED_TILE'
HEAVY_LOOP = 'HEAVY_LOOP'
UNNECESSARY_WAIT = 'UNNECESSARY_WAIT'
ENTERED_DEAD_END = 'ENTERED_DEAD_END'
LEFT_DEAD_END = 'LEFT_DEAD_END'
MOVED_TO_OPEN_SPACE = 'MOVED_TO_OPEN_SPACE'
MOVED_INTO_ENEMY_BOMB_RISK = 'MOVED_INTO_ENEMY_BOMB_RISK'
SAFE_MULTI_ESCAPE_BOMB = 'SAFE_MULTI_ESCAPE_BOMB'
RISKY_LOW_ESCAPE_BOMB = 'RISKY_LOW_ESCAPE_BOMB'


def setup_training(self):
    self.epsilon = max(
        EPSILON_MIN,
        float(self.model.get('epsilon', 1.0)),
    )
    self.round_reward = 0.0
    self.round_td_errors = []
    self.position_history = deque(maxlen=40)


def current_danger_timer(game_state):
    if game_state is None:
        return None

    x, y = game_state['self'][3]
    danger = danger_map(game_state)

    if game_state['explosion_map'][x, y] > 0:
        return 0.0

    value = float(danger[x, y])
    return None if value >= 5.0 else value


def _distance_after_our_move(old_game_state, new_position, distance_function):
    """
    Measure progress while holding opponents/board from the OLD state fixed.
    This avoids incorrectly rewarding/penalizing our action just because an
    opponent moved at the same time.
    """
    hypothetical = _state_with_self_position(old_game_state, new_position)
    return distance_function(hypothetical)


def add_custom_events(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or new_game_state is None:
        return

    old_position = tuple(old_game_state['self'][3])
    new_position = tuple(new_game_state['self'][3])

    if not self.position_history:
        self.position_history.append(old_position)

    recent_positions = list(self.position_history)

    # Immediate A -> B -> A oscillation.
    if len(recent_positions) >= 2 and new_position == recent_positions[-2]:
        events.append(OSCILLATION)

    # Larger loops.
    visit_count = recent_positions.count(new_position)
    if visit_count >= 1:
        events.append(REVISITED_TILE)
    if visit_count >= 3:
        events.append(HEAVY_LOOP)

    self.position_history.append(new_position)

    moved = self_action in {'UP', 'RIGHT', 'DOWN', 'LEFT'}

    # ========================================================
    # STAGE-4 MULTI-AGENT SURVIVAL SHAPING
    # ========================================================
    if moved:
        hypothetical_after_move = _state_with_self_position(
            old_game_state,
            new_position,
        )

        old_exits = local_exit_count(
            old_game_state,
            old_position,
        )
        new_exits = local_exit_count(
            hypothetical_after_move,
            new_position,
        )

        survival_pressure = (
            len(old_game_state['bombs']) > 0
            or nearby_enemy_count(old_game_state, radius=3) >= 2
        )

        if survival_pressure and old_exits > 1 and new_exits <= 1:
            events.append(ENTERED_DEAD_END)
        elif old_exits <= 1 and new_exits > 1:
            events.append(LEFT_DEAD_END)

        if new_exits >= 3 and new_exits > old_exits:
            events.append(MOVED_TO_OPEN_SPACE)

        old_enemy_risk = potential_enemy_bomb_risk_count(
            old_game_state,
            old_position,
        )
        new_enemy_risk = potential_enemy_bomb_risk_count(
            old_game_state,
            new_position,
        )

        old_known_danger = current_danger_timer(old_game_state)
        urgent_known_danger = (
            old_known_danger is not None
            and old_known_danger <= 2
        )

        if (
            not urgent_known_danger
            and new_enemy_risk > old_enemy_risk
        ):
            events.append(MOVED_INTO_ENEMY_BOMB_RISK)

    # ========================================================
    # COMBAT PROGRESS
    # ========================================================
    # Reward movement toward a SAFE attack position. Compare against a
    # hypothetical old state with only our own position changed so opponent
    # motion does not contaminate the reward.
    if moved and old_game_state['others']:
        old_attack_distance = nearest_attack_distance(old_game_state)
        moved_attack_distance = _distance_after_our_move(
            old_game_state,
            new_position,
            nearest_attack_distance,
        )

        if old_attack_distance is not None and moved_attack_distance is not None:
            if moved_attack_distance < old_attack_distance:
                events.append(MOVED_TOWARD_ATTACK)
            elif moved_attack_distance > old_attack_distance:
                events.append(MOVED_AWAY_FROM_ATTACK)

    # ========================================================
    # COIN PROGRESS
    # ========================================================
    if e.COIN_COLLECTED not in events:
        old_coin_distance = nearest_coin_distance(old_game_state)
        new_coin_distance = nearest_coin_distance(new_game_state)

        if old_coin_distance is not None and new_coin_distance is not None:
            if new_coin_distance < old_coin_distance:
                events.append(MOVED_TOWARD_COIN)
            elif new_coin_distance > old_coin_distance:
                events.append(MOVED_AWAY_FROM_COIN)

    # ========================================================
    # CRATE PROGRESS
    # ========================================================
    # Crate shaping remains useful when attack positions are not reachable yet.
    if (
        len(old_game_state['coins']) == 0
        and e.CRATE_DESTROYED not in events
        and e.COIN_FOUND not in events
        and self_action != 'BOMB'
    ):
        old_crate_distance = nearest_crate_distance(old_game_state)
        new_crate_distance = nearest_crate_distance(new_game_state)

        if old_crate_distance is not None and new_crate_distance is not None:
            if new_crate_distance < old_crate_distance:
                events.append(MOVED_TOWARD_CRATE)
            elif new_crate_distance > old_crate_distance:
                events.append(MOVED_AWAY_FROM_CRATE)

    # ========================================================
    # DANGER
    # ========================================================
    old_danger = current_danger_timer(old_game_state)
    new_danger = current_danger_timer(new_game_state)

    old_urgent = old_danger is not None and old_danger <= 2
    new_urgent = new_danger is not None and new_danger <= 2

    if old_urgent and not new_urgent:
        events.append(ESCAPED_DANGER)
    elif not old_urgent and new_urgent:
        events.append(ENTERED_DANGER)

    # ========================================================
    # BOMB QUALITY
    # ========================================================
    if self_action == 'BOMB':
        crates_hit = bomb_crate_count(old_game_state)
        enemies_hit = bomb_enemy_count(old_game_state)
        safe = can_escape_after_bomb(old_game_state)
        escape_routes = bomb_escape_route_count(old_game_state)

        if safe and enemies_hit > 0:
            events.append(GOOD_ATTACK_BOMB)

        if safe and crates_hit > 0:
            events.append(GOOD_BOMB)

        if escape_routes >= 2:
            events.append(SAFE_MULTI_ESCAPE_BOMB)

        current_position = tuple(old_game_state['self'][3])
        enemy_bomb_risk_here = potential_enemy_bomb_risk_count(
            old_game_state,
            current_position,
        )
        high_pressure = (
            nearby_enemy_count(old_game_state, radius=3) >= 2
            or (
                len(old_game_state['bombs']) > 0
                and enemy_bomb_risk_here > 0
            )
        )
        if high_pressure and escape_routes < 2:
            events.append(RISKY_LOW_ESCAPE_BOMB)

        if not safe or (crates_hit == 0 and enemies_hit == 0):
            events.append(USELESS_BOMB)

    # WAIT should be rare if safe movement exists.
    if self_action == 'WAIT':
        old_danger = current_danger_timer(old_game_state)
        available = legal_actions(old_game_state)
        movement_available = any(
            action in available for action in ['UP', 'RIGHT', 'DOWN', 'LEFT']
        )
        if old_danger is None and movement_available:
            events.append(UNNECESSARY_WAIT)


def learn(self, old_game_state, action, new_game_state, reward, terminal=False):
    state = state_to_features(old_game_state)
    if state is None or action not in ACTIONS:
        return

    action_index = ACTIONS.index(action)

    if terminal or new_game_state is None:
        target = reward
    else:
        next_state = state_to_features(new_game_state)
        next_values = q_values(self.model, next_state)
        next_valid_actions = legal_actions(new_game_state)

        best_next_value = max(
            float(next_values[ACTIONS.index(next_action)])
            for next_action in next_valid_actions
        )
        target = reward + GAMMA * best_next_value

    td_error = update_q(
        self.model,
        action_index,
        state,
        target,
        ALPHA,
    )
    self.round_td_errors.append(abs(td_error))


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    add_custom_events(
        self,
        old_game_state,
        self_action,
        new_game_state,
        events,
    )

    reward = reward_from_events(self, events)
    self.round_reward += reward

    learn(
        self,
        old_game_state,
        self_action,
        new_game_state,
        reward,
        terminal=False,
    )


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    reward = reward_from_events(self, events)
    self.round_reward += reward

    learn(
        self,
        last_game_state,
        last_action,
        None,
        reward,
        terminal=True,
    )

    self.model['episodes'] = int(self.model.get('episodes', 0)) + 1
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.model['epsilon'] = self.epsilon
    self.model['stage'] = 4
    self.model['variant'] = 'v2'

    mean_td_error = (
        float(np.mean(self.round_td_errors))
        if self.round_td_errors
        else 0.0
    )

    self.logger.info(
        f"Stage-4 v2 round {self.model['episodes']} | "
        f"reward={self.round_reward:.2f} | "
        f"epsilon={self.epsilon:.3f} | "
        f"mean_abs_td={mean_td_error:.3f}"
    )

    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(self.model, file)

    self.round_reward = 0.0
    self.round_td_errors = []
    self.position_history.clear()


def reward_from_events(self, events: List[str]) -> float:
    game_rewards = {
        # Real objectives.
        e.COIN_COLLECTED: 20.0,
        e.COIN_FOUND: 5.0,
        e.CRATE_DESTROYED: 3.0,
        e.KILLED_OPPONENT: 80.0,

        # Combat shaping.
        MOVED_TOWARD_ATTACK: 3.0,
        MOVED_AWAY_FROM_ATTACK: -1.75,
        GOOD_ATTACK_BOMB: 10.0,

        # Existing navigation shaping.
        MOVED_TOWARD_COIN: 1.5,
        MOVED_AWAY_FROM_COIN: -2.0,
        MOVED_TOWARD_CRATE: 1.0,
        MOVED_AWAY_FROM_CRATE: -1.2,

        # Crate-only bomb reward stays intentionally small.
        GOOD_BOMB: 1.0,
        USELESS_BOMB: -8.0,

        # Safety.
        ESCAPED_DANGER: 8.0,
        ENTERED_DANGER: -10.0,
        e.SURVIVED_ROUND: 6.0,

        # Stage-4 multi-agent survival.
        ENTERED_DEAD_END: -5.0,
        LEFT_DEAD_END: 2.5,
        MOVED_TO_OPEN_SPACE: 0.5,
        MOVED_INTO_ENEMY_BOMB_RISK: -2.0,
        SAFE_MULTI_ESCAPE_BOMB: 1.5,
        RISKY_LOW_ESCAPE_BOMB: -8.0,

        # Loop/stall control.
        OSCILLATION: -8.0,
        REVISITED_TILE: -1.2,
        HEAVY_LOOP: -8.0,
        UNNECESSARY_WAIT: -2.5,
        e.WAITED: 0.0,
        e.INVALID_ACTION: -5.0,
    }

    reward = sum(game_rewards.get(event, 0.0) for event in events)

    # Suicide emits both KILLED_SELF and GOT_KILLED in this framework.
    if e.KILLED_SELF in events:
        reward -= 85.0
    elif e.GOT_KILLED in events:
        reward -= 50.0

    self.logger.debug(f"Reward {reward:.2f} for events: {events}")
    return reward
