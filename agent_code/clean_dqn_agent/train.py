from collections import deque, namedtuple
import pickle
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import events as e

from .callbacks import ACTIONS, DEVICE, MODEL_FILE
from .features import (
    FEATURE_NAMES,
    MOVE_DELTAS,
    bomb_crate_count,
    bomb_enemy_count,
    bomb_escape_route_count,
    can_escape_after_bomb,
    danger_map,
    existing_bomb_overlap_count,
    local_exit_count,
    mechanically_legal_actions,
    nearest_coin_info,
    nearest_crate_info,
    nearest_enemy_info,
    potential_enemy_bomb_risk_count,
    state_to_features,
)


# ---------------------------- DQN hyperparameters ----------------------------
GAMMA = 0.97
LEARNING_RATE = 1.5e-4
BATCH_SIZE = 64
REPLAY_CAPACITY = 100_000
REPLAY_WARMUP = 2_000
TRAIN_EVERY = 4
TARGET_UPDATE_INTERVAL = 500
GRADIENT_CLIP_NORM = 10.0
REWARD_SCALE = 0.10

EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9985

Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'next_state', 'next_legal_mask', 'done'],
)


# ------------------------------- custom events -------------------------------
MOVED_TOWARD_COIN = 'MOVED_TOWARD_COIN'
MOVED_AWAY_FROM_COIN = 'MOVED_AWAY_FROM_COIN'
MOVED_TOWARD_CRATE = 'MOVED_TOWARD_CRATE'
MOVED_AWAY_FROM_CRATE = 'MOVED_AWAY_FROM_CRATE'
MOVED_TOWARD_ENEMY = 'MOVED_TOWARD_ENEMY'
MOVED_AWAY_FROM_ENEMY = 'MOVED_AWAY_FROM_ENEMY'
ESCAPED_DANGER = 'ESCAPED_DANGER'
ENTERED_DANGER = 'ENTERED_DANGER'
GOOD_CRATE_BOMB = 'GOOD_CRATE_BOMB'
GOOD_ATTACK_BOMB = 'GOOD_ATTACK_BOMB'
SAFE_BOMB = 'SAFE_BOMB'
RISKY_BOMB = 'RISKY_BOMB'
USELESS_BOMB = 'USELESS_BOMB'
OSCILLATION = 'OSCILLATION'
REVISITED_TILE = 'REVISITED_TILE'
HEAVY_LOOP = 'HEAVY_LOOP'
UNNECESSARY_WAIT = 'UNNECESSARY_WAIT'
ENTERED_DEAD_END_UNDER_PRESSURE = 'ENTERED_DEAD_END_UNDER_PRESSURE'
MOVED_INTO_ENEMY_BOMB_RISK = 'MOVED_INTO_ENEMY_BOMB_RISK'
MOVED_OUT_OF_ENEMY_BOMB_RISK = 'MOVED_OUT_OF_ENEMY_BOMB_RISK'
IMPROVED_DANGER = 'IMPROVED_DANGER'
STAYED_IN_URGENT_DANGER = 'STAYED_IN_URGENT_DANGER'
MOVED_TO_OPEN_SPACE_UNDER_PRESSURE = 'MOVED_TO_OPEN_SPACE_UNDER_PRESSURE'
ROBUST_ESCAPE_BOMB = 'ROBUST_ESCAPE_BOMB'


def setup_training(self):
    self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
    self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)

    if self._optimizer_state is not None:
        try:
            self.optimizer.load_state_dict(self._optimizer_state)
            for group in self.optimizer.param_groups:
                group['lr'] = LEARNING_RATE
            self.logger.info('Restored Adam optimizer state.')
        except Exception as exc:
            self.logger.warning(f'Could not restore optimizer state: {exc}')

    self.online_net.train()
    self.target_net.eval()
    self.epsilon = max(EPSILON_MIN, float(self.epsilon))
    # V3 is intended to fine-tune the survival-oriented V2 checkpoint rather than
    # relearn from scratch. Re-open only moderate exploration so the model can
    # discover more productive attacking trajectories without discarding safety.
    if int(getattr(self, 'checkpoint_version', 0)) < 3 and self.episodes > 0:
        self.epsilon = max(self.epsilon, 0.12)
        self.logger.info('V3 migration: reopened training epsilon to %.3f.', self.epsilon)

    self.round_reward = 0.0
    self.round_losses = []
    self.round_updates = 0
    self.position_history = deque(maxlen=40)


def _distance(info_function, game_state):
    _, distance = info_function(game_state)
    return distance


def _state_with_position(game_state, position):
    hypothetical = dict(game_state)
    old_self = game_state['self']
    hypothetical['self'] = (
        old_self[0], old_self[1], old_self[2], tuple(position)
    )
    return hypothetical


def current_danger_timer(game_state):
    if game_state is None:
        return None
    pos = tuple(game_state['self'][3])
    danger = danger_map(game_state)
    if game_state['explosion_map'][pos] > 0:
        return 0.0
    value = float(danger[pos])
    return None if value >= 5.0 else value


def add_custom_events(self, old_game_state, self_action, new_game_state, events):
    """Training-only reward shaping. No part of this function runs in evaluation."""
    if old_game_state is None or new_game_state is None:
        return

    old_pos = tuple(old_game_state['self'][3])
    new_pos = tuple(new_game_state['self'][3])
    moved = self_action in MOVE_DELTAS and self_action != 'WAIT'

    # Loop penalties teach the model rather than overriding it at inference.
    if not self.position_history:
        self.position_history.append(old_pos)
    history = list(self.position_history)
    if len(history) >= 2 and new_pos == history[-2]:
        events.append(OSCILLATION)
    visits = history.count(new_pos)
    if visits >= 1:
        events.append(REVISITED_TILE)
    if visits >= 3:
        events.append(HEAVY_LOOP)
    self.position_history.append(new_pos)

    # Dense navigation shaping.
    if moved:
        old_coin = _distance(nearest_coin_info, old_game_state)
        new_coin = _distance(nearest_coin_info, new_game_state)
        if old_coin is not None and new_coin is not None and e.COIN_COLLECTED not in events:
            if new_coin < old_coin:
                events.append(MOVED_TOWARD_COIN)
            elif new_coin > old_coin:
                events.append(MOVED_AWAY_FROM_COIN)

        # Crate progress matters primarily while no visible coin is available.
        if not old_game_state['coins']:
            old_crate = _distance(nearest_crate_info, old_game_state)
            new_crate = _distance(nearest_crate_info, new_game_state)
            if old_crate is not None and new_crate is not None:
                if new_crate < old_crate:
                    events.append(MOVED_TOWARD_CRATE)
                elif new_crate > old_crate:
                    events.append(MOVED_AWAY_FROM_CRATE)

        if old_game_state['others']:
            old_enemy = _distance(nearest_enemy_info, old_game_state)
            new_enemy = _distance(nearest_enemy_info, new_game_state)
            if old_enemy is not None and new_enemy is not None:
                if new_enemy < old_enemy:
                    events.append(MOVED_TOWARD_ENEMY)
                elif new_enemy > old_enemy:
                    events.append(MOVED_AWAY_FROM_ENEMY)

        old_danger = current_danger_timer(old_game_state)
        new_danger = current_danger_timer(new_game_state)
        old_urgent = old_danger is not None and old_danger <= 2
        new_urgent = new_danger is not None and new_danger <= 2
        if old_urgent and not new_urgent:
            events.append(ESCAPED_DANGER)
        elif not old_urgent and new_urgent:
            events.append(ENTERED_DANGER)
        elif old_urgent and new_urgent:
            # Still threatened, but distinguish genuine progress from lingering.
            if old_danger is not None and new_danger is not None and new_danger > old_danger:
                events.append(IMPROVED_DANGER)
            else:
                events.append(STAYED_IN_URGENT_DANGER)

        old_risk = potential_enemy_bomb_risk_count(old_game_state, old_pos)
        new_risk = potential_enemy_bomb_risk_count(old_game_state, new_pos)
        if not old_urgent and new_risk > old_risk:
            events.append(MOVED_INTO_ENEMY_BOMB_RISK)
        elif new_risk < old_risk:
            events.append(MOVED_OUT_OF_ENEMY_BOMB_RISK)

        old_exits = local_exit_count(old_game_state, old_pos)
        new_exits = local_exit_count(old_game_state, new_pos)
        pressure = (
            len(old_game_state['bombs']) > 0
            or len(old_game_state['others']) >= 2
            or existing_bomb_overlap_count(old_game_state, old_pos) > 0
        )
        if pressure and old_exits > 1 and new_exits <= 1:
            events.append(ENTERED_DEAD_END_UNDER_PRESSURE)
        elif pressure and new_exits > old_exits:
            events.append(MOVED_TO_OPEN_SPACE_UNDER_PRESSURE)

    # Bomb quality shaping is training-only. BOMB remains mechanically legal
    # during evaluation even when these conditions are poor.
    if self_action == 'BOMB':
        crates = bomb_crate_count(old_game_state)
        enemies = bomb_enemy_count(old_game_state)
        routes = bomb_escape_route_count(old_game_state)
        can_escape = can_escape_after_bomb(old_game_state)

        if crates > 0:
            events.append(GOOD_CRATE_BOMB)
        if enemies > 0:
            events.append(GOOD_ATTACK_BOMB)
        if can_escape:
            events.append(SAFE_BOMB)
        if can_escape and routes >= 2:
            events.append(ROBUST_ESCAPE_BOMB)
        if crates == 0 and enemies == 0:
            events.append(USELESS_BOMB)

        # Training-only safety target. BOMB is never removed from evaluation.
        # In multiplayer, a single theoretical route is often lost when another
        # bomb/opponent changes the corridor, so teach the network to value redundancy.
        risky_bomb = (
            not can_escape
            or (len(old_game_state['others']) >= 2 and routes < 2)
            or (len(old_game_state['bombs']) > 0 and routes < 2)
            or (existing_bomb_overlap_count(old_game_state, old_pos) > 0 and routes < 2)
        )
        if risky_bomb:
            events.append(RISKY_BOMB)

    if self_action == 'WAIT':
        danger = current_danger_timer(old_game_state)
        movement_available = any(
            action in mechanically_legal_actions(old_game_state)
            for action in ['UP', 'RIGHT', 'DOWN', 'LEFT']
        )
        objective_exists = bool(
            old_game_state['coins']
            or np.any(old_game_state['field'] == 1)
            or old_game_state['others']
        )
        if danger is None and movement_available and objective_exists:
            events.append(UNNECESSARY_WAIT)


def reward_from_events(self, events: List[str]) -> float:
    # V3 balances V1's offensive productivity with V2's learned survival.
    # These values affect training only; evaluation still uses learned Q-values
    # plus the mechanical legality mask and no tactical hand-written bonuses.
    rewards = {
        e.COIN_COLLECTED: 20.0,
        e.COIN_FOUND: 4.0,
        e.CRATE_DESTROYED: 3.0,
        e.KILLED_OPPONENT: 105.0,
        e.SURVIVED_ROUND: 12.0,
        e.INVALID_ACTION: -8.0,

        MOVED_TOWARD_COIN: 1.5,
        MOVED_AWAY_FROM_COIN: -1.8,
        MOVED_TOWARD_CRATE: 0.8,
        MOVED_AWAY_FROM_CRATE: -0.9,
        MOVED_TOWARD_ENEMY: 0.75,
        MOVED_AWAY_FROM_ENEMY: -0.25,

        ESCAPED_DANGER: 12.0,
        ENTERED_DANGER: -15.0,
        MOVED_INTO_ENEMY_BOMB_RISK: -4.5,
        MOVED_OUT_OF_ENEMY_BOMB_RISK: 2.0,
        IMPROVED_DANGER: 2.5,
        STAYED_IN_URGENT_DANGER: -2.5,
        ENTERED_DEAD_END_UNDER_PRESSURE: -7.0,
        MOVED_TO_OPEN_SPACE_UNDER_PRESSURE: 1.5,

        # Productive bombing is encouraged, but only strongly when it is both
        # offensive and survivable. A risky attack bomb therefore receives the
        # attack reward and the risk penalty together instead of being forbidden.
        GOOD_CRATE_BOMB: 1.5,
        GOOD_ATTACK_BOMB: 14.0,
        SAFE_BOMB: 2.5,
        ROBUST_ESCAPE_BOMB: 2.0,
        RISKY_BOMB: -17.0,
        USELESS_BOMB: -8.0,

        OSCILLATION: -5.0,
        REVISITED_TILE: -0.8,
        HEAVY_LOOP: -5.0,
        UNNECESSARY_WAIT: -2.0,
    }

    reward = sum(rewards.get(event, 0.0) for event in events)

    # Keep death expensive enough to preserve V2's safety, but not so expensive
    # that the network learns to avoid productive combat altogether.
    if e.KILLED_SELF in events:
        reward -= 155.0
    elif e.GOT_KILLED in events:
        reward -= 85.0

    self.logger.debug(f'Reward {reward:.2f} for events: {events}')
    return reward


def legal_mask(game_state):
    mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    if game_state is None:
        return mask
    for action in mechanically_legal_actions(game_state):
        mask[ACTIONS.index(action)] = True
    return mask


def remember_transition(self, old_game_state, action, reward, new_game_state, done):
    if old_game_state is None or action not in ACTIONS:
        return

    state = state_to_features(old_game_state)
    if new_game_state is None:
        next_state = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        next_mask = np.zeros(len(ACTIONS), dtype=np.bool_)
    else:
        next_state = state_to_features(new_game_state)
        next_mask = legal_mask(new_game_state)

    self.replay_buffer.append(
        Transition(
            state=np.asarray(state, dtype=np.float32),
            action=ACTIONS.index(action),
            reward=float(reward) * REWARD_SCALE,
            next_state=np.asarray(next_state, dtype=np.float32),
            next_legal_mask=np.asarray(next_mask, dtype=np.bool_),
            done=bool(done),
        )
    )
    self.env_steps += 1


def optimize_model(self):
    if len(self.replay_buffer) < max(REPLAY_WARMUP, BATCH_SIZE):
        return None
    if self.env_steps % TRAIN_EVERY != 0:
        return None

    batch = random.sample(self.replay_buffer, BATCH_SIZE)
    states = torch.as_tensor(
        np.stack([t.state for t in batch]), dtype=torch.float32, device=DEVICE
    )
    actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=DEVICE)
    rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=DEVICE)
    next_states = torch.as_tensor(
        np.stack([t.next_state for t in batch]), dtype=torch.float32, device=DEVICE
    )
    next_masks = torch.as_tensor(
        np.stack([t.next_legal_mask for t in batch]), dtype=torch.bool, device=DEVICE
    )
    dones = torch.as_tensor([t.done for t in batch], dtype=torch.bool, device=DEVICE)

    self.online_net.train()
    current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN: online network selects; target network evaluates.
    with torch.no_grad():
        online_next_q = self.online_net(next_states)
        online_next_q = online_next_q.masked_fill(~next_masks, -torch.inf)

        # Terminal rows have an all-false mask; choose arbitrary action there
        # because their bootstrap term is multiplied by zero anyway.
        all_illegal = ~next_masks.any(dim=1)
        if all_illegal.any():
            online_next_q[all_illegal] = 0.0

        next_actions = online_next_q.argmax(dim=1)
        target_next_q_all = self.target_net(next_states)
        target_next_q = target_next_q_all.gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)
        target_next_q = torch.where(dones, torch.zeros_like(target_next_q), target_next_q)
        targets = rewards + GAMMA * target_next_q

    loss = F.smooth_l1_loss(current_q, targets)

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), GRADIENT_CLIP_NORM)
    self.optimizer.step()

    self.training_steps += 1
    self.round_updates += 1
    self.round_losses.append(float(loss.item()))

    if self.training_steps % TARGET_UPDATE_INTERVAL == 0:
        self.target_net.load_state_dict(self.online_net.state_dict())

    return float(loss.item())


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    add_custom_events(self, old_game_state, self_action, new_game_state, events)
    reward = reward_from_events(self, events)
    self.round_reward += reward

    remember_transition(
        self,
        old_game_state,
        self_action,
        reward,
        new_game_state,
        done=False,
    )
    optimize_model(self)


def save_checkpoint(self):
    checkpoint = {
        'version': 3,
        'architecture': 'clean_dueling_double_dqn_balanced_v3',
        'input_dim': len(FEATURE_NAMES),
        'action_dim': len(ACTIONS),
        'online_state_dict': self.online_net.state_dict(),
        'target_state_dict': self.target_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'episodes': int(self.episodes),
        'epsilon': float(self.epsilon),
        'training_steps': int(self.training_steps),
        'env_steps': int(self.env_steps),
    }
    torch.save(checkpoint, MODEL_FILE)


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: List[str],
):
    reward = reward_from_events(self, events)
    self.round_reward += reward

    remember_transition(
        self,
        last_game_state,
        last_action,
        reward,
        None,
        done=True,
    )
    optimize_model(self)

    self.episodes += 1
    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    mean_loss = float(np.mean(self.round_losses)) if self.round_losses else 0.0
    self.logger.info(
        f'Clean-DDQN-V3 round {self.episodes} | reward={self.round_reward:.2f} | '
        f'epsilon={self.epsilon:.3f} | replay={len(self.replay_buffer)} | '
        f'updates={self.round_updates} | mean_loss={mean_loss:.5f}'
    )

    save_checkpoint(self)

    self.round_reward = 0.0
    self.round_losses = []
    self.round_updates = 0
    self.position_history.clear()
