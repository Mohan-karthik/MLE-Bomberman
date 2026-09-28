import os
import random

import numpy as np
import torch

from .features import ACTIONS, FEATURE_NAMES, mechanically_legal_actions, state_to_features
from .model import DuelingDQN


MODEL_FILE = 'clean_dqn_model.pt'
DEVICE = torch.device('cpu')


def setup(self):
    """
    Final-policy contract:
      1. Encode the game state into descriptive features.
      2. Let the trained network predict Q(s, a).
      3. Mask only mechanically impossible actions.
      4. Choose argmax Q among the mechanically legal actions.

    There are deliberately NO hand-written tactical Q bonuses/penalties and
    NO strategic/safety action filters in this callback.
    """
    torch.set_num_threads(1)

    self.online_net = DuelingDQN(len(FEATURE_NAMES), len(ACTIONS)).to(DEVICE)
    self.target_net = DuelingDQN(len(FEATURE_NAMES), len(ACTIONS)).to(DEVICE)
    self.target_net.load_state_dict(self.online_net.state_dict())

    self.episodes = 0
    self.training_steps = 0
    self.env_steps = 0
    self.epsilon = 1.0 if self.train else 0.0
    self._optimizer_state = None
    self.checkpoint_version = 0

    if os.path.isfile(MODEL_FILE):
        try:
            checkpoint = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)
            input_dim = int(checkpoint.get('input_dim', -1))
            action_dim = int(checkpoint.get('action_dim', -1))
            if input_dim != len(FEATURE_NAMES) or action_dim != len(ACTIONS):
                raise ValueError(
                    f'Checkpoint dimensions {input_dim}x{action_dim} do not match '
                    f'{len(FEATURE_NAMES)}x{len(ACTIONS)}.'
                )

            self.online_net.load_state_dict(checkpoint['online_state_dict'])
            self.target_net.load_state_dict(
                checkpoint.get('target_state_dict', checkpoint['online_state_dict'])
            )
            self.episodes = int(checkpoint.get('episodes', 0))
            self.training_steps = int(checkpoint.get('training_steps', 0))
            self.env_steps = int(checkpoint.get('env_steps', 0))
            self._optimizer_state = checkpoint.get('optimizer_state_dict')
            self.checkpoint_version = int(checkpoint.get('version', 1))
            self.epsilon = (
                float(checkpoint.get('epsilon', 1.0)) if self.train else 0.0
            )
            self.logger.info(
                f'Loaded clean Dueling Double-DQN checkpoint v{self.checkpoint_version} after {self.episodes} rounds | '
                f'epsilon={self.epsilon:.3f} | updates={self.training_steps}'
            )
        except Exception as exc:
            self.logger.warning(f'Could not load clean checkpoint; starting fresh: {exc}')
            self.target_net.load_state_dict(self.online_net.state_dict())
    else:
        self.logger.info(
            f'Created clean Dueling Double-DQN: {len(FEATURE_NAMES)} inputs -> '
            f'128 -> 128 -> value/advantage -> {len(ACTIONS)} Q-values.'
        )

    self.online_net.eval()
    self.target_net.eval()


def act(self, game_state: dict) -> str:
    legal = mechanically_legal_actions(game_state)

    # Exploration is training-only and uniform over mechanically legal actions.
    if self.train and random.random() < self.epsilon:
        return random.choice(legal)

    features = state_to_features(game_state)
    feature_tensor = torch.as_tensor(
        features,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    self.online_net.eval()
    with torch.no_grad():
        q_values = self.online_net(feature_tensor).squeeze(0).cpu().numpy()

    # Mechanical mask only. No tactical/safety desirability is imposed here.
    masked_q = np.full(len(ACTIONS), -np.inf, dtype=np.float32)
    for action in legal:
        idx = ACTIONS.index(action)
        masked_q[idx] = q_values[idx]

    # Deterministic learned greedy policy during evaluation.
    best_index = int(np.argmax(masked_q))
    return ACTIONS[best_index]
