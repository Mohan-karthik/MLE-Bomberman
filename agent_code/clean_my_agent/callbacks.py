import os
import pickle
import random

import numpy as np

from .features import FEATURE_NAMES, mechanically_legal_actions, state_to_features
from .model import new_model, q_values


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = 'clean_q_model.pt'
SEED_FILES = [
    'checkpoint_stage4_v2_final.pt',
    'checkpoint_before_cleaning.pt',
    'my-saved-model.pt',
]


def _load_compatible_model(path):
    with open(path, 'rb') as file:
        candidate = pickle.load(file)

    if not isinstance(candidate, dict) or 'weights' not in candidate:
        raise ValueError('Not a valid Approximate-Q checkpoint.')

    expected = (len(ACTIONS), len(FEATURE_NAMES))
    if tuple(candidate['weights'].shape) != expected:
        raise ValueError(
            f"Checkpoint shape {candidate['weights'].shape} does not match {expected}."
        )
    return candidate


def setup(self):
    """
    Compliance-clean final policy:
      features -> learned Q-values -> mechanical legality mask -> argmax.

    There are NO inference-time progress bonuses, survival adjustments,
    anti-loop penalties, tactical action priorities, or strategic action
    removal in this callback.
    """
    self.model = None
    loaded_from_seed = False

    # Prefer a checkpoint already fine-tuned under the clean policy.
    if os.path.isfile(MODEL_FILE):
        try:
            self.model = _load_compatible_model(MODEL_FILE)
            self.logger.info(
                f"Loaded clean Approx-Q model after "
                f"{self.model.get('episodes', 0)} rounds."
            )
        except Exception as exc:
            self.logger.warning(f'Could not load {MODEL_FILE}: {exc}')

    # Optional warm start from the user's strong 64-feature Stage-4 v2 model.
    # It is only an initialization; subsequent experience is collected using
    # this clean policy. The final act() remains fully learned.
    if self.model is None:
        for seed_file in SEED_FILES:
            if not os.path.isfile(seed_file):
                continue
            try:
                self.model = _load_compatible_model(seed_file)
                loaded_from_seed = True
                self.logger.info(f'Warm-started clean policy from {seed_file}.')
                break
            except Exception as exc:
                self.logger.warning(f'Could not use seed checkpoint {seed_file}: {exc}')

    if self.model is None:
        self.model = new_model(len(ACTIONS), len(FEATURE_NAMES))
        self.logger.info(
            f'Created fresh clean Approximate-Q model with '
            f'{len(FEATURE_NAMES)} descriptive features.'
        )

    # Re-open exploration when converting the old assisted checkpoint to the
    # clean behavior policy so the values can adapt to the new policy/data.
    if self.train:
        self.epsilon = float(self.model.get('epsilon', 1.0))
        if loaded_from_seed:
            self.epsilon = max(self.epsilon, 0.20)
            self.model['epsilon'] = self.epsilon
    else:
        self.epsilon = 0.0


def act(self, game_state: dict) -> str:
    legal = mechanically_legal_actions(game_state)

    # Training-only exploration, uniform over mechanically executable actions.
    if self.train and random.random() < self.epsilon:
        return random.choice(legal)

    features = state_to_features(game_state)
    values = q_values(self.model, features)

    # The ONLY post-model operation is mechanical legality masking.
    masked = np.full(len(ACTIONS), -np.inf, dtype=np.float32)
    for action in legal:
        idx = ACTIONS.index(action)
        masked[idx] = values[idx]

    return ACTIONS[int(np.argmax(masked))]
