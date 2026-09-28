"""Six-action SARSA policy; safety information is learned from, never an override."""
from .features import ACTIONS
from .model import initialize, choose, consume


def setup(self):
    initialize(self)


def act(self, game_state):
    state, allowed = self.builder.describe(game_state)
    action = choose(self, state, allowed)
    if self.train and self.pending is not None:
        if self.pending['key'][0] != game_state['round'] or self.pending['next_state'] != state:
            raise RuntimeError('SARSA pending state mismatch')
        consume(self, self.pending, state, action)
        self.pending = None
    return ACTIONS[action]
