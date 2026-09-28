"""Self-contained sparse tabular SARSA for Task 3; no Task-1 imports."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import pickle
from time import sleep

import numpy as np

from .features import ACTIONS, FeatureBuilder

AGENT_DIR = Path(__file__).resolve().parent
VERSION = ('SARSA-v6', 'F5', 'R3')
VERSIONS = {'F5': VERSION, 'F6': ('SARSA-v7','F6','R3'), 'F7': ('SARSA-v10','F7','R3'), 'F8': ('SARSA-v11','F8','R3')}
VERSIONS['F9']=('SARSA-v13','F9','R3')
VERSIONS['F10']=('SARSA-v17','F10','R7')
REWARD_VERSIONS = {'R3','R4','R5','R6','R7'}


def checkpoint_version(config):
    if config.feature_version=='F10': return VERSIONS['F10']
    if config.n_step==6: return ('SARSA-v9','F5','R3')
    if config.reward_version=='R5': return ('SARSA-v12','F5','R5')
    if config.reward_version=='R6': return ('SARSA-v14','F9','R6')
    if config.reward_version=='R7': return ('SARSA-v15','F9','R7')
    return ('SARSA-v8',config.feature_version,'R4') if config.reward_version=='R4' else VERSIONS[config.feature_version]


@dataclass
class Config:
    experiment_id: str = 'task4_manual'
    model_file: str = 'checkpoints/task4_manual.pkl'
    mode: str = 'fresh'
    seed: int = 101
    alpha: float = .1
    gamma: float = .95
    epsilon_start: float = .05
    epsilon_min: float = .005
    epsilon_decay: float = .99
    feature_version: str = 'F10'
    reward_version: str = 'R7'
    n_step: int = 1
    transfer: bool = True


def model_path(name):
    path = Path(name)
    result = (AGENT_DIR / path).resolve()
    if path.is_absolute() or '..' in path.parts or AGENT_DIR not in result.parents:
        raise ValueError('Checkpoint must be relative and inside this agent directory')
    return result


def load_checkpoint(path):
    with path.open('rb') as stream:
        data = pickle.load(stream)
    compatible=set(VERSIONS.values()) | {('SARSA-v16','F10','R7'),('SARSA-v8','F5','R4'),('SARSA-v9','F5','R3'),('SARSA-v12','F5','R5'),('SARSA-v14','F9','R6'),('SARSA-v15','F9','R7')}
    if data['version'] not in compatible or data['actions'] != ACTIONS:
        raise ValueError('Incompatible Task-4 checkpoint')
    if any(len(values) != 6 or not np.isfinite(values).all() for values in list(data['q'].values())+list(data['prior_q'].values())):
        raise ValueError('Invalid Task-4 Q-values')
    return data


def initialize(self):
    self.config = Config(**json.loads(os.environ.get('SARSA4_CONFIG', '{}')))
    c = self.config
    if c.feature_version not in VERSIONS:
        raise ValueError('Unsupported feature version')
    if c.reward_version not in REWARD_VERSIONS or (c.reward_version in ('R4','R5') and c.feature_version!='F5'):
        raise ValueError('Unsupported feature/reward combination')
    if c.feature_version=='F10' and c.reward_version!='R7': raise ValueError('F10 requires R7')
    if c.reward_version in ('R6','R7') and c.feature_version not in ('F9','F10'): raise ValueError('Pursuit shaping requires F9')
    if c.n_step not in (1,6) or (c.n_step==6 and (c.feature_version,c.reward_version)!=('F5','R3')):
        raise ValueError('Unsupported SARSA horizon')
    self.feature_version = c.feature_version
    if c.mode not in ('fresh', 'resume') or not c.experiment_id or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in c.experiment_id):
        raise ValueError('Invalid run identity or mode')
    if not 0 < c.alpha <= 1 or not 0 <= c.gamma <= 1 or not 0 <= c.epsilon_min <= c.epsilon_start <= 1 or not 0 < c.epsilon_decay <= 1:
        raise ValueError('Invalid hyperparameters')
    self.checkpoint = model_path(c.model_file if self.train else os.environ.get('SARSA4_MODEL', 'model.pkl'))
    self.q, self.visits, self.episodes = {}, {}, 0
    self.prior_q = {}
    self.rng = np.random.default_rng(c.seed if self.train else int(os.environ.get('SARSA4_EVAL_SEED', '0')))
    self.eval_epsilon = float(os.environ.get('SARSA4_EVAL_EPSILON', '0'))
    if not 0 <= self.eval_epsilon <= 1:
        raise ValueError('Invalid evaluation epsilon')
    if not self.train or c.mode == 'resume':
        payload = load_checkpoint(self.checkpoint)
        self.q, self.visits, self.episodes = payload['q'], payload['visits'], payload['episodes']
        self.prior_q=payload['prior_q']
        self.feature_version = payload['version'][1]
        if self.train:
            previous = payload['config'].copy()
            previous.setdefault('feature_version', 'F5')
            previous.setdefault('reward_version', 'R3')
            previous.setdefault('n_step', 1)
            previous['mode'] = 'resume'
            if previous != asdict(c):
                raise ValueError('Resume configuration mismatch')
            self.rng.bit_generator.state = payload['rng_state']
        else:
            for values in list(self.q.values())+list(self.prior_q.values()):
                values.flags.writeable = False
    elif self.checkpoint.exists():
        raise FileExistsError('Choose a new Task-4 run ID; fresh training cannot overwrite checkpoints')
    elif c.transfer:
        with (AGENT_DIR/'bootstrap.pkl').open('rb') as stream: prior=pickle.load(stream)
        if prior['version'] != ('SARSA-v16','F10','R7') or prior['actions'] != ACTIONS:
            raise ValueError('Task4 bootstrap requires the confirmed E153 F10/R7 policy')
        self.q={key:value.copy() for key,value in prior['q'].items()}
        self.prior_q={key:value.copy() for key,value in prior['prior_q'].items()}
    self.builder = FeatureBuilder(self.feature_version)
    self.pending = None
    self.trajectory = []


def values(self, state):
    if state not in self.q:
        prior_state=state[:-1]
        if state[0]==0 and state[2]>=33:
            prior_state=state[:2]+(state[2]-16,)+state[3:-1]
        initial=self.prior_q.get(prior_state,np.zeros(6))
        if not self.train:
            return initial
        self.q[state] = initial.copy()
    return self.q[state]


def exploration(self):
    return max(self.config.epsilon_min, self.config.epsilon_start * self.config.epsilon_decay ** self.episodes) if self.train else self.eval_epsilon


def choose(self, state, allowed):
    eligible = np.flatnonzero(allowed)
    epsilon = exploration(self)
    if epsilon > 0 and self.rng.random() < epsilon:
        return int(self.rng.choice(eligible))
    estimates = values(self, state)[eligible]
    return int(self.rng.choice(eligible[estimates == estimates.max()]))


def update(self, transition, next_state=None, next_action=None):
    target = transition['reward']
    if next_state is not None:
        target += self.config.gamma * values(self, next_state)[next_action]
    row = values(self, transition['state'])
    action = transition['action']
    row[action] += self.config.alpha * (target - row[action])


def consume(self, transition, next_state=None, next_action=None, terminal=False):
    """Actual-action n-step SARSA; terminal returns contain no bootstrap."""
    if self.config.n_step==1:
        update(self,transition,None if terminal else next_state,None if terminal else next_action)
        return
    self.trajectory.append(transition)
    if not terminal and len(self.trajectory)<self.config.n_step:
        return
    while self.trajectory:
        count=len(self.trajectory)
        target=sum(self.config.gamma**i * item['reward'] for i,item in enumerate(self.trajectory))
        if not terminal:
            target+=self.config.gamma**count * values(self,next_state)[next_action]
        first=self.trajectory.pop(0)
        update(self,dict(first,reward=target))
        if not terminal: break


def save_checkpoint(self):
    if not self.train:
        raise RuntimeError('Frozen evaluation cannot save checkpoints')
    if self.trajectory:
        raise RuntimeError('Checkpoints are saved only after terminal n-step flushing')
    payload = dict(version=checkpoint_version(self.config), actions=ACTIONS, config=asdict(self.config), q=self.q,
                   prior_q=self.prior_q,visits=self.visits, episodes=self.episodes, rng_state=self.rng.bit_generator.state)
    temporary = self.checkpoint.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(payload, stream, protocol=4)
    for attempt in range(20):
        try:
            temporary.replace(self.checkpoint)
            break
        except PermissionError:
            if attempt == 19:
                raise
            sleep(.05)
