"""Task-4 R3 rewards, on-policy updates and duplicate-safe terminal bookkeeping."""
from collections import Counter
import csv
from time import perf_counter

import events as e

from .features import ACTIONS,blast_cells,weighted_opponent_distance
from .model import AGENT_DIR, exploration, save_checkpoint, consume

FIELDS = ('episode','score','coins','kills','crates','bombs','suicides','deaths','survived','steps',
          'shaped_return','epsilon','invalid_actions','waits','visited_states','elapsed_seconds')
OFFENSIVE_BOMB='OFFENSIVE_BOMB_DROPPED'
APPROACHED='OPPONENT_APPROACHED'
RETREATED='OPPONENT_RETREATED'


def training_events(self,state,events,new_state=None):
    augmented=list(events)
    if self.config.reward_version in ('R5','R7') and e.BOMB_DROPPED in events:
        blast=set(blast_cells(state['field'],tuple(state['self'][3])))
        if any(tuple(other[3]) in blast for other in state['others']): augmented.append(OFFENSIVE_BOMB)
    if self.config.reward_version in ('R6','R7') and new_state is not None and state['others']:
        before=tuple(state['self'][3]); after=tuple(new_state['self'][3])
        if before!=after:
            distances=weighted_opponent_distance(state['field'],[tuple(other[3]) for other in state['others']])
            if distances[before]>=0 and distances[after]>=0:
                if distances[after]<distances[before]: augmented.append(APPROACHED)
                elif distances[after]>distances[before]: augmented.append(RETREATED)
    return augmented


def setup_training(self):
    self.history_fields=FIELDS+(('offensive_bombs',) if self.config.reward_version=='R5' else ())
    if self.config.reward_version=='R6': self.history_fields=FIELDS+('opponent_approaches','opponent_retreats')
    if self.config.reward_version=='R7': self.history_fields=FIELDS+('offensive_bombs','opponent_approaches','opponent_retreats')
    self.history_path = AGENT_DIR.parent.parent / 'results/training' / (self.config.experiment_id + '.csv')
    self.history_path.parent.mkdir(parents=True, exist_ok=True)
    self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if self.config.mode == 'fresh':
        with self.history_path.open('x', newline='') as stream:
            csv.DictWriter(stream, fieldnames=self.history_fields).writeheader()
        with self.checkpoint.open('xb'):
            pass
        save_checkpoint(self)
    else:
        with self.history_path.open(newline='') as stream:
            if len(list(csv.DictReader(stream))) != self.episodes:
                raise ValueError('History/checkpoint mismatch')
    self.counts, self.reward, self.steps = Counter(), 0., 0
    self.started = perf_counter()
    self.finished_round = None


def reward_from_events(events, reward_version='R3'):
    # Death is counted only once even though suicide emits KILLED_SELF and GOT_KILLED.
    if reward_version not in ('R3','R4','R5','R6','R7'): raise ValueError('Unknown reward version')
    kill_reward=20 if reward_version=='R4' else 5
    return (events.count(e.COIN_COLLECTED) + kill_reward * events.count(e.KILLED_OPPONENT) + .2 * events.count(e.CRATE_DESTROYED)
            - .05 * events.count(e.BOMB_DROPPED) - 5 * int(e.GOT_KILLED in events) - .01
            + (.2 * events.count(OFFENSIVE_BOMB) if reward_version in ('R5','R7') else 0)
            + (.05 * (events.count(APPROACHED)-events.count(RETREATED)) if reward_version in ('R6','R7') else 0))


def transition(self, state, action, new_state, events):
    return dict(key=(state['round'],state['step']), state=self.builder.describe(state)[0],
                action=ACTIONS.index(action), next_state=None if new_state is None else self.builder.describe(new_state)[0],
                reward=reward_from_events(events,self.config.reward_version))


def record(self, item, events):
    key = (item['state'], item['action'])
    self.visits[key] = self.visits.get(key, 0) + 1
    self.counts.update(events)
    self.reward += item['reward']
    self.steps += 1


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return
    if self.pending is not None:
        raise RuntimeError('Unconsumed SARSA transition')
    events=training_events(self,old_game_state,events,new_game_state)
    self.pending = transition(self, old_game_state, self_action, new_game_state, events)
    record(self, self.pending, events)


def end_of_round(self, last_game_state, last_action, events):
    if last_game_state is None:
        return
    key = (last_game_state['round'],last_game_state['step'])
    if self.finished_round == key[0]:
        raise RuntimeError('Duplicate terminal callback')
    if self.pending is not None:
        if self.pending['key'] != key:
            raise RuntimeError('Terminal transition mismatch')
        item = self.pending
    else:
        events=training_events(self,last_game_state,events)
        item = transition(self, last_game_state, last_action, None, events)
        record(self, item, events)
    consume(self, item, terminal=True)
    self.pending, self.finished_round = None, key[0]
    c = self.counts
    row = dict(episode=self.episodes+1,score=c[e.COIN_COLLECTED]+5*c[e.KILLED_OPPONENT],coins=c[e.COIN_COLLECTED],kills=c[e.KILLED_OPPONENT],
               crates=c[e.CRATE_DESTROYED],bombs=c[e.BOMB_DROPPED],suicides=c[e.KILLED_SELF],
               deaths=c[e.GOT_KILLED],survived=int(e.SURVIVED_ROUND in events),steps=self.steps,
               shaped_return=self.reward,epsilon=exploration(self),invalid_actions=c[e.INVALID_ACTION],
               waits=c[e.WAITED],visited_states=len(self.q),elapsed_seconds=perf_counter()-self.started)
    if self.config.reward_version in ('R5','R7'): row['offensive_bombs']=c[OFFENSIVE_BOMB]
    if self.config.reward_version in ('R6','R7'): row.update(opponent_approaches=c[APPROACHED],opponent_retreats=c[RETREATED])
    with self.history_path.open('a',newline='') as stream:
        csv.DictWriter(stream,fieldnames=self.history_fields).writerow(row)
    self.episodes += 1
    save_checkpoint(self)
    self.counts.clear()
    self.reward,self.steps,self.started=0.,0,perf_counter()
