"""F5: opponent routes/exposure plus discrete Task-2 navigation and escape.

No helper returns the policy action. Only mechanical invalid actions are masked.
The forecast follows the inspected local engine: blasts cross crates, stop at
walls, explode after timer+1 actions and are lethal for two action outcomes.
"""
from collections import deque
import heapq

import numpy as np
import settings as settings

ACTIONS = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB')
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))


def blast_cells(field, position):
    cells = [tuple(position)]
    for dx, dy in DELTAS[:4]:
        for distance in range(1, settings.BOMB_POWER + 1):
            x, y = position[0] + dx * distance, position[1] + dy * distance
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]) or field[x, y] == -1:
                break
            cells.append((x, y))
    return cells


def distance_and_target(free, targets):
    """Multi-source BFS returns distance and target coordinates, not a first action."""
    distance = np.full(free.shape, -1, dtype=np.int16)
    nearest = np.full((*free.shape, 2), -1, dtype=np.int16)
    queue = deque()
    for position in sorted(set(targets)):
        if free[position]:
            distance[position] = 0
            nearest[position] = position
            queue.append(position)
    while queue:
        x, y = queue.popleft()
        for dx, dy in DELTAS[:4]:
            destination = x + dx, y + dy
            a, b = destination
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[destination] and distance[destination] < 0:
                distance[destination] = distance[x, y] + 1
                nearest[destination] = nearest[x, y]
                queue.append(destination)
    return distance, nearest


def weighted_opponent_distance(field, targets):
    """Route costs only: free entry costs1, crate entry costs5, walls block."""
    distance=np.full(field.shape,32767,dtype=np.int16)
    queue=[]
    for x,y in sorted(set(targets)):
        distance[x,y]=0; heapq.heappush(queue,(0,x,y))
    while queue:
        cost,x,y=heapq.heappop(queue)
        if cost!=distance[x,y]: continue
        candidate=cost+(5 if field[x,y]==1 else 1)
        for dx,dy in DELTAS[:4]:
            a,b=x+dx,y+dy
            if 0<=a<field.shape[0] and 0<=b<field.shape[1] and field[a,b]!=-1 and candidate<distance[a,b]:
                distance[a,b]=candidate; heapq.heappush(queue,(candidate,a,b))
    distance[distance==32767]=-1
    return distance


def crate_yields(field, free):
    """All free-tile blast yields, vectorized with exactly the same wall stops."""
    result=np.zeros(field.shape,dtype=np.int16)
    width,height=field.shape
    for dx,dy in DELTAS[:4]:
        reachable=free.copy()
        for distance in range(1,settings.BOMB_POWER+1):
            ox,oy=dx*distance,dy*distance
            shifted=np.full(field.shape,-1,dtype=field.dtype)
            x0,x1=max(0,-ox),min(width,width-ox)
            y0,y1=max(0,-oy),min(height,height-oy)
            if x0<x1 and y0<y1:
                shifted[x0:x1,y0:y1]=field[x0+ox:x1+ox,y0+oy:y1+oy]
            reachable &= shifted != -1
            result += reachable & (shifted == 1)
    return result


def forecast(field, bombs, explosion_map):
    """Return outcome hazards and pre-action bomb occupancy for horizons 1..H.

    Crates are conservatively kept blocked during escape planning. The real engine
    clears them at detonation; current open escape paths do not require that change.
    The engine has no chain detonation. Smoke contributes no hazard.
    """
    horizon = max([int(timer) + 2 for _, timer in bombs] + [int(explosion_map.max()), 1])
    danger = np.zeros((horizon + 1, *field.shape), dtype=bool)
    occupied = np.zeros_like(danger)
    for h in range(1, horizon + 1):
        danger[h] |= explosion_map >= h
    for position, timer in bombs:
        position = tuple(position)
        detonation = int(timer) + 1
        occupied[1:detonation + 1, position[0], position[1]] = True
        for cell in blast_cells(field, position):
            danger[detonation:detonation + 2, cell[0], cell[1]] = True
    return danger, occupied


def escape_layers(field, bombs, explosion_map, others=()):
    """Backward dynamic programming: existence of a safe continuation at each tile/time."""
    danger, occupied = forecast(field, bombs, explosion_map)
    free = field == 0
    for position in others:
        free[tuple(position)] = False
    viable = np.zeros_like(danger)
    viable[-1] = free
    for h in range(len(danger) - 2, -1, -1):
        # Staying atop a bomb is legal; entering its tile from elsewhere is not.
        safe_wait = viable[h + 1] & free & ~danger[h + 1]
        destinations = safe_wait & ~occupied[h + 1]
        reachable = safe_wait.copy()
        reachable[1:, :] |= destinations[:-1, :]
        reachable[:-1, :] |= destinations[1:, :]
        reachable[:, 1:] |= destinations[:, :-1]
        reachable[:, :-1] |= destinations[:, 1:]
        viable[h] = reachable & free
    return danger, occupied, viable


class FeatureBuilder:
    """Cache board-wide maps; state tuples never contain mutable game arrays."""
    def __init__(self, version='F5'):
        if version not in ('F5','F6','F7','F8','F9','F10'):
            raise ValueError('Unknown Task-4 feature version')
        self.version = version
        self.board_key = None
        self.coin_key = None
        self.hazard_key = None
        self.hypothetical = {}

    def describe(self, state):
        field = state['field']
        position = tuple(state['self'][3])
        others = tuple(tuple(other[3]) for other in state['others'])
        bombs = tuple((tuple(p), int(t)) for p, t in state['bombs'])
        field_key = (field.shape, field.tobytes(), others)
        if field_key != self.board_key:
            self.board_key = field_key
            self.hypothetical.clear()
            self.free = field == 0
            for other in others:
                self.free[other] = False
            self.yield_map = crate_yields(field,self.free)
            targets = list(zip(*np.where(self.free & (self.yield_map > 0))))
            self.crate_distance, self.crate_target = distance_and_target(self.free, targets)
            adjacent=set()
            for x,y in others:
                for dx,dy in DELTAS[:4]:
                    p=x+dx,y+dy
                    if 0<=p[0]<field.shape[0] and 0<=p[1]<field.shape[1] and self.free[p]: adjacent.add(p)
            self.enemy_distance,self.enemy_target=distance_and_target(self.free,adjacent)
            if self.version in ('F9','F10'): self.enemy_distance=weighted_opponent_distance(field,others)
            self.coin_key = None
        coin_key = tuple(sorted(tuple(p) for p in state['coins']))
        if coin_key != self.coin_key:
            self.coin_key = coin_key
            self.coin_distance, self.coin_target = distance_and_target(self.free, coin_key)
        bomb_positions = {p for p, _ in bombs}
        allowed = []
        for dx, dy in DELTAS[:4]:
            p = position[0] + dx, position[1] + dy
            allowed.append(0 <= p[0] < field.shape[0] and 0 <= p[1] < field.shape[1]
                           and bool(self.free[p]) and p not in bomb_positions)
        allowed.extend((True, bool(state['self'][2])))
        free_mask = sum(int(value) << i for i, value in enumerate(allowed[:4]))
        tactical = bool(bombs) or bool(np.any(state['explosion_map'] > 0))
        if tactical:
            hazard_key = (field_key, bombs, state['explosion_map'].tobytes())
            if hazard_key != self.hazard_key:
                self.hazard_key = hazard_key
                self.layers = escape_layers(field, bombs, state['explosion_map'], others)
            danger, occupied, viable = self.layers
            feasible_mask = 0
            for action, (dx, dy) in enumerate(DELTAS):
                p = position[0] + dx, position[1] + dy
                if allowed[action] and not danger[1][p] and viable[1][p]:
                    feasible_mask |= 1 << action
            exposed = bool(danger[:, position[0], position[1]].any())
            urgency = min([min(3, timer) for _, timer in bombs] + ([0] if np.any(state['explosion_map'] > 0) else [3]))
            key = (1, free_mask, feasible_mask, int(exposed), urgency)
            if self.version in ('F5','F6','F7','F8','F9','F10'):
                pending_crates = {cell for p, _ in bombs for cell in blast_cells(field, p) if field[cell] == 1}
                key += (min(2, len(pending_crates)),)
                pending_exposure=any(other in blast_cells(field,p) for p,_ in bombs for other in others)
                key += (int(pending_exposure),)
            if self.version != 'F6' or exposed:
                return key, np.array(allowed, bool)
        bomb_escape = False
        if allowed[5]:
            hypothesis_key=(position,bombs,state['explosion_map'].tobytes()) if self.version=='F6' else position
            if hypothesis_key not in self.hypothetical:
                existing=list(bombs) if self.version=='F6' else []
                explosions=state['explosion_map'] if self.version=='F6' else np.zeros(field.shape)
                danger, _, viable = escape_layers(field, existing+[(position, settings.BOMB_TIMER)], explosions, others)
                self.hypothetical[hypothesis_key] = bool(not danger[1][position] and viable[1][position])
            bomb_escape = self.hypothetical[hypothesis_key]
        # Supply target coordinates selected by reachable path distance; SARSA chooses actions.
        if self.version in ('F8','F9') and self.enemy_distance[position]>=0:
            kind,target=3,(position if self.version=='F9' else self.enemy_target[position])
        elif self.coin_distance[position] >= 0:
            kind, target = 1, self.coin_target[position]
        elif self.enemy_distance[position] >= 0:
            kind, target = 3, (position if self.version=='F10' else self.enemy_target[position])
        elif self.crate_distance[position] >= 0:
            kind, target = 2, self.crate_target[position]
        else:
            kind, target = 0, position
        dx, dy = int(np.sign(target[0] - position[0])), int(np.sign(target[1] - position[1]))
        direction = (dx + 1) * 3 + dy + 1
        goal = 0 if kind == 0 else 1 + (kind - 1) * 9 + direction
        if self.version in ('F5','F6','F7','F8','F9','F10') and kind:
            distances = {1:self.coin_distance,2:self.crate_distance,3:self.enemy_distance}[kind]
            route_mask = 0
            for action, (dx, dy) in enumerate(DELTAS[:4]):
                p = position[0]+dx, position[1]+dy
                if allowed[action] and 0 <= distances[p] < distances[position]:
                    route_mask |= 1 << action
            goal = 1 + (kind-1)*16 + route_mask
        key = (0, free_mask, goal, min(2, int(self.yield_map[position])), int(bomb_escape), int(allowed[5]))
        targets_in_blast=[other for other in others if other in blast_cells(field,position)]
        exposure=int(bool(targets_in_blast))
        if self.version=='F7' and exposure and allowed[5]:
            # Optimistic opponent escape: omit player blocking; terrain stays fixed.
            # This describes an opportunity, never selects BOMB or masks a move.
            trap_key=('opponent_escape',position)
            if trap_key not in self.hypothetical:
                _,_,opponent_viable=escape_layers(field,[(position,settings.BOMB_TIMER)],np.zeros(field.shape),())
                self.hypothetical[trap_key]=any(not opponent_viable[0][other] for other in targets_in_blast)
            if self.hypothetical[trap_key]: exposure=2
        key += (exposure,)
        return key, np.array(allowed, bool)
