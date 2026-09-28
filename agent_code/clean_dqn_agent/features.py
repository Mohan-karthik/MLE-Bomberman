from collections import deque

import numpy as np
import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVE_ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
MOVE_DELTAS = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'WAIT': (0, 0),
}

# IMPORTANT COMPLIANCE DESIGN:
# These are descriptive state features only. None of them is used to add a
# hand-written score to an action at inference time. The network learns the
# value of every feature from experience.
FEATURE_NAMES = [
    # Global state (13)
    'bias',
    'step_fraction',
    'bomb_available',
    'current_danger_level',
    'current_imminent_danger',
    'current_exit_count',
    'enemy_bomb_risk_here',
    'existing_bomb_overlap_here',
    'crowdedness',
    'active_bomb_count',
    'visible_coin_count',
    'remaining_crate_count',
    'opponent_count',

    # Nearest coin pathfinding (5)
    'coin_up', 'coin_right', 'coin_down', 'coin_left', 'coin_distance',

    # Nearest crate-adjacent free tile (5)
    'crate_up', 'crate_right', 'crate_down', 'crate_left', 'crate_distance',

    # Nearest opponent-adjacent free tile (5)
    'enemy_up', 'enemy_right', 'enemy_down', 'enemy_left', 'enemy_distance',

    # Bomb-context descriptors (7)
    'bomb_crate_count',
    'bomb_enemy_count',
    'bomb_escape_route_count',
    'bomb_can_escape',
    'bomb_hits_something',
    'adjacent_crate_count',
    'adjacent_enemy_count',

    # Action-specific descriptors for each movement direction (10 x 4)
    # UP
    'up_free', 'up_danger_level', 'up_imminent_danger', 'up_survivable',
    'up_exit_count', 'up_enemy_bomb_risk', 'up_bomb_overlap',
    'up_coin_alignment', 'up_crate_alignment', 'up_enemy_alignment',
    # RIGHT
    'right_free', 'right_danger_level', 'right_imminent_danger', 'right_survivable',
    'right_exit_count', 'right_enemy_bomb_risk', 'right_bomb_overlap',
    'right_coin_alignment', 'right_crate_alignment', 'right_enemy_alignment',
    # DOWN
    'down_free', 'down_danger_level', 'down_imminent_danger', 'down_survivable',
    'down_exit_count', 'down_enemy_bomb_risk', 'down_bomb_overlap',
    'down_coin_alignment', 'down_crate_alignment', 'down_enemy_alignment',
    # LEFT
    'left_free', 'left_danger_level', 'left_imminent_danger', 'left_survivable',
    'left_exit_count', 'left_enemy_bomb_risk', 'left_bomb_overlap',
    'left_coin_alignment', 'left_crate_alignment', 'left_enemy_alignment',
]


def in_bounds(field, x, y):
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def blast_coordinates(field, bomb_xy, power=None):
    """Match the supplied framework: only stone walls (-1) stop blasts."""
    if power is None:
        power = s.BOMB_POWER

    x, y = bomb_xy
    coords = [(x, y)]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for distance in range(1, power + 1):
            nx = x + dx * distance
            ny = y + dy * distance
            if not in_bounds(field, nx, ny):
                break
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def blocked_positions(game_state):
    blocked = {tuple(xy) for xy, _ in game_state['bombs']}
    blocked.update(tuple(opponent[3]) for opponent in game_state['others'])
    return blocked


def mechanically_legal_actions(game_state):
    """
    Mechanical legality only.

    No action is removed because it is dangerous, tactically useless, or
    strategically undesirable. This function mirrors what the environment can
    execute: free-tile movement, WAIT, and BOMB when one is available.
    """
    if game_state is None:
        return ['WAIT']

    field = game_state['field']
    _, _, bomb_available, (x, y) = game_state['self']
    blocked = blocked_positions(game_state)

    legal = []
    for action in MOVE_ACTIONS:
        dx, dy = MOVE_DELTAS[action]
        nx, ny = x + dx, y + dy
        if (
            in_bounds(field, nx, ny)
            and field[nx, ny] == 0
            and (nx, ny) not in blocked
        ):
            legal.append(action)

    # Waiting is always mechanically valid.
    legal.append('WAIT')

    if bomb_available:
        legal.append('BOMB')

    return legal


def danger_map(game_state):
    """Earliest visible bomb timer covering each tile; 5 means no known threat."""
    field = game_state['field']
    danger = np.full(field.shape, 5.0, dtype=np.float32)

    for bomb_xy, timer in game_state['bombs']:
        for tile in blast_coordinates(field, tuple(bomb_xy)):
            danger[tile] = min(danger[tile], float(timer))

    danger[game_state['explosion_map'] > 0] = 0.0
    return danger


def hazard_schedule(game_state, extra_bomb=None):
    """Map future world-update index -> tiles dangerous after that update."""
    field = game_state['field']
    schedule = {}

    def add(step, tiles):
        schedule.setdefault(step, set()).update(tiles)

    active = set(map(tuple, np.argwhere(game_state['explosion_map'] > 0)))
    if active:
        add(1, active)

    # At a state with bomb timer t, the framework explodes it after t+1 more
    # world updates because it checks <=0 before decrementing.
    for bomb_xy, timer in game_state['bombs']:
        explosion_step = int(timer) + 1
        blast = blast_coordinates(field, tuple(bomb_xy))
        add(explosion_step, blast)
        add(explosion_step + 1, blast)

    if extra_bomb is not None:
        bomb_xy, explosion_step = extra_bomb
        blast = blast_coordinates(field, tuple(bomb_xy))
        add(explosion_step, blast)
        add(explosion_step + 1, blast)

    return schedule


def stable_safe(tile, start_time, schedule, horizon):
    for t in range(start_time, horizon + 1):
        if tile in schedule.get(t, set()):
            return False
    return True


def can_survive_from(
    game_state,
    start_position,
    start_time=1,
    schedule=None,
    max_time=8,
    extra_blocked=None,
):
    """Descriptive life-saving feature: does any route survive known bombs?"""
    field = game_state['field']
    start_position = tuple(start_position)

    if schedule is None:
        schedule = hazard_schedule(game_state)

    horizon = max([max_time] + list(schedule.keys()))
    blocked = blocked_positions(game_state)
    if extra_blocked:
        blocked.update(tuple(p) for p in extra_blocked)
    blocked.discard(start_position)

    if start_position in schedule.get(start_time, set()):
        return False
    if stable_safe(start_position, start_time, schedule, horizon):
        return True

    queue = deque([(start_position, start_time)])
    visited = {(start_position, start_time)}

    while queue:
        position, time_step = queue.popleft()
        if time_step >= horizon:
            continue

        x, y = position
        next_time = time_step + 1
        for next_pos in [
            (x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)
        ]:
            nx, ny = next_pos
            if not in_bounds(field, nx, ny):
                continue
            if field[nx, ny] != 0:
                continue
            if next_pos in blocked:
                continue
            if next_pos in schedule.get(next_time, set()):
                continue

            state = (next_pos, next_time)
            if state in visited:
                continue
            if stable_safe(next_pos, next_time, schedule, horizon):
                return True

            visited.add(state)
            queue.append(state)

    return False


def local_exit_count(game_state, position=None):
    if game_state is None:
        return 0

    field = game_state['field']
    position = tuple(game_state['self'][3] if position is None else position)
    blocked = blocked_positions(game_state)
    blocked.discard(position)

    x, y = position
    count = 0
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if (
            in_bounds(field, nx, ny)
            and field[nx, ny] == 0
            and (nx, ny) not in blocked
        ):
            count += 1
    return count


def existing_bomb_overlap_count(game_state, tile, urgent_timer=2):
    field = game_state['field']
    tile = tuple(tile)
    count = 0
    for bomb_xy, timer in game_state['bombs']:
        if int(timer) <= urgent_timer and tile in set(blast_coordinates(field, tuple(bomb_xy))):
            count += 1
    return count


def enemy_positions(game_state):
    return [tuple(opponent[3]) for opponent in game_state['others']]


def potential_enemy_bomb_risk_count(game_state, tile):
    """How many opponents could, if they bombed now, cover this tile?"""
    if game_state is None:
        return 0

    field = game_state['field']
    tile = tuple(tile)
    count = 0
    for opponent in game_state['others']:
        bomb_available = bool(opponent[2])
        opponent_pos = tuple(opponent[3])
        if bomb_available and tile in set(blast_coordinates(field, opponent_pos)):
            count += 1
    return count


def nearby_enemy_count(game_state, radius=4):
    x, y = game_state['self'][3]
    return sum(
        abs(ex - x) + abs(ey - y) <= radius
        for ex, ey in enemy_positions(game_state)
    )


def bomb_crate_count(game_state):
    field = game_state['field']
    position = tuple(game_state['self'][3])
    return sum(field[x, y] == 1 for x, y in blast_coordinates(field, position))


def bomb_enemy_count(game_state):
    field = game_state['field']
    blast = set(blast_coordinates(field, tuple(game_state['self'][3])))
    return sum(enemy in blast for enemy in enemy_positions(game_state))


def adjacent_crate_count(game_state):
    field = game_state['field']
    x, y = game_state['self'][3]
    return sum(
        in_bounds(field, x + dx, y + dy) and field[x + dx, y + dy] == 1
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
    )


def adjacent_enemy_count(game_state):
    x, y = game_state['self'][3]
    return sum(abs(ex - x) + abs(ey - y) == 1 for ex, ey in enemy_positions(game_state))


def bomb_escape_route_count(game_state):
    """Number of distinct first movement directions that can escape our new bomb."""
    if game_state is None:
        return 0

    _, _, bomb_available, start = game_state['self']
    if not bomb_available:
        return 0

    start = tuple(start)
    field = game_state['field']

    # A bomb created with timer=4 is first checked at 4, then decremented on
    # the same update; it explodes on the fifth world update from this state.
    explosion_step = int(s.BOMB_TIMER) + 1
    schedule = hazard_schedule(game_state, extra_bomb=(start, explosion_step))

    x, y = start
    blocked = blocked_positions(game_state)
    blocked.add(start)
    routes = 0

    # After choosing BOMB, the agent remains on start at time 1. Its first
    # movement can place it on a neighbor at time 2.
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        next_pos = (x + dx, y + dy)
        nx, ny = next_pos
        if not in_bounds(field, nx, ny):
            continue
        if field[nx, ny] != 0:
            continue
        if next_pos in blocked:
            continue
        if next_pos in schedule.get(2, set()):
            continue
        if can_survive_from(
            game_state,
            next_pos,
            start_time=2,
            schedule=schedule,
            max_time=explosion_step + 1,
            extra_blocked={start},
        ):
            routes += 1

    return routes


def can_escape_after_bomb(game_state):
    return bomb_escape_route_count(game_state) > 0


def bfs_first_step_and_distance(field, start, targets, blocked=None):
    if not targets:
        return None, None

    start = tuple(start)
    targets = set(map(tuple, targets))
    blocked = set() if blocked is None else set(blocked)
    blocked.discard(start)

    if start in targets:
        return start, 0

    queue = deque([start])
    parent = {start: None}
    distance = {start: 0}
    found = None

    while queue:
        current = queue.popleft()
        x, y = current
        for nxt in [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]:
            nx, ny = nxt
            if not in_bounds(field, nx, ny):
                continue
            if field[nx, ny] != 0:
                continue
            if nxt in blocked or nxt in parent:
                continue
            parent[nxt] = current
            distance[nxt] = distance[current] + 1
            if nxt in targets:
                found = nxt
                queue.clear()
                break
            queue.append(nxt)

    if found is None:
        return None, None

    step = found
    while parent[step] != start:
        step = parent[step]
    return step, distance[found]


def direction_one_hot(start, first_step):
    result = np.zeros(4, dtype=np.float32)
    if first_step is None or tuple(first_step) == tuple(start):
        return result

    x, y = start
    mapping = {
        (x, y - 1): 0,
        (x + 1, y): 1,
        (x, y + 1): 2,
        (x - 1, y): 3,
    }
    index = mapping.get(tuple(first_step))
    if index is not None:
        result[index] = 1.0
    return result


def nearest_coin_info(game_state):
    if game_state is None or not game_state['coins']:
        return None, None
    return bfs_first_step_and_distance(
        game_state['field'],
        game_state['self'][3],
        game_state['coins'],
        blocked_positions(game_state),
    )


def crate_target_tiles(game_state):
    """All mechanically reachable-style free squares adjacent to any crate."""
    field = game_state['field']
    targets = set()
    for cx, cy in np.argwhere(field == 1):
        cx, cy = int(cx), int(cy)
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if in_bounds(field, nx, ny) and field[nx, ny] == 0:
                targets.add((nx, ny))
    return targets


def nearest_crate_info(game_state):
    targets = crate_target_tiles(game_state)
    if not targets:
        return None, None
    return bfs_first_step_and_distance(
        game_state['field'],
        game_state['self'][3],
        targets,
        blocked_positions(game_state),
    )


def enemy_target_tiles(game_state):
    """Free squares adjacent to opponents; no tactical safety judgement."""
    if not game_state['others']:
        return set()

    field = game_state['field']
    blocked = blocked_positions(game_state)
    targets = set()
    for ex, ey in enemy_positions(game_state):
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = ex + dx, ey + dy
            pos = (nx, ny)
            if (
                in_bounds(field, nx, ny)
                and field[nx, ny] == 0
                and pos not in blocked
            ):
                targets.add(pos)
    return targets


def nearest_enemy_info(game_state):
    targets = enemy_target_tiles(game_state)
    if not targets:
        return None, None
    return bfs_first_step_and_distance(
        game_state['field'],
        game_state['self'][3],
        targets,
        blocked_positions(game_state),
    )


def _norm_distance(distance, cap=30):
    return 1.0 if distance is None else min(float(distance), cap) / float(cap)


def _norm_danger(raw_value):
    # 1.0 means no known bomb threat; 0.0 means explosion/current detonation.
    return min(max(float(raw_value), 0.0), 5.0) / 5.0


def state_to_features(game_state):
    if game_state is None:
        return None

    field = game_state['field']
    _, _, bomb_available, (x, y) = game_state['self']
    current = (x, y)
    danger = danger_map(game_state)
    schedule = hazard_schedule(game_state)
    blocked = blocked_positions(game_state)

    coin_step, coin_distance = nearest_coin_info(game_state)
    crate_step, crate_distance = nearest_crate_info(game_state)
    enemy_step, enemy_distance = nearest_enemy_info(game_state)

    coin_dir = direction_one_hot(current, coin_step)
    crate_dir = direction_one_hot(current, crate_step)
    enemy_dir = direction_one_hot(current, enemy_step)

    global_features = np.array([
        1.0,
        min(float(game_state.get('step', 0)), float(s.MAX_STEPS)) / float(s.MAX_STEPS),
        float(bool(bomb_available)),
        _norm_danger(danger[current]),
        float(danger[current] <= 1 or game_state['explosion_map'][current] > 0),
        min(local_exit_count(game_state, current), 4) / 4.0,
        min(potential_enemy_bomb_risk_count(game_state, current), 3) / 3.0,
        min(existing_bomb_overlap_count(game_state, current), 3) / 3.0,
        min(nearby_enemy_count(game_state, radius=4), 3) / 3.0,
        min(len(game_state['bombs']), 4) / 4.0,
        min(len(game_state['coins']), 10) / 10.0,
        min(int(np.sum(field == 1)), 120) / 120.0,
        min(len(game_state['others']), 3) / 3.0,
    ], dtype=np.float32)

    bomb_crates = bomb_crate_count(game_state)
    bomb_enemies = bomb_enemy_count(game_state)
    bomb_routes = bomb_escape_route_count(game_state) if bomb_available else 0
    bomb_features = np.array([
        min(bomb_crates, 6) / 6.0,
        min(bomb_enemies, 3) / 3.0,
        min(bomb_routes, 4) / 4.0,
        float(bool(bomb_routes > 0)),
        float(bool(bomb_crates > 0 or bomb_enemies > 0)),
        min(adjacent_crate_count(game_state), 4) / 4.0,
        min(adjacent_enemy_count(game_state), 4) / 4.0,
    ], dtype=np.float32)

    movement_features = []
    target_dirs = {
        'coin': coin_dir,
        'crate': crate_dir,
        'enemy': enemy_dir,
    }

    for action_index, action in enumerate(MOVE_ACTIONS):
        dx, dy = MOVE_DELTAS[action]
        next_pos = (x + dx, y + dy)
        nx, ny = next_pos
        mechanically_free = (
            in_bounds(field, nx, ny)
            and field[nx, ny] == 0
            and next_pos not in blocked
        )

        if mechanically_free:
            survivable = can_survive_from(
                game_state,
                next_pos,
                start_time=1,
                schedule=schedule,
                max_time=8,
            )
            exit_count = local_exit_count(game_state, next_pos)
            enemy_risk = potential_enemy_bomb_risk_count(game_state, next_pos)
            overlap = existing_bomb_overlap_count(game_state, next_pos)
            danger_level = _norm_danger(danger[next_pos])
            imminent = float(
                danger[next_pos] <= 1
                or game_state['explosion_map'][next_pos] > 0
            )
        else:
            survivable = False
            exit_count = 0
            enemy_risk = 3
            overlap = 3
            danger_level = 0.0
            imminent = 1.0

        movement_features.extend([
            float(mechanically_free),
            danger_level,
            imminent,
            float(bool(survivable)),
            min(exit_count, 4) / 4.0,
            min(enemy_risk, 3) / 3.0,
            min(overlap, 3) / 3.0,
            float(target_dirs['coin'][action_index]),
            float(target_dirs['crate'][action_index]),
            float(target_dirs['enemy'][action_index]),
        ])

    features = np.concatenate([
        global_features,
        coin_dir,
        np.array([_norm_distance(coin_distance)], dtype=np.float32),
        crate_dir,
        np.array([_norm_distance(crate_distance)], dtype=np.float32),
        enemy_dir,
        np.array([_norm_distance(enemy_distance)], dtype=np.float32),
        bomb_features,
        np.asarray(movement_features, dtype=np.float32),
    ])

    assert len(features) == len(FEATURE_NAMES), (len(features), len(FEATURE_NAMES))
    return features.astype(np.float32)
