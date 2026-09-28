from collections import deque

import numpy as np
import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MOVE_DELTAS = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'WAIT': (0, 0),
}

# Model sizes used for checkpoint migration.
LEGACY_FEATURE_COUNT = 22
STAGE2_FEATURE_COUNT = 36
STAGE3_FEATURE_COUNT = 50

FEATURE_NAMES = [
    # Stage 1 (22)
    'bias',
    'coin_up', 'coin_right', 'coin_down', 'coin_left', 'coin_distance',
    'safe_up', 'safe_right', 'safe_down', 'safe_left', 'safe_wait',
    'danger_now',
    'bomb_available',
    'crate_up', 'crate_right', 'crate_down', 'crate_left',
    'enemy_up', 'enemy_right', 'enemy_down', 'enemy_left', 'enemy_distance',
    # Stage 2 (+14 = 36)
    'danger_soon',
    'escape_up', 'escape_right', 'escape_down', 'escape_left',
    'safe_to_bomb',
    'useful_bomb',
    'crate_target_up', 'crate_target_right', 'crate_target_down', 'crate_target_left',
    'crate_distance',
    'adjacent_crate_count',
    'bomb_crate_count',
    # Stage 3 (+14 = 50)
    'attack_up', 'attack_right', 'attack_down', 'attack_left',
    'attack_distance',
    'enemy_in_blast',
    'enemy_count_in_blast',
    'safe_attack_bomb',
    'enemy_approach_up', 'enemy_approach_right', 'enemy_approach_down', 'enemy_approach_left',
    'enemy_approach_distance',
    'enemy_adjacent',
    # Stage 4 (+14 = 64): multi-agent survival
    'mobility_up', 'mobility_right', 'mobility_down', 'mobility_left',
    'enemy_bomb_risk_up', 'enemy_bomb_risk_right', 'enemy_bomb_risk_down', 'enemy_bomb_risk_left',
    'safe_neighbor_count',
    'dead_end_now',
    'bomb_escape_route_count',
    'existing_bomb_overlap',
    'crowdedness',
    'enemy_bomb_threat_here',
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


def danger_map(game_state):
    """Earliest bomb timer affecting each tile; 5 means no known bomb threat."""
    field = game_state['field']
    danger = np.full(field.shape, 5.0, dtype=np.float32)

    for bomb_xy, timer in game_state['bombs']:
        for tile in blast_coordinates(field, bomb_xy):
            danger[tile] = min(danger[tile], float(timer))

    danger[game_state['explosion_map'] > 0] = 0.0
    return danger


def hazard_schedule(game_state, extra_bomb=None):
    """Return mapping future_step -> set of dangerous tiles."""
    field = game_state['field']
    schedule = {}

    def add(step, tiles):
        schedule.setdefault(step, set()).update(tiles)

    active_tiles = set(map(tuple, np.argwhere(game_state['explosion_map'] > 0)))
    if active_tiles:
        add(1, active_tiles)

    for bomb_xy, timer in game_state['bombs']:
        explosion_step = int(timer) + 1
        blast = blast_coordinates(field, bomb_xy)
        add(explosion_step, blast)
        add(explosion_step + 1, blast)

    if extra_bomb is not None:
        bomb_xy, explosion_step = extra_bomb
        blast = blast_coordinates(field, bomb_xy)
        add(explosion_step, blast)
        add(explosion_step + 1, blast)

    return schedule


def blocked_positions(game_state):
    blocked = {tuple(xy) for xy, _ in game_state['bombs']}
    blocked.update(tuple(opponent[3]) for opponent in game_state['others'])
    return blocked


def stable_safe(tile, start_time, schedule, horizon):
    for time_step in range(start_time, horizon + 1):
        if tile in schedule.get(time_step, set()):
            return False
    return True


def find_escape_first_step(
    game_state,
    schedule=None,
    initial_time=0,
    extra_blocked=None,
    max_time=8,
):
    """Time-aware BFS. Returns (first_step_position, distance) or (None, None)."""
    field = game_state['field']
    start = tuple(game_state['self'][3])

    if schedule is None:
        schedule = hazard_schedule(game_state)

    horizon = max([max_time] + list(schedule.keys()))
    blocked = blocked_positions(game_state)
    if extra_blocked:
        blocked.update(extra_blocked)

    if stable_safe(start, initial_time + 1, schedule, horizon):
        return None, 0

    queue = deque([(start, initial_time, None)])
    visited = {(start, initial_time)}

    while queue:
        position, time_step, first_step = queue.popleft()
        if time_step >= horizon:
            continue

        x, y = position
        next_time = time_step + 1

        for next_pos in [
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
            (x, y),
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
            visited.add(state)

            chosen_first = next_pos if first_step is None else first_step

            if stable_safe(next_pos, next_time, schedule, horizon):
                return chosen_first, next_time - initial_time

            queue.append((next_pos, next_time, chosen_first))

    return None, None


def can_survive_from(
    game_state,
    start_position,
    start_time=1,
    schedule=None,
    max_time=8,
    extra_blocked=None,
):
    """True if at least one future path survives all currently known explosions."""
    field = game_state['field']
    start_position = tuple(start_position)

    if schedule is None:
        schedule = hazard_schedule(game_state)

    horizon = max([max_time] + list(schedule.keys()))
    blocked = blocked_positions(game_state)
    if extra_blocked:
        blocked.update(tuple(pos) for pos in extra_blocked)
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
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
            (x, y),
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


def bomb_crate_count(game_state):
    field = game_state['field']
    position = tuple(game_state['self'][3])
    return sum(field[x, y] == 1 for x, y in blast_coordinates(field, position))


def adjacent_crate_count(game_state):
    field = game_state['field']
    x, y = game_state['self'][3]
    count = 0

    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if in_bounds(field, nx, ny) and field[nx, ny] == 1:
            count += 1

    return count


def enemy_positions(game_state):
    return [tuple(opponent[3]) for opponent in game_state['others']]


def bomb_enemy_count(game_state):
    """How many live opponents a bomb placed NOW would cover."""
    if not game_state['others']:
        return 0

    field = game_state['field']
    position = tuple(game_state['self'][3])
    blast = set(blast_coordinates(field, position))
    return sum(position in blast for position in enemy_positions(game_state))


def enemy_adjacent_count(game_state):
    x, y = game_state['self'][3]
    return sum(
        abs(ex - x) + abs(ey - y) == 1
        for ex, ey in enemy_positions(game_state)
    )


def nearby_enemy_count(game_state, radius=4):
    """Number of opponents within Manhattan radius of our current position."""
    if game_state is None:
        return 0

    x, y = game_state['self'][3]
    return sum(
        abs(ex - x) + abs(ey - y) <= radius
        for ex, ey in enemy_positions(game_state)
    )


def local_exit_count(game_state, position=None):
    """
    Count immediately traversable cardinal exits. This intentionally ignores
    future bomb timing; it measures geometric mobility / dead-end structure.
    """
    if game_state is None:
        return 0

    field = game_state['field']
    if position is None:
        position = tuple(game_state['self'][3])
    else:
        position = tuple(position)

    blocked = blocked_positions(game_state)
    blocked.discard(position)

    x, y = position
    count = 0
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        next_pos = (nx, ny)
        if (
            in_bounds(field, nx, ny)
            and field[nx, ny] == 0
            and next_pos not in blocked
        ):
            count += 1
    return count


def potential_enemy_bomb_risk_count(game_state, tile):
    """
    Count opponents that could drop a bomb NOW whose blast would cover tile.
    This is a risk feature, not a hard prediction that they will bomb.
    """
    if game_state is None or not game_state['others']:
        return 0

    field = game_state['field']
    tile = tuple(tile)
    count = 0

    for opponent in game_state['others']:
        bomb_available = bool(opponent[2])
        opponent_pos = tuple(opponent[3])
        if not bomb_available:
            continue

        if tile in set(blast_coordinates(field, opponent_pos)):
            count += 1

    return count


def existing_bomb_overlap_count(game_state, tile=None, urgent_timer=2):
    """Number of imminent existing bomb blast zones covering a tile."""
    if game_state is None:
        return 0

    if tile is None:
        tile = tuple(game_state['self'][3])
    else:
        tile = tuple(tile)

    field = game_state['field']
    count = 0
    for bomb_xy, timer in game_state['bombs']:
        if int(timer) <= urgent_timer and tile in set(blast_coordinates(field, bomb_xy)):
            count += 1
    return count


def safe_movement_actions(game_state):
    """Movement actions that survive all currently known explosions."""
    if game_state is None:
        return []

    field = game_state['field']
    _, _, _, (x, y) = game_state['self']
    blocked = blocked_positions(game_state)
    schedule = hazard_schedule(game_state)

    movement_actions = []
    for action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        dx, dy = MOVE_DELTAS[action]
        next_pos = (x + dx, y + dy)
        nx, ny = next_pos

        if not in_bounds(field, nx, ny):
            continue
        if field[nx, ny] != 0:
            continue
        if next_pos in blocked:
            continue
        if next_pos in schedule.get(1, set()):
            continue
        if not can_survive_from(
            game_state,
            next_pos,
            start_time=1,
            schedule=schedule,
            max_time=8,
        ):
            continue

        movement_actions.append(action)

    return movement_actions


def bomb_escape_route_count(game_state):
    """
    Number of distinct FIRST movement directions that can lead to survival
    after dropping a bomb now. Multiple routes matter in multiplayer because
    another agent can occupy or invalidate a single escape corridor.
    """
    if game_state is None:
        return 0

    _, _, bomb_available, start = game_state['self']
    if not bomb_available:
        return 0

    start = tuple(start)
    field = game_state['field']

    # A new bomb is created at timer=4 and decremented at the end of the same
    # world step. It therefore explodes on future step 5 relative to the
    # decision at which BOMB is selected.
    explosion_step = int(s.BOMB_TIMER) + 1
    schedule = hazard_schedule(
        game_state,
        extra_bomb=(start, explosion_step),
    )

    blocked = blocked_positions(game_state)
    blocked.add(start)

    x, y = start
    route_count = 0

    # Bomb is placed on future step 1; first escape movement occurs on step 2.
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
            route_count += 1

    return route_count


def bomb_is_stage4_safe(game_state):
    """
    Stage-4 v2 bomb safety.

    Keep the validated Stage-3 requirement that at least one complete escape
    path exists. Require TWO independent first-step escape routes only in
    genuinely high-pressure multiplayer states, not merely because one enemy
    happens to be nearby.
    """
    if not can_escape_after_bomb(game_state):
        return False

    routes = bomb_escape_route_count(game_state)
    if routes <= 0:
        return False

    current_position = tuple(game_state['self'][3])
    nearby = nearby_enemy_count(game_state, radius=3)
    enemy_bomb_risk_here = potential_enemy_bomb_risk_count(
        game_state,
        current_position,
    )
    bomb_busy = len(game_state['bombs']) > 0
    urgent_overlap = existing_bomb_overlap_count(
        game_state,
        current_position,
        urgent_timer=2,
    ) > 0

    high_pressure = (
        urgent_overlap
        or nearby >= 2
        or (bomb_busy and enemy_bomb_risk_here > 0)
    )

    if high_pressure:
        return routes >= 2

    return True


def bomb_target_is_safe(game_state):
    """
    Fast v2 approximation while searching candidate attack/crate positions.
    One verified escape remains acceptable in ordinary combat. Two-exit
    geometry is demanded only under stronger multiplayer pressure.
    """
    if not can_escape_after_bomb(game_state):
        return False

    current_position = tuple(game_state['self'][3])
    nearby = nearby_enemy_count(game_state, radius=3)
    enemy_bomb_risk_here = potential_enemy_bomb_risk_count(
        game_state,
        current_position,
    )
    bomb_busy = len(game_state['bombs']) > 0

    high_pressure = (
        nearby >= 2
        or (bomb_busy and enemy_bomb_risk_here > 0)
    )

    if high_pressure:
        return local_exit_count(game_state) >= 2

    return True


def can_escape_after_bomb(game_state):
    """Simulate dropping a bomb now and require a full escape route."""
    if game_state is None:
        return False

    _, _, bomb_available, start = game_state['self']
    start = tuple(start)
    if not bomb_available:
        return False

    # Keep the Stage-2 timing convention that was already validated in play.
    explosion_step = int(s.BOMB_TIMER) + 1

    schedule = hazard_schedule(
        game_state,
        extra_bomb=(start, explosion_step),
    )

    if start in schedule.get(1, set()):
        return False

    first_step, _ = find_escape_first_step(
        game_state,
        schedule=schedule,
        initial_time=1,
        extra_blocked={start},
        max_time=explosion_step + 1,
    )

    return first_step is not None


def safe_attack_bomb(game_state):
    return (
        game_state is not None
        and game_state['self'][2]
        and bomb_enemy_count(game_state) > 0
        and bomb_is_stage4_safe(game_state)
    )


def legal_actions(game_state):
    """
    Stage-4 action mask.

    Movement must survive all KNOWN bomb hazards. BOMB must be useful and pass
    the multiplayer-aware escape test. WAIT is only a tactical fallback.
    """
    if game_state is None:
        return ['WAIT']

    _, _, bomb_available, (x, y) = game_state['self']
    schedule = hazard_schedule(game_state)
    movement_actions = safe_movement_actions(game_state)
    possible = list(movement_actions)

    bomb_is_useful = (
        bomb_crate_count(game_state) > 0
        or bomb_enemy_count(game_state) > 0
    )

    if (
        bomb_available
        and bomb_is_useful
        and bomb_is_stage4_safe(game_state)
    ):
        possible.append('BOMB')

    # WAIT is not a normal navigation action. Keep it only if no survivable
    # movement is available and remaining in place itself can survive.
    if not movement_actions:
        wait_pos = (x, y)
        if (
            wait_pos not in schedule.get(1, set())
            and can_survive_from(
                game_state,
                wait_pos,
                start_time=1,
                schedule=schedule,
                max_time=8,
            )
        ):
            possible.append('WAIT')

    return possible if possible else ['WAIT']


def bfs_first_step_and_distance(field, start, targets, blocked=None):
    if not targets:
        return None, None

    start = tuple(start)
    targets = set(map(tuple, targets))
    blocked = set() if blocked is None else set(blocked)
    blocked.discard(start)

    queue = deque([start])
    parent = {start: None}
    distance = {start: 0}
    found = None

    while queue:
        current = queue.popleft()
        if current in targets:
            found = current
            break

        x, y = current
        for next_tile in [
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        ]:
            nx, ny = next_tile
            if not in_bounds(field, nx, ny):
                continue
            if field[nx, ny] != 0:
                continue
            if next_tile in blocked:
                continue
            if next_tile in parent:
                continue

            parent[next_tile] = current
            distance[next_tile] = distance[current] + 1
            queue.append(next_tile)

    if found is None:
        return None, None
    if found == start:
        return start, 0

    step = found
    while parent[step] != start:
        step = parent[step]

    return step, distance[found]


def direction_one_hot(start, first_step):
    direction = np.zeros(4, dtype=np.float32)
    start = tuple(start)

    if first_step is None or tuple(first_step) == start:
        return direction

    x, y = start
    mapping = {
        (x, y - 1): 0,
        (x + 1, y): 1,
        (x, y + 1): 2,
        (x - 1, y): 3,
    }

    index = mapping.get(tuple(first_step))
    if index is not None:
        direction[index] = 1.0

    return direction


def nearest_coin_info(game_state):
    if game_state is None or not game_state['coins']:
        return None, None

    return bfs_first_step_and_distance(
        game_state['field'],
        game_state['self'][3],
        game_state['coins'],
        blocked_positions(game_state),
    )


def nearest_coin_distance(game_state):
    _, distance = nearest_coin_info(game_state)
    return distance


def _state_with_self_position(game_state, position):
    hypothetical = dict(game_state)
    old_self = game_state['self']
    hypothetical['self'] = (
        old_self[0],
        old_self[1],
        old_self[2],
        tuple(position),
    )
    return hypothetical


def nearest_crate_target_info(game_state):
    """Nearest reachable position from which a useful, escapable crate bomb can be placed."""
    if game_state is None:
        return None, None

    field = game_state['field']
    start = tuple(game_state['self'][3])
    blocked = blocked_positions(game_state)
    blocked.discard(start)

    queue = deque([start])
    parent = {start: None}
    distance = {start: 0}

    while queue:
        current = queue.popleft()
        hypothetical = _state_with_self_position(game_state, current)

        if (
            bomb_crate_count(hypothetical) > 0
            and bomb_target_is_safe(hypothetical)
        ):
            if current == start:
                return start, 0

            step = current
            while parent[step] != start:
                step = parent[step]
            return step, distance[current]

        x, y = current
        for next_tile in [
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        ]:
            nx, ny = next_tile
            if not in_bounds(field, nx, ny):
                continue
            if field[nx, ny] != 0:
                continue
            if next_tile in blocked:
                continue
            if next_tile in parent:
                continue

            parent[next_tile] = current
            distance[next_tile] = distance[current] + 1
            queue.append(next_tile)

    return None, None


def nearest_crate_distance(game_state):
    _, distance = nearest_crate_target_info(game_state)
    return distance


def _candidate_attack_positions(game_state):
    """Small set of bomb positions that could hit an opponent if a bomb were placed there."""
    field = game_state['field']
    blocked = blocked_positions(game_state)
    my_pos = tuple(game_state['self'][3])
    candidates = set()

    for ex, ey in enemy_positions(game_state):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for distance in range(1, s.BOMB_POWER + 1):
                px = ex + dx * distance
                py = ey + dy * distance

                if not in_bounds(field, px, py):
                    break
                if field[px, py] == -1:
                    break

                pos = (px, py)
                if field[px, py] != 0:
                    continue
                if pos in blocked and pos != my_pos:
                    continue

                candidates.add(pos)

    # Current position can also be an attack position.
    if bomb_enemy_count(game_state) > 0:
        candidates.add(my_pos)

    return candidates


def nearest_attack_target_info(game_state):
    """
    Find nearest reachable SAFE bomb position from which at least one opponent
    is in blast range. Returns (first_step, distance).
    """
    if game_state is None or not game_state['others']:
        return None, None

    field = game_state['field']
    start = tuple(game_state['self'][3])
    blocked = blocked_positions(game_state)
    blocked.discard(start)
    candidates = _candidate_attack_positions(game_state)

    if not candidates:
        return None, None

    queue = deque([start])
    parent = {start: None}
    distance = {start: 0}

    while queue:
        current = queue.popleft()

        if current in candidates:
            hypothetical = _state_with_self_position(game_state, current)
            if (
                bomb_enemy_count(hypothetical) > 0
                and bomb_target_is_safe(hypothetical)
            ):
                if current == start:
                    return start, 0

                step = current
                while parent[step] != start:
                    step = parent[step]
                return step, distance[current]

        x, y = current
        for next_tile in [
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        ]:
            nx, ny = next_tile
            if not in_bounds(field, nx, ny):
                continue
            if field[nx, ny] != 0:
                continue
            if next_tile in blocked:
                continue
            if next_tile in parent:
                continue

            parent[next_tile] = current
            distance[next_tile] = distance[current] + 1
            queue.append(next_tile)

    return None, None


def nearest_attack_distance(game_state):
    _, distance = nearest_attack_target_info(game_state)
    return distance


def nearest_enemy_approach_info(game_state):
    """Fallback target: nearest free tile adjacent to an opponent."""
    if game_state is None or not game_state['others']:
        return None, None

    field = game_state['field']
    targets = set()
    occupied = blocked_positions(game_state)

    for ex, ey in enemy_positions(game_state):
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = ex + dx, ey + dy
            pos = (nx, ny)
            if (
                in_bounds(field, nx, ny)
                and field[nx, ny] == 0
                and pos not in occupied
            ):
                targets.add(pos)

    return bfs_first_step_and_distance(
        field,
        game_state['self'][3],
        targets,
        blocked_positions(game_state),
    )


def nearest_enemy_approach_distance(game_state):
    _, distance = nearest_enemy_approach_info(game_state)
    return distance


def enemy_direction(game_state):
    opponents = enemy_positions(game_state)
    if not opponents:
        return np.zeros(4, dtype=np.float32), 0.0

    x, y = game_state['self'][3]
    nearest_enemy = min(
        opponents,
        key=lambda enemy: abs(enemy[0] - x) + abs(enemy[1] - y),
    )

    dx = nearest_enemy[0] - x
    dy = nearest_enemy[1] - y
    direction = np.zeros(4, dtype=np.float32)

    if abs(dx) >= abs(dy) and dx != 0:
        direction[1 if dx > 0 else 3] = 1.0
    elif dy != 0:
        direction[2 if dy > 0 else 0] = 1.0

    distance = min(abs(dx) + abs(dy), 30) / 30.0
    return direction, distance


def state_to_features(game_state):
    if game_state is None:
        return None

    field = game_state['field']
    _, _, bomb_available, (x, y) = game_state['self']

    coin_step, coin_distance = nearest_coin_info(game_state)
    coin_direction = direction_one_hot((x, y), coin_step)
    normalized_coin_distance = 1.0 if coin_distance is None else min(coin_distance, 30) / 30.0

    legal = set(legal_actions(game_state))
    safe_moves = np.array([
        float('UP' in legal),
        float('RIGHT' in legal),
        float('DOWN' in legal),
        float('LEFT' in legal),
        float('WAIT' in legal),
    ], dtype=np.float32)

    raw_danger = danger_map(game_state)
    danger_now = float(
        raw_danger[x, y] <= 1
        or game_state['explosion_map'][x, y] > 0
    )
    danger_soon = float(raw_danger[x, y] <= 2)

    crate_features = np.array([
        float(in_bounds(field, x, y - 1) and field[x, y - 1] == 1),
        float(in_bounds(field, x + 1, y) and field[x + 1, y] == 1),
        float(in_bounds(field, x, y + 1) and field[x, y + 1] == 1),
        float(in_bounds(field, x - 1, y) and field[x - 1, y] == 1),
    ], dtype=np.float32)

    enemy_dir, enemy_distance = enemy_direction(game_state)

    escape_step, _ = find_escape_first_step(game_state)
    escape_direction = direction_one_hot((x, y), escape_step)

    safe_to_bomb = float(can_escape_after_bomb(game_state))
    crates_hit = bomb_crate_count(game_state)
    useful_bomb = float(safe_to_bomb and crates_hit > 0)

    crate_step, crate_distance = nearest_crate_target_info(game_state)
    crate_direction = direction_one_hot((x, y), crate_step)
    normalized_crate_distance = 1.0 if crate_distance is None else min(crate_distance, 30) / 30.0

    adjacent_normalized = min(adjacent_crate_count(game_state), 4) / 4.0
    bomb_crates_normalized = min(crates_hit, 4) / 4.0

    # Stage 3 combat features.
    attack_step, attack_distance = nearest_attack_target_info(game_state)
    attack_direction = direction_one_hot((x, y), attack_step)
    normalized_attack_distance = 1.0 if attack_distance is None else min(attack_distance, 30) / 30.0

    enemies_hit = bomb_enemy_count(game_state)
    enemy_in_blast = float(enemies_hit > 0)
    enemy_count_in_blast = min(enemies_hit, 3) / 3.0
    safe_attack = float(safe_attack_bomb(game_state))

    approach_step, approach_distance = nearest_enemy_approach_info(game_state)
    approach_direction = direction_one_hot((x, y), approach_step)
    normalized_approach_distance = 1.0 if approach_distance is None else min(approach_distance, 30) / 30.0

    enemy_adjacent = float(enemy_adjacent_count(game_state) > 0)

    # Stage 4 multi-agent survival features.
    current_position = (x, y)
    mobility = []
    enemy_bomb_risk = []
    for action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        dx, dy = MOVE_DELTAS[action]
        next_pos = (x + dx, y + dy)
        nx, ny = next_pos

        if not in_bounds(field, nx, ny) or field[nx, ny] != 0:
            mobility.append(0.0)
            enemy_bomb_risk.append(1.0)
            continue

        hypothetical = _state_with_self_position(game_state, next_pos)
        mobility.append(min(local_exit_count(hypothetical, next_pos), 4) / 4.0)
        enemy_bomb_risk.append(
            min(potential_enemy_bomb_risk_count(game_state, next_pos), 3) / 3.0
        )

    safe_neighbors = sum(
        float(action in legal)
        for action in ['UP', 'RIGHT', 'DOWN', 'LEFT']
    ) / 4.0
    dead_end_now = float(local_exit_count(game_state, current_position) <= 1)
    bomb_useful_now = (crates_hit > 0 or enemies_hit > 0)
    escape_routes = (
        min(bomb_escape_route_count(game_state), 4) / 4.0
        if bomb_available and bomb_useful_now
        else 0.0
    )
    overlap = min(existing_bomb_overlap_count(game_state, current_position), 3) / 3.0
    crowdedness = min(nearby_enemy_count(game_state, radius=4), 3) / 3.0
    threat_here = min(
        potential_enemy_bomb_risk_count(game_state, current_position), 3
    ) / 3.0

    features = np.concatenate([
        # Stage 1 (22)
        np.array([1.0], dtype=np.float32),
        coin_direction,
        np.array([normalized_coin_distance], dtype=np.float32),
        safe_moves,
        np.array([danger_now, float(bomb_available)], dtype=np.float32),
        crate_features,
        enemy_dir,
        np.array([enemy_distance], dtype=np.float32),
        # Stage 2 (+14)
        np.array([danger_soon], dtype=np.float32),
        escape_direction,
        np.array([safe_to_bomb, useful_bomb], dtype=np.float32),
        crate_direction,
        np.array([
            normalized_crate_distance,
            adjacent_normalized,
            bomb_crates_normalized,
        ], dtype=np.float32),
        # Stage 3 (+14)
        attack_direction,
        np.array([
            normalized_attack_distance,
            enemy_in_blast,
            enemy_count_in_blast,
            safe_attack,
        ], dtype=np.float32),
        approach_direction,
        np.array([
            normalized_approach_distance,
            enemy_adjacent,
        ], dtype=np.float32),
        # Stage 4 (+14)
        np.array(mobility, dtype=np.float32),
        np.array(enemy_bomb_risk, dtype=np.float32),
        np.array([
            safe_neighbors,
            dead_end_now,
            escape_routes,
            overlap,
            crowdedness,
            threat_here,
        ], dtype=np.float32),
    ])

    return features.astype(np.float32)
