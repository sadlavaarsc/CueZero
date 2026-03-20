"""
Mock 台球环境 - 简单逻辑模拟击球结果

由于底层 API 只能返回击球前后的离散状态快照，
本模块使用简单规则推算击球结果。
"""

import math
import random
from typing import Any


def find_nearest_ball_in_direction(cue_pos: list[float], direction: float, balls: dict) -> dict | None:
    """
    找到击球方向上最近的球

    Args:
        cue_pos: 白球位置 [x, y]
        direction: 击球角度（弧度）
        balls: 球字典

    Returns:
        最近的球信息或 None
    """
    # 击球方向向量
    dx = math.cos(direction)
    dy = math.sin(direction)

    nearest = None
    min_distance = float('inf')

    for ball_id, ball_data in balls.items():
        if ball_id == 'cue' or ball_id == 'cue2':
            continue

        ball_pos = ball_data['pos']
        # 从白球到目标球的向量
        to_ball_x = ball_pos[0] - cue_pos[0]
        to_ball_y = ball_pos[1] - cue_pos[1]
        dist = math.sqrt(to_ball_x**2 + to_ball_y**2)

        if dist < 0.1:  # 忽略重叠
            continue

        # 归一化
        to_ball_x /= dist
        to_ball_y /= dist

        # 计算方向夹角（点积）
        dot = dx * to_ball_x + dy * to_ball_y

        # 如果在击球方向上（夹角小于 15 度）
        if dot > math.cos(math.radians(15)):
            if dist < min_distance:
                min_distance = dist
                nearest = {'id': ball_id, 'data': ball_data, 'distance': dist}

    return nearest


def calculate_hit_probability(distance: float, power: float) -> float:
    """
    计算进球概率

    Args:
        distance: 目标球距离
        power: 击球力度

    Returns:
        进球概率 (0-1)
    """
    # 基础概率随距离递减
    base_prob = max(0.3, 1.0 - distance / 10.0)

    # 力度适中时概率最高
    optimal_power = 50.0
    power_factor = 1.0 - abs(power - optimal_power) / 100.0
    power_factor = max(0.5, min(1.0, power_factor))

    return base_prob * power_factor


def calculate_stop_position(cue_pos: list[float], direction: float, power: float, table_bounds: dict) -> list[float]:
    """
    计算白球反弹后停止位置

    Args:
        cue_pos: 白球初始位置
        direction: 击球角度
        power: 击球力度

    Returns:
        停止位置 [x, y]
    """
    # 模拟白球运动
    x, y = cue_pos[0], cue_pos[1]
    dx = math.cos(direction) * power * 0.1
    dy = math.sin(direction) * power * 0.1

    # 简单模拟几次反弹
    for _ in range(3):
        x += dx * 5
        y += dy * 5

        # 边界反弹
        if x < table_bounds['left']:
            x = table_bounds['left'] + abs(x - table_bounds['left'])
            dx = -dx * 0.7
        elif x > table_bounds['right']:
            x = table_bounds['right'] - abs(x - table_bounds['right'])
            dx = -dx * 0.7

        if y < table_bounds['bottom']:
            y = table_bounds['bottom'] + abs(y - table_bounds['bottom'])
            dy = -dy * 0.7
        elif y > table_bounds['top']:
            y = table_bounds['top'] - abs(y - table_bounds['top'])
            dy = -dy * 0.7

        # 摩擦力减速
        dx *= 0.8
        dy *= 0.8

        if abs(dx) < 0.01 and abs(dy) < 0.01:
            break

    return [x, y]


def simulate_shot(action: dict, state: dict) -> dict:
    """
    模拟击球结果

    Args:
        action: 击球动作 {'phi': 角度，'V0': 力度}
        state: 当前状态

    Returns:
        新状态
    """
    import copy
    new_state = copy.deepcopy(state)

    cue_ball = new_state['balls']['cue']
    direction = action['phi']
    power = action['V0']

    # 1. 判断是否瞄准目标球
    target = find_nearest_ball_in_direction(cue_ball['pos'], direction, new_state['balls'])

    pocketed_balls = []

    # 2. 根据力度和距离计算进球概率
    if target and power > 10.0:
        hit_prob = calculate_hit_probability(target['distance'], power)
        if random.random() < hit_prob:
            # 目标球进袋
            target['data']['pocketed'] = True
            del new_state['balls'][target['id']]
            pocketed_balls.append(target['id'])

            # 更新得分
            team = target['data'].get('team', 'neutral')
            if team == 'red':
                new_state['red_score'] = new_state.get('red_score', 0) + 1
            elif team == 'yellow':
                new_state['yellow_score'] = new_state.get('yellow_score', 0) + 1
            elif team == 'neutral':
                new_state['score'] = new_state.get('score', 0) + 10  # 黑球 10 分

    # 3. 计算白球停止位置
    table_bounds = state.get('table', {'left': -7, 'right': 7, 'bottom': -3.5, 'top': 3.5})
    cue_ball['pos'] = calculate_stop_position(cue_ball['pos'], direction, power, table_bounds)

    # 4. 添加一些随机扰动
    cue_ball['pos'][0] += random.uniform(-0.2, 0.2)
    cue_ball['pos'][1] += random.uniform(-0.2, 0.2)

    # 确保在白球在台面内
    cue_ball['pos'][0] = max(table_bounds['left'] + 0.5, min(table_bounds['right'] - 0.5, cue_ball['pos'][0]))
    cue_ball['pos'][1] = max(table_bounds['bottom'] + 0.5, min(table_bounds['top'] - 0.5, cue_ball['pos'][1]))

    # 5. 更新回合
    new_state['turn'] = state.get('turn', 1) + 1

    return new_state


def get_initial_state() -> dict:
    """
    获取初始台球状态 - 标准台球配置

    白球：右侧（适合右手习惯）
    红方（左侧）：7 个球 (1-7)
    黄方（右侧）：7 个球 (9-15)
    黑八：中心
    """
    return {
        'balls': {
            # 白球（玩家，右侧 - 适合右手习惯）
            'cue': {'pos': [4.0, 0.0], 'pocketed': False},
            # 红方球（左侧）- 1 到 7 号
            '1': {'pos': [-3.0, 0.0], 'pocketed': False, 'team': 'red'},
            '2': {'pos': [-4.0, 0.8], 'pocketed': False, 'team': 'red'},
            '3': {'pos': [-4.0, -0.8], 'pocketed': False, 'team': 'red'},
            '4': {'pos': [-5.0, 1.6], 'pocketed': False, 'team': 'red'},
            '5': {'pos': [-5.0, 0.0], 'pocketed': False, 'team': 'red'},
            '6': {'pos': [-5.0, -1.6], 'pocketed': False, 'team': 'red'},
            '7': {'pos': [-6.0, 0.8], 'pocketed': False, 'team': 'red'},
            # 黄方球（右侧，但避开白球位置）- 9 到 15 号
            '9': {'pos': [0.0, 0.0], 'pocketed': False, 'team': 'yellow'},
            '10': {'pos': [-1.0, 0.8], 'pocketed': False, 'team': 'yellow'},
            '11': {'pos': [-1.0, -0.8], 'pocketed': False, 'team': 'yellow'},
            '12': {'pos': [-2.0, 1.6], 'pocketed': False, 'team': 'yellow'},
            '13': {'pos': [-2.0, 0.0], 'pocketed': False, 'team': 'yellow'},
            '14': {'pos': [-2.0, -1.6], 'pocketed': False, 'team': 'yellow'},
            '15': {'pos': [-3.0, 0.8], 'pocketed': False, 'team': 'yellow'},
            # 黑八（8 号球，中心偏左）
            '8': {'pos': [-1.5, 0.0], 'pocketed': False, 'team': 'neutral'},
        },
        'table': {
            'left': -7,
            'right': 7,
            'bottom': -3.5,
            'top': 3.5
        },
        'turn': 1,
        'player': 'user',
        'score': 0,
        'red_score': 0,
        'yellow_score': 0
    }
