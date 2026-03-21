"""
CueZero Web Server - Enhanced with Battle API

支持多种 Agent 对战：
- human: 人工操作
- mcts_fast: MCTS 快速模式
- mcts_full: MCTS 完整模式
- policy: 策略网络
- basic: 基础规则 Agent
- random: 随机 Agent
"""

import argparse
import copy
import time
import uuid
import os
import sys
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 导入 mock 环境
from server.mock_env import get_initial_state, simulate_shot

# 导入 AI 模块
from cuezero.models.dual_network import DualNetwork
from cuezero.mcts.search import MCTS
from cuezero.env.billiards_env import BilliardsEnv
from cuezero.inference.agent import (
    Agent, MCTSAgent, PolicyAgent,
    HumanAgent, BasicAgent, RandomAgent
)

app = FastAPI(title="CueZero - AlphaZero Billiards AI")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置静态文件目录
import os
web_dir = os.path.join(os.path.dirname(__file__), '..', 'web')
app.mount("/static", StaticFiles(directory=web_dir), name="static")


class AgentType(str, Enum):
    """Supported agent types"""
    HUMAN = "human"
    MCTS_FAST = "mcts_fast"
    MCTS_FULL = "mcts_full"
    POLICY = "policy"
    BASIC = "basic"
    RANDOM = "random"


# ============== PoolEnv 适配层 ==============
class PoolEnvAdapter:
    """适配PoolEnv到现有BilliardsEnv接口"""
    def __init__(self):
        # 导入PoolEnv，关闭调试输出
        import sys
        import os
        import types

        # Mock agents模块，避免导入失败
        mock_agents = types.ModuleType('agents')
        mock_agents.Agent = object
        mock_agents.BasicAgent = object
        mock_agents.BasicAgentPro = object
        mock_agents.NewAgent = object
        sys.modules['agents'] = mock_agents

        # 添加路径并导入PoolEnv
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '大作业提交版'))
        from poolenv import PoolEnv
        self.env = PoolEnv(debug=False)
        self.done = False
        self.winner = None
        self.balls = {}
        self.table = None

    def reset(self, target_ball='solid'):
        """兼容原有reset接口"""
        self.env.reset(target_ball=target_ball)
        self.done = False
        self.winner = None
        self.balls = self.env.balls
        self.table = self.env.table
        return self.get_observation(self.env.get_curr_player())

    def step(self, action):
        """兼容原有step接口，返回(obs, reward, done, info)格式"""
        # PoolEnv使用take_shot执行击球
        step_info = self.env.take_shot(action)
        self.balls = self.env.balls

        # 检查游戏是否结束
        done, info = self.env.get_done()
        self.done = done
        if done:
            self.winner = info.get('winner', 'DRAW')
            # 转换PoolEnv的SAME为DRAW
            if self.winner == 'SAME':
                self.winner = 'DRAW'

        # 获取新的观测
        obs = self.get_observation(self.env.get_curr_player())
        return obs, 0, done, step_info

    def get_observation(self, player):
        """兼容原有get_observation接口"""
        return self.env.get_observation(player)

    @property
    def hit_count(self):
        return self.env.hit_count

def convert_frontend_action_to_poolenv(action: dict) -> dict:
    """
    将前端发送的动作格式转换为 PoolEnv 需要的格式

    前端格式: {phi: 弧度(0-2π), V0: 0-100}
    PoolEnv格式: {V0: 0.5-8.0 m/s, phi: 0-360度, theta: 0, a: 0, b: 0}
    """
    # 转换 V0: 0-100 -> 0.5-8.0 m/s
    frontend_v0 = action.get('V0', 50)
    poolenv_v0 = 0.5 + (frontend_v0 / 100.0) * 7.5  # 0.5 到 8.0

    # 转换 phi: 弧度 -> 角度 (0-360度)
    frontend_phi = action.get('phi', 0)
    poolenv_phi = math.degrees(frontend_phi) % 360  # 确保在 0-360 范围内

    return {
        'V0': poolenv_v0,
        'phi': poolenv_phi,
        'theta': 0.0,
        'a': 0.0,
        'b': 0.0
    }


def clean_step_info(step_info: dict) -> dict:
    """
    清理 step_info，移除不可序列化的对象（如 Ball 对象）
    只保留可 JSON 序列化的简单类型
    """
    if not step_info:
        return {}

    cleaned = {}
    # 只保留这些键，它们都是简单类型
    allowed_keys = [
        'ME_INTO_POCKET', 'ENEMY_INTO_POCKET',
        'WHITE_BALL_INTO_POCKET', 'BLACK_BALL_INTO_POCKET',
        'FOUL_FIRST_HIT', 'NO_POCKET_NO_RAIL', 'NO_HIT',
        'winner'
    ]

    for key in allowed_keys:
        if key in step_info:
            cleaned[key] = step_info[key]

    return cleaned


def convert_pooltool_coords(pos_raw: list[float], table_l: float, table_w: float,
                             target_left: float, target_right: float,
                             target_bottom: float, target_top: float) -> list[float]:
    """
    将 pooltool 原始坐标转换为前端兼容坐标

    硬编码转换逻辑（基于调试数据分析）：
    - pooltool的x: 0.38-0.61 (宽度方向) -> 我们的y轴: -3.5到3.5
    - pooltool的y: 0.45-1.68 (长度方向) -> 我们的x轴: -7到7

    后续如需调整或移除硬编码，只需修改此函数。

    Args:
        pos_raw: pooltool原始坐标 [x, y]
        table_l: pooltool球桌长度 (y轴方向)
        table_w: pooltool球桌宽度 (x轴方向)
        target_left/target_right/target_bottom/target_top: 目标坐标范围

    Returns:
        转换后的坐标 [x, y]
    """
    target_width = target_right - target_left
    target_height = target_top - target_bottom

    # 映射关系：
    # pooltool的y -> 我们的x
    # pooltool的x -> 我们的y
    # pooltool原点在左下角，我们原点在中心
    pos = [
        (pos_raw[1] - table_l / 2.0) * (target_width / table_l),
        (pos_raw[0] - table_w / 2.0) * (target_height / table_w)
    ]

    # 边界钳制：确保球在球桌范围内（保留0.2单位安全距离）
    pos[0] = max(target_left + 0.2, min(target_right - 0.2, pos[0]))
    pos[1] = max(target_bottom + 0.2, min(target_top - 0.2, pos[1]))

    return pos


# ============== Battle State Management ==============

class BattleState:
    """Represents a battle session between two agents"""

    def __init__(self, battle_id: str, agent_a: Agent, agent_b: Agent,
                 agent_a_type: str, agent_b_type: str):
        self.battle_id = battle_id
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.agent_a_type = agent_a_type
        self.agent_b_type = agent_b_type
        self.env = PoolEnvAdapter()  # 使用完整PoolEnv环境
        self.current_game = 1
        self.total_games = 1
        self.results = {'A': 0, 'B': 0, 'DRAW': 0}
        self.current_player = 'A'
        self.game_status = "waiting"  # waiting, playing, finished
        self.winner = None
        self.last_action = None
        self.last_shot_info = None  # 记录最后一次击球的详细信息
        self.shot_history = []  # 历史击球记录
        self.created_at = time.time()
        self.target_ball = 'solid'  # 当前局的目标球型

    def start_game(self, target_ball: str = 'solid'):
        """Start a new game"""
        self.target_ball = target_ball
        self.env.reset(target_ball=target_ball)
        self.game_status = "playing"
        self.current_player = self.env.env.get_curr_player()  # 从PoolEnv获取初始玩家
        self.last_action = None

        # Reset agent buffers
        if hasattr(self.agent_a, 'clear_buffer'):
            self.agent_a.clear_buffer()
        if hasattr(self.agent_b, 'clear_buffer'):
            self.agent_b.clear_buffer()

    def get_current_agent(self) -> Agent:
        """Get the current player's agent"""
        return self.agent_a if self.current_player == 'A' else self.agent_b

    def execute_action(self, action: dict) -> dict:
        """Execute an action and return step info"""
        # 记录击球前的状态
        start_state = self.get_ball_state()

        # 转换动作格式：前端格式 -> PoolEnv 格式
        # 检查是否是前端格式（只有 phi 和 V0）
        if set(action.keys()) == {'phi', 'V0'}:
            poolenv_action = convert_frontend_action_to_poolenv(action)
        else:
            # 已经是完整格式，直接使用
            poolenv_action = action

        step_info = self.env.step(poolenv_action)[-1]
        self.last_action = poolenv_action

        # 清理 step_info，移除不可序列化的对象
        cleaned_step_info = clean_step_info(step_info)

        # 记录击球后的状态和信息
        end_state = self.get_ball_state()
        self.last_shot_info = {
            'player': self.current_player,
            'action': action,
            'step_info': cleaned_step_info,
            'start_state': start_state,
            'end_state': end_state,
            'timestamp': time.time()
        }
        self.shot_history.append(self.last_shot_info)

        # Check if game is done
        done = self.env.done if hasattr(self.env, 'done') else False

        # 额外检查：是否达到最大回合数（60）
        hit_count = self.env.hit_count if hasattr(self.env, 'hit_count') else len(self.shot_history)
        if not done and hit_count >= 60:
            # 达到最大回合数，手动结束游戏
            done = True
            # 计算剩余球数决定胜负
            red_remaining = 0
            yellow_remaining = 0
            for ball_id, ball in self.env.balls.items():
                if hasattr(ball, 'state') and hasattr(ball.state, 's'):
                    if ball.state.s == 4:
                        continue  # 已进袋
                elif getattr(ball, 'pocketed', False):
                    continue
                # 统计未进袋的球
                if str(ball_id).isdigit():
                    num = int(ball_id)
                    if 1 <= num < 8:
                        red_remaining += 1
                    elif 9 <= num <= 15:
                        yellow_remaining += 1
            # 决定胜者
            if red_remaining < yellow_remaining:
                self.env.winner = 'A'
            elif yellow_remaining < red_remaining:
                self.env.winner = 'B'
            else:
                self.env.winner = 'DRAW'
            self.env.done = True

        # 重新检查 done 状态（可能在上面被修改了）
        done = self.env.done if hasattr(self.env, 'done') else done

        if done:
            self.game_status = "finished"
            winner = self.env.winner if hasattr(self.env, 'winner') else 'DRAW'
            self.results[winner] += 1
            self.winner = winner
            # 记录本局结果到历史
            self.shot_history.append({
                'type': 'game_end',
                'winner': winner,
                'results': self.results.copy(),
                'timestamp': time.time()
            })

        # 同步PoolEnv的当前玩家
        self.current_player = self.env.env.get_curr_player()

        return step_info

    def get_state(self) -> dict:
        """Get current battle state as dictionary"""
        return {
            'battle_id': self.battle_id,
            'agent_a': self.agent_a.name,
            'agent_b': self.agent_b.name,
            'agent_a_type': self.agent_a_type,
            'agent_b_type': self.agent_b_type,
            'current_game': self.current_game,
            'total_games': self.total_games,
            'results': self.results,
            'current_player': self.current_player,
            'game_status': self.game_status,
            'winner': self.winner,
            'last_action': self.last_action,
            'last_shot_info': self.last_shot_info,
            'shot_history': self.shot_history,
            'target_ball': self.target_ball
        }

    def get_ball_state(self) -> dict:
        """Get simplified ball state compatible with frontend render"""
        balls = {}

        # 获取pooltool球桌的真实尺寸
        # pooltool中：table.l是长度(x轴方向)，table.w是宽度(y轴方向)
        table_l = 1.9812  # 默认值：约78英寸 (2.0193米，但实际pooltool可能是1.9812米)
        table_w = 0.9906   # 默认值：约39英寸 (1.00965米，但实际pooltool可能是0.9906米)

        if hasattr(self.env.table, 'l'):
            table_l = self.env.table.l
        if hasattr(self.env.table, 'w'):
            table_w = self.env.table.w

        # 目标坐标范围（与mock接口一致）
        target_left = -7.0
        target_right = 7.0
        target_bottom = -3.5
        target_top = 3.5
        target_width = target_right - target_left
        target_height = target_top - target_bottom

        # 计算缩放比例
        scale_x = target_width / table_l
        scale_y = target_height / table_w

        for ball_id, ball in self.env.balls.items():
            # 获取球位置 - 优先从pooltool的rvw中获取
            pos_raw = [0.0, 0.0]
            if hasattr(ball, 'state') and hasattr(ball.state, 'rvw'):
                # pooltool格式：rvw[0]是位置[x, y, z]，单位米
                pos_raw = [ball.state.rvw[0][0], ball.state.rvw[0][1]]
            elif hasattr(ball, 'x') and hasattr(ball, 'y'):
                pos_raw = [ball.x, ball.y]
            elif hasattr(ball, 'pos'):
                pos_raw = [ball.pos[0], ball.pos[1]]
            elif hasattr(ball, 'state'):
                state = ball.state
                if hasattr(state, 'x') and hasattr(state, 'y'):
                    pos_raw = [state.x, state.y]
                elif hasattr(state, 'pos'):
                    pos_raw = [state.pos[0], state.pos[1]]
                elif hasattr(state, 'rv'):
                    pos_raw = [state.rv[0], state.rv[1]]
                elif hasattr(state, '__getitem__'):
                    pos_raw = [state[0], state[1]]

            # 使用单独的转换函数进行坐标转换
            pos = convert_pooltool_coords(
                pos_raw, table_l, table_w,
                target_left, target_right, target_bottom, target_top
            )

            # 检测是否已经进袋
            # pooltool 中：ball.state.s == 4 表示进袋
            pocketed = False
            if hasattr(ball, 'state') and hasattr(ball.state, 's'):
                pocketed = (ball.state.s == 4)
            else:
                pocketed = getattr(ball, 'pocketed', False)

            # 获取队伍信息
            team = None
            if hasattr(ball, 'team'):
                team = ball.team
            elif str(ball_id) == '8' or ball_id == 8:
                team = 'neutral'
            elif str(ball_id).isdigit():
                num = int(ball_id)
                if num == 0 or ball_id == 'cue':
                    team = None  # cue ball
                elif 1 <= num <=7:
                    team = 'red'
                elif 9 <= num <=15:
                    team = 'yellow'
                elif num ==8:
                    team = 'neutral'

            balls[ball_id] = {
                'pos': pos,
                'pocketed': pocketed,
                'team': team
            }

        # 使用目标范围作为桌台信息
        table_info = {
            'left': target_left,
            'right': target_right,
            'bottom': target_bottom,
            'top': target_top
        }

        # 计算得分
        red_score = 0
        yellow_score = 0
        if hasattr(self.env, 'pocketed_balls'):
            for b in self.env.pocketed_balls:
                num = int(getattr(b, 'number', 0))
                if 1 <= num < 8:
                    red_score += 1
                elif 9 <= num <=15:
                    yellow_score +=1

        state = {
            'balls': balls,
            'table': table_info,
            'turn': len(self.shot_history) + 1,
            'player': self.current_player,
            'red_score': red_score,
            'yellow_score': yellow_score
        }
        return state


# Battle storage
_battles: Dict[str, BattleState] = {}


# ============== AI Components ==============

_ai_components = None
_use_ai = False


def get_ai_components():
    """Lazy load AI components"""
    global _ai_components
    if _ai_components is None:
        print("Loading AI components...")
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'dual_network_final.pt')
            model = DualNetwork()
            model.load(model_path)

            _ai_components = {'model': model}
            print("AI components loaded successfully")
        except Exception as e:
            print(f"Failed to load AI components: {e}")
            _ai_components = {}
    return _ai_components


def create_agent(agent_type: str, env: BilliardsEnv) -> Agent:
    """Create an agent based on type string"""
    agent_type = agent_type.lower()

    if agent_type == 'human':
        return HumanAgent(name="Human")
    elif agent_type in ('mcts_fast', 'mcts_full'):
        components = get_ai_components()
        if not components or 'model' not in components:
            raise RuntimeError("AI model not loaded")
        mode = "fast" if agent_type == 'mcts_fast' else "full"
        mcts = MCTS(model=components['model'], mode=mode)
        agent = MCTSAgent(mcts=mcts, name=f"MCTS({mode.capitalize()})")
        agent.set_env(env)
        return agent
    elif agent_type == 'policy':
        components = get_ai_components()
        if not components or 'model' not in components:
            raise RuntimeError("AI model not loaded")
        return PolicyAgent(policy_network=components['model'], name="Policy")
    elif agent_type == 'basic':
        return BasicAgent(name="Basic")
    elif agent_type == 'random':
        return RandomAgent(name="Random")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ============== API Models ==============

class ShotAction(BaseModel):
    """击球动作"""
    phi: float
    V0: float


class ShotRequest(BaseModel):
    """击球请求"""
    action: ShotAction
    state: dict | None = None


class ShotResponse(BaseModel):
    """击球响应"""
    success: bool
    start_state: dict
    end_state: dict
    message: str


class BattleStartRequest(BaseModel):
    """Battle start request"""
    agent_a_type: str = "mcts_fast"
    agent_b_type: str = "basic"
    total_games: int = 5
    first_game_target: str = "solid"


class BattleStartResponse(BaseModel):
    """Battle start response"""
    battle_id: str
    status: str
    agent_a: str
    agent_b: str
    message: str


class BattleNextRequest(BaseModel):
    """Battle next step request"""
    action: dict | None = None  # If None, agent will decide


class BattleNextResponse(BaseModel):
    """Battle next step response"""
    battle_id: str
    step_info: dict | None
    action: dict | None = None  # 执行的击球动作
    game_status: str
    current_player: str
    winner: str | None
    message: str
    start_state: dict | None = None  # 击球前状态，用于前端动画
    end_state: dict | None = None  # 击球后状态，用于前端渲染
    target_ball: str | None = None  # 当前局的目标球型 (solid/stripe)
    current_game: int | None = None  # 当前局数


class BattleStatusResponse(BaseModel):
    """Battle status response"""
    battle_id: str
    state: dict


# ============== API Endpoints ==============

@app.get("/")
async def root() -> HTMLResponse:
    """返回主页面"""
    try:
        web_dir = os.path.join(os.path.dirname(__file__), '..', 'web')
        index_path = os.path.join(web_dir, 'index.html')
        return FileResponse(index_path)
    except FileNotFoundError:
        return HTMLResponse("<h1>CueZero</h1><p>前端文件未找到</p>")


@app.get("/api/state")
async def get_state() -> dict:
    """获取当前游戏状态"""
    return get_initial_state()


@app.post("/api/shot", response_model=ShotResponse)
async def execute_shot(request: ShotRequest) -> ShotResponse:
    """执行击球"""
    global game_state

    current_state = request.state if request.state else get_initial_state()
    start_state = copy.deepcopy(current_state)

    try:
        end_state = simulate_shot(request.action.model_dump(), current_state)
        pocketed = set(start_state['balls'].keys()) - set(end_state['balls'].keys())

        message = "击球完成"
        if pocketed:
            pocketed_list = [b for b in pocketed if b != 'cue']
            if pocketed_list:
                message = f"进球：{', '.join(pocketed_list)}"

        return ShotResponse(
            success=True,
            start_state=start_state,
            end_state=end_state,
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
async def reset_game() -> dict:
    """重置游戏"""
    return {"success": True, "state": get_initial_state()}


@app.post("/api/battle/start", response_model=BattleStartResponse)
async def start_battle(request: BattleStartRequest) -> BattleStartResponse:
    """Start a new battle between two agents"""
    try:
        # Validate agent types
        valid_types = ['human', 'mcts_fast', 'mcts_full', 'policy', 'basic', 'random']
        if request.agent_a_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid agent_a_type: {request.agent_a_type}")
        if request.agent_b_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid agent_b_type: {request.agent_b_type}")

        # Create battle ID
        battle_id = str(uuid.uuid4())[:8]

        # Create environment (needed for agent creation)
        env = BilliardsEnv()

        # Create agents
        agent_a = create_agent(request.agent_a_type, env)
        agent_b = create_agent(request.agent_b_type, env)

        # Create battle state
        battle = BattleState(
            battle_id=battle_id,
            agent_a=agent_a,
            agent_b=agent_b,
            agent_a_type=request.agent_a_type,
            agent_b_type=request.agent_b_type
        )
        battle.total_games = request.total_games
        battle.start_game(target_ball=request.first_game_target)

        # Store battle
        _battles[battle_id] = battle

        return BattleStartResponse(
            battle_id=battle_id,
            status="playing",
            agent_a=agent_a.name,
            agent_b=agent_b.name,
            message=f"Battle started: {agent_a.name} vs {agent_b.name}"
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/battle/{battle_id}/next", response_model=BattleNextResponse)
async def battle_next(battle_id: str, request: BattleNextRequest) -> BattleNextResponse:
    """Execute next step in battle"""
    if battle_id not in _battles:
        raise HTTPException(status_code=404, detail="Battle not found")

    battle = _battles[battle_id]

    if battle.game_status != "playing":
        return BattleNextResponse(
            battle_id=battle_id,
            step_info=None,
            game_status=battle.game_status,
            current_player=battle.current_player,
            winner=battle.winner,
            message="Game not in playing state"
        )

    # Get current agent
    agent = battle.get_current_agent()

    # Get action
    if request.action is not None:
        action = request.action
    else:
        # Agent decides
        obs = battle.env.get_observation(battle.current_player)
        balls, my_targets, table = obs
        try:
            action = agent.decision(balls, my_targets, table)
        except Exception as e:
            return BattleNextResponse(
                battle_id=battle_id,
                step_info=None,
                game_status=battle.game_status,
                current_player=battle.current_player,
                winner=None,
                message=f"Agent decision error: {e}"
            )

    # Execute action
    step_info = battle.execute_action(action)

    # 清理 step_info，移除不可序列化的对象
    cleaned_step_info = clean_step_info(step_info)

    # Build response message
    start_state = battle.last_shot_info['start_state'] if battle.last_shot_info else None
    end_state = battle.last_shot_info['end_state'] if battle.last_shot_info else None

    if battle.game_status == "finished":
        if battle.winner == 'DRAW':
            msg = "Game ended in a draw"
        else:
            winner_name = battle.agent_a.name if battle.winner == 'A' else battle.agent_b.name
            msg = f"{winner_name} wins!"

        # 如果还有下一局，自动开始下一局
        if battle.current_game < battle.total_games:
            battle.current_game += 1
            # 轮换目标球型：solid, solid, stripe, stripe
            target_index = (battle.current_game - 1) % 4
            target_ball = 'solid' if target_index < 2 else 'stripe'
            battle.start_game(target_ball=target_ball)
            msg += f" 第{battle.current_game}局即将开始"
    else:
        msg = f"Action executed by {agent.name}, next player: {battle.current_player}"

    return BattleNextResponse(
        battle_id=battle_id,
        step_info=cleaned_step_info,
        action=action,
        game_status=battle.game_status,
        current_player=battle.current_player,
        winner=battle.winner,
        message=msg,
        start_state=start_state if start_state else None,
        end_state=end_state if end_state else None,
        target_ball=battle.target_ball,
        current_game=battle.current_game
    )


@app.get("/api/battle/{battle_id}/status", response_model=BattleStatusResponse)
async def get_battle_status(battle_id: str) -> BattleStatusResponse:
    """Get battle status"""
    if battle_id not in _battles:
        raise HTTPException(status_code=404, detail="Battle not found")

    battle = _battles[battle_id]
    return BattleStatusResponse(
        battle_id=battle_id,
        state=battle.get_state()
    )


@app.get("/api/battle/{battle_id}/ball_state")
async def get_battle_ball_state(battle_id: str) -> dict:
    """Get simplified ball state for frontend rendering"""
    if battle_id not in _battles:
        raise HTTPException(status_code=404, detail="Battle not found")

    battle = _battles[battle_id]
    return battle.get_ball_state()


@app.get("/api/battles")
async def list_battles() -> list:
    """List all active battles"""
    return [
        {
            'battle_id': b.battle_id,
            'agent_a': b.agent_a.name,
            'agent_b': b.agent_b.name,
            'status': b.game_status,
            'current_game': b.current_game,
            'results': b.results,
        }
        for b in _battles.values()
    ]


@app.delete("/api/battle/{battle_id}")
async def delete_battle(battle_id: str) -> dict:
    """Delete a battle"""
    if battle_id not in _battles:
        raise HTTPException(status_code=404, detail="Battle not found")

    del _battles[battle_id]
    return {"success": True, "message": f"Battle {battle_id} deleted"}


@app.get("/api/ai_shot")
async def ai_shot() -> dict:
    """获取 AI 推荐的击球动作"""
    # Simplified AI shot for mock environment
    return {"success": False, "message": "Use /api/battle/* endpoints for AI battles"}


@app.get("/test")
async def test() -> HTMLResponse:
    """返回测试页面"""
    try:
        web_dir = os.path.join(os.path.dirname(__file__), '..', 'web')
        test_path = os.path.join(web_dir, 'test.html')
        return FileResponse(test_path)
    except FileNotFoundError:
        return HTMLResponse("<h1>Test page not found</h1>")


@app.get("/api/health")
async def health_check() -> dict:
    """健康检查"""
    ai_loaded = get_ai_components() is not None and _ai_components is not None
    return {
        "status": "ok",
        "mode": "ai" if _use_ai else "mock",
        "ai_loaded": ai_loaded,
        "active_battles": len(_battles)
    }


# ============== Main ==============

def main():
    """主函数"""
    global _use_ai

    parser = argparse.ArgumentParser(description="CueZero Web Server")
    parser.add_argument(
        "--ai",
        action="store_true",
        default=False,
        help="使用 AI 模式（默认 mock 模式）"
    )
    parser.add_argument(
        "--agent-a",
        type=str,
        default="mcts_fast",
        choices=['human', 'mcts_fast', 'mcts_full', 'policy', 'basic', 'random'],
        help="Default Agent A type for battles (default: mcts_fast)"
    )
    parser.add_argument(
        "--agent-b",
        type=str,
        default="basic",
        choices=['human', 'mcts_fast', 'mcts_full', 'policy', 'basic', 'random'],
        help="Default Agent B type for battles (default: basic)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="绑定主机（默认：0.0.0.0）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="绑定端口（默认：8000）"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载（开发模式）"
    )

    args = parser.parse_args()
    _use_ai = args.ai

    print(f"🎱 CueZero Server starting...")
    print(f"   Mode: {'AI' if _use_ai else 'mock'}")
    print(f"   Default Battle: {args.agent_a} vs {args.agent_b}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   URL:  http://localhost:{args.port}")
    print()
    print("   Battle API Endpoints:")
    print("   - POST /api/battle/start - Start new battle")
    print("   - POST /api/battle/{id}/next - Execute next step")
    print("   - GET  /api/battle/{id}/status - Get battle status")
    print("   - GET  /api/battles - List all battles")
    print()

    if _use_ai:
        print("   Loading AI components...")
        get_ai_components()

    import uvicorn
    # 直接传入 app 对象，避免模块路径问题
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
