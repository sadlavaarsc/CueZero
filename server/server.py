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
from typing import Dict, Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 导入 mock 环境
from .mock_env import get_initial_state, simulate_shot

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
        self.env = BilliardsEnv()
        self.current_game = 1
        self.total_games = 1
        self.results = {'A': 0, 'B': 0, 'DRAW': 0}
        self.current_player = 'A'
        self.game_status = "waiting"  # waiting, playing, finished
        self.winner = None
        self.last_action = None
        self.created_at = time.time()

    def start_game(self, target_ball: str = 'solid'):
        """Start a new game"""
        self.env.reset(target_ball=target_ball)
        self.game_status = "playing"
        self.current_player = 'A'
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
        step_info = self.env.take_shot(action)
        self.last_action = action

        # Check if game is done
        done, info = self.env.get_done()
        if done:
            self.game_status = "finished"
            winner = info.get('winner', 'DRAW')
            self.results[winner] += 1
            self.winner = winner
        else:
            # Switch player if no ball pocketed
            if not step_info.get('LEGAL_INTO_POCKET'):
                self.current_player = 'B' if self.current_player == 'A' else 'A'

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
        }


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
    game_status: str
    current_player: str
    winner: str | None
    message: str


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

    # Build response message
    if battle.game_status == "finished":
        if battle.winner == 'DRAW':
            msg = "Game ended in a draw"
        else:
            winner_name = battle.agent_a.name if battle.winner == 'A' else battle.agent_b.name
            msg = f"{winner_name} wins!"
    else:
        msg = f"Action executed by {agent.name}, next player: {battle.current_player}"

    return BattleNextResponse(
        battle_id=battle_id,
        step_info=step_info,
        game_status=battle.game_status,
        current_player=battle.current_player,
        winner=battle.winner,
        message=msg
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
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
