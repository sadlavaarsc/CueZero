"""
AlphaZero 台球 AI - Web 服务端

支持两种模式：
- mock 模式：使用简单规则模拟击球结果
- 真实模式：连接到 AlphaZero AI 后端
"""

import argparse
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 导入 mock 环境
from .mock_env import get_initial_state, simulate_shot

# 导入真实 AI 模块
from cuezero.models.dual_network import DualNetwork
from cuezero.mcts.search import MCTS

app = FastAPI(title="CueZero - AlphaZero Billiards AI")

# 配置 CORS（允许本地开发）
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

# 全局状态
game_state = None

# AI 组件（懒加载）
_ai_components = None
_use_ai = False  # 默认使用 mock 模式

def get_ai_components():
    """Lazy load AI components"""
    global _ai_components
    if _ai_components is None:
        print("Loading AI components...")
        try:
            # 加载模型
            model_path = os.path.join(os.path.dirname(__file__), '..', 'dual_network_final.pt')
            model = DualNetwork()
            model.load(model_path)

            # 创建 MCTS（默认快速模式）
            mcts = MCTS(model=model, mode="fast")

            _ai_components = {'model': model, 'mcts': mcts}
            print("AI components loaded successfully")
        except Exception as e:
            print(f"Failed to load AI components: {e}")
            _ai_components = {}
    return _ai_components


def ai_get_shot(balls, my_targets, table):
    """Get shot from AI using MCTS.

    Args:
        balls: Ball state dictionary (pooltool Ball objects)
        my_targets: Target ball list
        table: Table object

    Returns:
        dict: {'phi': angle, 'V0': power} or None if AI not available
    """
    components = get_ai_components()
    if not components or not components.get('mcts'):
        return None

    mcts = components['mcts']

    try:
        # 生成候选动作并选择最佳的
        actions = mcts.generate_heuristic_actions(balls, my_targets, table)
        if actions:
            # 简单版本：返回第一个动作
            # 完整版本可以运行一次 MCTS 搜索来选择最佳动作
            best_action = actions[0]
            return {'phi': best_action['phi'], 'V0': best_action['V0']}
    except Exception as e:
        print(f"AI decision error: {e}")

    return None


class ShotAction(BaseModel):
    """击球动作"""
    phi: float  # 击球角度（弧度）
    V0: float   # 击球力度


class ShotRequest(BaseModel):
    """击球请求"""
    action: ShotAction
    state: dict | None = None


class ShotResponse(BaseModel):
    """击球响应"""
    success: bool
    start_state: dict  # 击球前状态
    end_state: dict    # 击球后状态
    message: str


@app.get("/")
async def root() -> HTMLResponse:
    """返回主页面"""
    try:
        # 使用绝对路径（web 目录在 server 的上一级）
        import os
        web_dir = os.path.join(os.path.dirname(__file__), '..', 'web')
        index_path = os.path.join(web_dir, 'index.html')
        return FileResponse(index_path)
    except FileNotFoundError:
        return HTMLResponse("<h1>CueZero</h1><p>前端文件未找到，请确保 web/index.html 存在</p>")


@app.get("/api/state")
async def get_state() -> dict:
    """获取当前游戏状态"""
    global game_state
    if game_state is None:
        game_state = get_initial_state()
    return game_state


@app.post("/api/shot", response_model=ShotResponse)
async def execute_shot(request: ShotRequest) -> ShotResponse:
    """
    执行击球

    接收击球动作和当前状态，返回击球前后的状态快照
    """
    global game_state

    # 使用请求中的状态或当前状态
    if request.state:
        current_state = request.state
    else:
        current_state = game_state

    if current_state is None:
        current_state = get_initial_state()

    # 记录击球前状态（深拷贝）
    import copy
    start_state = copy.deepcopy(current_state)

    # 执行击球模拟
    try:
        end_state = simulate_shot(request.action.model_dump(), current_state)
        game_state = end_state

        # 检查是否有球进袋
        start_balls = set(start_state['balls'].keys())
        end_balls = set(end_state['balls'].keys())
        pocketed = start_balls - end_balls

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
    global game_state
    game_state = get_initial_state()
    return {"success": True, "state": game_state}


@app.post("/api/ai_shot")
async def ai_shot() -> dict:
    """获取 AI 推荐的击球动作"""
    global game_state

    if game_state is None:
        game_state = get_initial_state()

    # 将 web UI 的状态格式转换为 AI 需要的格式
    # 注意：这里需要适配 mock_env 和真实物理环境的差异
    # 简化版本：直接使用 heuristic 生成动作

    # 转换球状态（mock 格式 -> pooltool 格式）
    from cuezero.env.physics_wrapper import PhysicsWrapper
    import pooltool as pt

    # 创建临时的 pooltool 环境
    table = pt.Table.create(pt.TableType.SNOOKER)
    balls = {}

    # 转换白球
    cue_data = game_state['balls'].get('cue')
    if cue_data:
        cue_ball = pt.Ball.create('cue', pos=(cue_data['pos'][0], cue_data['pos'][1]))
        # 设置状态为静止
        cue_ball.state.s = 1
        balls['cue'] = cue_ball

    # 转换其他球
    for ball_id, ball_data in game_state['balls'].items():
        if ball_id == 'cue':
            continue
        if ball_data.get('pocketed', False):
            continue
        try:
            ball = pt.Ball.create(ball_id, pos=(ball_data['pos'][0], ball_data['pos'][1]))
            ball.state.s = 1
            balls[ball_id] = ball
        except Exception:
            pass

    # 确定目标球（简化：假设用户是 solid 方）
    my_targets = ['1', '2', '3', '4', '5', '6', '7']

    # 获取 AI 建议
    ai_action = ai_get_shot(balls, my_targets, table)

    if ai_action:
        return {"success": True, "action": ai_action}
    else:
        return {"success": False, "message": "AI not available"}


@app.get("/test")
async def test() -> HTMLResponse:
    """返回测试页面"""
    try:
        import os
        web_dir = os.path.join(os.path.dirname(__file__), '..', 'web')
        test_path = os.path.join(web_dir, 'test.html')
        return FileResponse(test_path)
    except FileNotFoundError:
        return HTMLResponse("<h1>Test page not found</h1>")


@app.get("/api/health")
async def health_check() -> dict:
    """健康检查"""
    mode = "ai" if _use_ai else "mock"
    ai_loaded = get_ai_components() is not None and _ai_components is not None
    return {"status": "ok", "mode": mode, "ai_loaded": ai_loaded}


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
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   URL:  http://localhost:{args.port}")

    if _use_ai:
        print("   Loading AI components...")
        get_ai_components()

    print()

    import uvicorn
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
