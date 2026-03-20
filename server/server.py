"""
AlphaZero 台球 AI - Web 服务端

支持两种模式：
- mock 模式：使用简单规则模拟击球结果
- 真实模式：连接到 AlphaZero AI 后端（待实现）
"""

import argparse
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 导入 mock 环境
from mock_env import get_initial_state, simulate_shot

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
    return {"status": "ok", "mode": "mock"}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CueZero Web Server")
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="使用 mock 模式（默认）"
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

    print(f"🎱 CueZero Server starting...")
    print(f"   Mode: {'mock' if args.mock else 'real'}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   URL:  http://localhost:{args.port}")
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
