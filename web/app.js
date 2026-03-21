/**
 * CueZero 前端逻辑 - 极简扁平化风格
 */

// ==================== 配置 ====================
const CONFIG = {
    API_BASE: window.location.origin + '/api',
    TABLE_PADDING: 30,
    BALL_RADIUS: 10,
    CUE_MAX_LENGTH: 120,
    CUE_MIN_LENGTH: 25,
    ANIMATION_DURATION: 1500,
    COLORS: {
        cueBall: '#ffffff',
        ballRed: '#e74c3c',
        ballYellow: '#f1c40f',
        ballBlack: '#1a1a1a',
        cue: '#8b7355',
        aimLine: 'rgba(255, 255, 255, 0.5)',
        pocket: '#0a0a0a',
        table: '#2d5a3d',
        tableBorder: '#3d2817'
    }
};

// ==================== 状态 ====================
let gameState = null;
let currentState = null;
let isAnimating = false;
let isAiming = false;
let isDragging = false;
let aimStart = null;
let aimCurrent = null;
let power = 0;

// 对战模式状态
let currentBattleId = null;
let battleConfig = null;
let isBattleMode = false;
let matchHistory = [];

const canvas = document.getElementById('table');
const ctx = canvas.getContext('2d');

// ==================== 工具函数 ====================
function lerp(start, end, t) {
    return start + (end - start) * t;
}

function lerpArray(start, end, t) {
    return start.map((v, i) => lerp(start[i], end[i], t));
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

function distance(p1, p2) {
    return Math.sqrt(Math.pow(p2[0] - p1[0], 2) + Math.pow(p2[1] - p1[1], 2));
}

// 坐标转换：游戏坐标 -> 画布坐标
function toCanvas(pos) {
    if (!gameState) return [0, 0];
    const table = gameState.table;
    const tableWidth = table.right - table.left;
    const tableHeight = table.top - table.bottom;
    const scaleX = (canvas.width - 2 * CONFIG.TABLE_PADDING) / tableWidth;
    const scaleY = (canvas.height - 2 * CONFIG.TABLE_PADDING) / tableHeight;
    const scale = Math.min(scaleX, scaleY);
    const x = CONFIG.TABLE_PADDING + (pos[0] - table.left) * scale;
    const y = canvas.height - CONFIG.TABLE_PADDING - (pos[1] - table.bottom) * scale;
    return [x, y];
}

// 坐标转换：画布坐标 -> 游戏坐标
function toGame(x, y) {
    if (!gameState) return [0, 0];
    const table = gameState.table;
    const tableWidth = table.right - table.left;
    const tableHeight = table.top - table.bottom;
    const scaleX = (canvas.width - 2 * CONFIG.TABLE_PADDING) / tableWidth;
    const scaleY = (canvas.height - 2 * CONFIG.TABLE_PADDING) / tableHeight;
    const scale = Math.min(scaleX, scaleY);
    const gameX = table.left + (x - CONFIG.TABLE_PADDING) / scale;
    const gameY = table.bottom + (canvas.height - CONFIG.TABLE_PADDING - y) / scale;
    return [gameX, gameY];
}

// ==================== 渲染 ====================
function render() {
    if (!gameState) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawTable();
    drawPockets();
    drawBalls();

    if (isAiming || isDragging) {
        drawAim();
    }
}

function drawTable() {
    const table = gameState.table;
    const [leftX, topY] = toCanvas([table.left, table.top]);
    const [rightX, bottomY] = toCanvas([table.right, table.bottom]);
    const width = rightX - leftX;
    const height = bottomY - topY;

    // 台呢（纯色，无渐变）
    ctx.fillStyle = CONFIG.COLORS.table;
    ctx.fillRect(leftX, topY, width, height);

    // 桌边（纯色边框）
    ctx.strokeStyle = CONFIG.COLORS.tableBorder;
    ctx.lineWidth = 8;
    ctx.strokeRect(leftX, topY, width, height);
}

function drawPockets() {
    const pockets = [
        [gameState.table.left, gameState.table.top],
        [gameState.table.right, gameState.table.top],
        [gameState.table.left, 0],
        [gameState.table.right, 0],
        [gameState.table.left, gameState.table.bottom],
        [gameState.table.right, gameState.table.bottom]
    ];

    pockets.forEach(pos => {
        const [x, y] = toCanvas(pos);
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fillStyle = CONFIG.COLORS.pocket;
        ctx.fill();
    });
}

function drawBalls() {
    const stateToRender = isAnimating && currentState ? currentState : gameState;

    Object.entries(stateToRender.balls).forEach(([id, ball]) => {
        if (ball.pocketed) return;

        const [x, y] = toCanvas(ball.pos);

        // 确定颜色（纯色，无渐变）
        let color;
        if (id === 'cue') {
            color = CONFIG.COLORS.cueBall;
        } else if (id === '8') {
            color = CONFIG.COLORS.ballBlack;
        } else if (['1', '2', '3', '4', '5', '6', '7'].includes(id)) {
            color = CONFIG.COLORS.ballRed;
        } else if (['9', '10', '11', '12', '13', '14', '15'].includes(id)) {
            color = CONFIG.COLORS.ballYellow;
        } else {
            color = CONFIG.COLORS.ballYellow; // 默认黄色
        }

        // 绘制球（纯色圆，无阴影无高光）
        ctx.beginPath();
        ctx.arc(x, y, CONFIG.BALL_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // 黑八绘制白色数字区域
        if (id === '8') {
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
            ctx.fillStyle = '#000';
            ctx.font = 'bold 8px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('8', x, y);
        }
    });
}

function drawAim() {
    if (!aimStart) return;

    const [startX, startY] = toCanvas(aimStart);

    let aimEnd;
    if (isDragging && aimCurrent) {
        const [currX, currY] = toCanvas(aimCurrent);
        const dx = startX - currX;
        const dy = startY - currY;
        const len = Math.sqrt(dx * dx + dy * dy);

        if (len > 0) {
            aimEnd = [
                startX + (dx / len) * CONFIG.CUE_MAX_LENGTH,
                startY + (dy / len) * CONFIG.CUE_MAX_LENGTH
            ];
        } else {
            aimEnd = [startX + 80, startY];
        }
    } else {
        aimEnd = [startX + 80, startY];
    }

    // 绘制瞄准线
    ctx.strokeStyle = CONFIG.COLORS.aimLine;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(aimEnd[0], aimEnd[1]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 绘制球杆
    if (isDragging && aimCurrent) {
        const [currX, currY] = toCanvas(aimCurrent);
        const dx = currX - startX;
        const dy = currY - startY;
        const len = Math.sqrt(dx * dx + dy * dy);

        if (len > CONFIG.CUE_MIN_LENGTH) {
            const cueLen = Math.min(len - CONFIG.BALL_RADIUS, CONFIG.CUE_MAX_LENGTH);
            const angle = Math.atan2(dy, dx);
            const cueStartX = startX + Math.cos(angle) * CONFIG.BALL_RADIUS;
            const cueStartY = startY + Math.sin(angle) * CONFIG.BALL_RADIUS;
            const cueEndX = startX + Math.cos(angle) * (CONFIG.BALL_RADIUS + cueLen);
            const cueEndY = startY + Math.sin(angle) * (CONFIG.BALL_RADIUS + cueLen);

            ctx.strokeStyle = CONFIG.COLORS.cue;
            ctx.lineWidth = 6;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(cueStartX, cueStartY);
            ctx.lineTo(cueEndX, cueEndY);
            ctx.stroke();

            power = Math.min((len / CONFIG.CUE_MAX_LENGTH) * 100, 100);
            updateStatus(`力度：${Math.round(power)}%`);
        }
    }
}

// ==================== 动画 ====================
async function playAnimation(startState, endState) {
    isAnimating = true;
    document.getElementById('overlay').style.display = 'flex';

    const startTime = performance.now();

    return new Promise(resolve => {
        function animate(currentTime) {
            const progress = Math.min((currentTime - startTime) / CONFIG.ANIMATION_DURATION, 1);
            const eased = easeOutCubic(progress);
            currentState = interpolateState(startState, endState, eased);
            render();

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                isAnimating = false;
                currentState = null;
                document.getElementById('overlay').style.display = 'none';
                resolve();
            }
        }
        requestAnimationFrame(animate);
    });
}

function interpolateState(start, end, t) {
    const state = {
        balls: {},
        table: { ...end.table },
        turn: end.turn,
        player: end.player,
        score: end.score,
        red_score: end.red_score || 0,
        yellow_score: end.yellow_score || 0
    };

    for (const [id, endBall] of Object.entries(end.balls)) {
        const startBall = start.balls[id];
        state.balls[id] = startBall
            ? { ...endBall, pos: lerpArray(startBall.pos, endBall.pos, t) }
            : { ...endBall };
    }
    return state;
}

// ==================== API ====================
async function fetchState() {
    try {
        if (isBattleMode && currentBattleId) {
            // 对战模式下获取对战球状态
            const response = await fetch(`${CONFIG.API_BASE}/battle/${currentBattleId}/ball_state`);
            gameState = await response.json();
        } else {
            // 普通模式下获取mock状态
            const response = await fetch(`${CONFIG.API_BASE}/state`);
            gameState = await response.json();
        }
        updateUI();
        render();
        updateStatus('状态已更新');
    } catch (error) {
        updateStatus('获取状态失败：' + error.message);
    }
}

// 开始对战
async function startBattle() {
    try {
        const agentA = document.getElementById('agent-a').value;
        const agentB = document.getElementById('agent-b').value;
        const totalGames = parseInt(document.getElementById('total-games').value);

        updateStatus('正在创建对战...');

        const response = await fetch(`${CONFIG.API_BASE}/battle/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                agent_a_type: agentA,
                agent_b_type: agentB,
                total_games: totalGames
            })
        });

        const result = await response.json();
        if (response.ok) {
            currentBattleId = result.battle_id;
            isBattleMode = true;
            battleConfig = { agentA, agentB, totalGames };
            matchHistory = [];

            // 更新UI
            document.getElementById('battle-config').style.display = 'none';
            document.getElementById('current-player').style.display = 'block';
            document.getElementById('game-score').style.display = 'block';
            document.querySelector('.controls button:first-child').textContent = '退出对战';

            appendMatchLog(`=== 对战开始 ===`, 'info');
            appendMatchLog(`Agent A: ${result.agent_a} vs Agent B: ${result.agent_b}`);
            appendMatchLog(`总局数：${totalGames}局`);
            appendMatchLog(`第 1 局比赛开始`, 'success');

            await fetchState();

            // 检查当前玩家是否是AI，如果是自动击球
            await checkAIAction();
        } else {
            updateStatus('创建对战失败：' + result.detail);
        }
    } catch (error) {
        updateStatus('创建对战失败：' + error.message);
    }
}

// 执行对战击球
async function executeBattleShot(action) {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/battle/${currentBattleId}/next`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action })
        });

        const result = await response.json();
        if (result.start_state && result.end_state) {
            // 播放击球动画
            await playAnimation(result.start_state, result.end_state);
            gameState = result.end_state;

            // 记录击球信息到赛程
            const playerName = result.current_player === 'A' ? battleConfig.agentA : battleConfig.agentB;
            const shotNum = gameState.turn;
            appendMatchLog(`[第${shotNum}次击球] ${playerName} 回合`);
            if (result.message.includes('进球')) {
                appendMatchLog(`  ${result.message}`, 'success');
            }

            // 检查是否局结束
            if (result.game_status === 'finished' && result.winner) {
                const winner = result.winner === 'A' ? battleConfig.agentA : battleConfig.agentB;
                appendMatchLog(`第 ${gameState.turn} 局结束，${winner} 获胜！`, 'success');

                // 获取最新状态更新比分
                const statusRes = await fetch(`${CONFIG.API_BASE}/battle/${currentBattleId}/status`);
                const status = await statusRes.json();
                const results = status.state.results;
                appendMatchLog(`当前比分：A ${results.A} - ${results.B} B`, 'info');
            }

            updateUI();
            updateStatus(result.message);

            // 检查是否需要继续AI动作
            setTimeout(() => checkAIAction(), 500);
        }
    } catch (error) {
        updateStatus('击球失败：' + error.message);
    }
}

// 检查当前玩家是否是AI，自动执行动作
async function checkAIAction() {
    if (!isBattleMode || !currentBattleId || isAnimating) return;

    try {
        const statusRes = await fetch(`${CONFIG.API_BASE}/battle/${currentBattleId}/status`);
        const status = await statusRes.json();
        const currentPlayer = status.state.current_player;
        const agentType = currentPlayer === 'A' ? battleConfig.agentA : battleConfig.agentB;

        // 如果是AI类型，自动执行击球
        if (agentType !== 'human') {
            updateStatus(`${agentType} 正在思考...`);
            await executeBattleShot(null); // 不传action，由AI决策
        }
    } catch (error) {
        updateStatus('AI动作执行失败：' + error.message);
    }
}

// 追加赛程日志
function appendMatchLog(text, type = 'normal') {
    const logEl = document.getElementById('match-log');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = text;
    logEl.appendChild(entry);
    // 自动滚动到底部
    logEl.scrollTop = logEl.scrollHeight;
}

async function executeShot(phi, v0) {
    if (isAnimating) return;

    if (isBattleMode && currentBattleId) {
        // 对战模式下调用对战击球接口
        await executeBattleShot({ phi, V0: v0 });
    } else {
        // 普通模式下调用原接口
        try {
            const response = await fetch(`${CONFIG.API_BASE}/shot`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: { phi, V0: v0 }, state: gameState })
            });

            const result = await response.json();
            if (result.success) {
                await playAnimation(result.start_state, result.end_state);
                gameState = result.end_state;
                updateUI();
                if (result.message !== '击球完成') {
                    showPocketed(result.message);
                }
                updateStatus('击球完成');
            }
        } catch (error) {
            updateStatus('击球失败：' + error.message);
        }
    }
}

async function resetGame() {
    if (isBattleMode && currentBattleId) {
        // 对战模式下退出对战
        try {
            await fetch(`${CONFIG.API_BASE}/battle/${currentBattleId}`, { method: 'DELETE' });
        } catch (e) { /* 忽略删除错误 */ }

        // 重置对战状态
        isBattleMode = false;
        currentBattleId = null;
        battleConfig = null;

        // 更新UI
        document.getElementById('battle-config').style.display = 'flex';
        document.getElementById('current-player').style.display = 'none';
        document.getElementById('game-score').style.display = 'none';
        document.querySelector('.controls button:first-child').textContent = '重新开始';

        appendMatchLog('=== 已退出对战 ===', 'warning');
        appendMatchLog('选择对战双方和局数，点击"开始对战"开始比赛');
        appendMatchLog('或直接点击白球进行自由练习');
    }

    // 重置游戏状态
    try {
        const response = await fetch(`${CONFIG.API_BASE}/reset`, { method: 'POST' });
        const result = await response.json();
        if (result.success) {
            gameState = result.state;
            updateUI();
            render();
            updateStatus('游戏已重置');
        }
    } catch (error) {
        updateStatus('重置失败：' + error.message);
    }
}

async function updateUI() {
    if (!gameState) return;
    document.getElementById('turn').textContent = gameState.turn;
    document.getElementById('red-score').textContent = gameState.red_score || 0;
    document.getElementById('yellow-score').textContent = gameState.yellow_score || 0;
    document.getElementById('remaining').textContent = Object.keys(gameState.balls).length - 1;

    // 如果是对战模式，更新对战相关信息
    if (isBattleMode && currentBattleId) {
        try {
            const statusRes = await fetch(`${CONFIG.API_BASE}/battle/${currentBattleId}/status`);
            const status = await statusRes.json();
            const results = status.state.results;

            // 更新当前玩家
            const currentPlayerEl = document.querySelector('#current-player .info-value');
            currentPlayerEl.textContent = status.state.current_player;
            currentPlayerEl.style.color = status.state.current_player === 'A' ? '#4fc3f7' : '#f1c40f';

            // 更新大比分
            document.querySelector('#game-score .info-value').textContent = `${results.A} - ${results.B}`;
        } catch (e) { /* 忽略错误 */ }
    }
}

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

function showPocketed(message) {
    const el = document.getElementById('pocketed');
    el.textContent = message;
    el.style.display = 'block';
    setTimeout(() => { el.style.display = 'none'; }, 2000);
}

// ==================== 事件处理 ====================
function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

canvas.addEventListener('mousedown', handlePointerDown);
canvas.addEventListener('touchstart', handlePointerDown, { passive: false });
canvas.addEventListener('mousemove', handlePointerMove);
canvas.addEventListener('touchmove', handlePointerMove, { passive: false });
canvas.addEventListener('mouseup', handlePointerUp);
canvas.addEventListener('touchend', handlePointerUp);

function handlePointerDown(e) {
    if (isAnimating || !gameState) return;
    e.preventDefault();

    const coords = getCanvasCoordinates(e);
    const [gameX, gameY] = toGame(coords.x, coords.y);
    const cueBall = gameState.balls['cue'];

    if (cueBall && distance(cueBall.pos, [gameX, gameY]) < CONFIG.BALL_RADIUS * 3) {
        isAiming = true;
        aimStart = [...cueBall.pos];
        aimCurrent = [gameX, gameY];
        updateStatus('向后拖拽调整力度和方向');
        render();
    }
}

function handlePointerMove(e) {
    if (!isAiming || isAnimating) return;
    e.preventDefault();

    const coords = getCanvasCoordinates(e);
    aimCurrent = toGame(coords.x, coords.y);

    if (aimStart && distance(aimStart, aimCurrent) > CONFIG.CUE_MIN_LENGTH) {
        isDragging = true;
    }
    render();
}

function handlePointerUp(e) {
    if (!isAiming || isAnimating) return;
    isAiming = false;

    if (isDragging && aimStart && aimCurrent) {
        const dx = aimStart[0] - aimCurrent[0];
        const dy = aimStart[1] - aimCurrent[1];
        const dist = distance(aimStart, aimCurrent);

        if (dist > CONFIG.CUE_MIN_LENGTH) {
            const phi = Math.atan2(dy, dx);
            const v0 = Math.min((dist / CONFIG.CUE_MAX_LENGTH) * 100, 100);
            if (v0 > 10) {
                updateStatus('击球中...');
                executeShot(phi, v0);
            }
        }
    }

    isDragging = false;
    aimStart = null;
    aimCurrent = null;
    power = 0;
    updateStatus('点击白球进入瞄准模式');
    render();
}

// ==================== 初始化 ====================
function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    // 设置 Canvas 实际分辨率等于 CSS 显示尺寸
    canvas.width = Math.floor(rect.width);
    canvas.height = Math.floor(rect.height);
}

async function init() {
    updateStatus('正在连接服务器...');

    // 初始化 Canvas 尺寸
    resizeCanvas();
    // 窗口大小变化时重新调整 Canvas
    window.addEventListener('resize', resizeCanvas);

    await fetchState();

    function renderLoop() {
        if (!isAnimating) render();
        requestAnimationFrame(renderLoop);
    }
    requestAnimationFrame(renderLoop);
    updateStatus('点击白球进入瞄准模式');
}

init();
