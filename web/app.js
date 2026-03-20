/**
 * CueZero 前端逻辑 - 极简扁平化风格
 */

// ==================== 配置 ====================
const CONFIG = {
    API_BASE: 'http://localhost:8000/api',
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
        const response = await fetch(`${CONFIG.API_BASE}/state`);
        gameState = await response.json();
        updateUI();
        render();
        updateStatus('状态已更新');
    } catch (error) {
        updateStatus('获取状态失败：' + error.message);
    }
}

async function executeShot(phi, v0) {
    if (isAnimating) return;

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

async function resetGame() {
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

function updateUI() {
    if (!gameState) return;
    document.getElementById('turn').textContent = gameState.turn;
    document.getElementById('red-score').textContent = gameState.red_score || 0;
    document.getElementById('yellow-score').textContent = gameState.yellow_score || 0;
    document.getElementById('remaining').textContent = Object.keys(gameState.balls).length - 1;
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
