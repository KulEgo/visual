"""
Главный модуль библиотеки viz3d.
 
Запускает локальный сервер с веб-интерфейсом, где пользователь
вводит математическую функцию и видит её 3D-визуализацию.
 
Поддерживаются:
- z = f(x, y)         — обычная поверхность
- (x, y, z) = f(u, v) — параметрическая поверхность (3 выражения через ;)
"""
 
import ast
import math
import webbrowser
import threading
import socket
from typing import Optional
 
import numpy as np
from flask import Flask, render_template_string, request, jsonify
 
 
# === Безопасный калькулятор выражений ===
 
_SAFE_NAMES = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10,
    "sqrt": np.sqrt, "cbrt": np.cbrt,
    "pow": np.power, "power": np.power,
    "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil, "round": np.round,
    "min": np.minimum, "max": np.maximum,
    "minimum": np.minimum, "maximum": np.maximum,
    "pi": np.pi, "e": np.e, "tau": 2 * np.pi,
}
 
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
    ast.Name, ast.Load, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)
 
 
def _safe_compile(expr: str):
    """Безопасно компилирует выражение. ValueError при недопустимых конструкциях."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Синтаксическая ошибка: {e.msg}")
 
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Запрещённая конструкция: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Разрешены только прямые вызовы функций")
            if node.func.id not in _SAFE_NAMES:
                raise ValueError(f"Неизвестная функция: {node.func.id}")
        if isinstance(node, ast.Name):
            allowed = set(_SAFE_NAMES.keys()) | {"x", "y", "u", "v"}
            if node.id not in allowed:
                raise ValueError(f"Неизвестное имя: {node.id}")
 
    return compile(tree, "<expr>", "eval")
 
 
def _evaluate(expr: str, var_names: tuple, var_values: tuple):
    """Безопасно вычисляет выражение, подставляя значения переменных."""
    code = _safe_compile(expr)
    namespace = dict(_SAFE_NAMES)
    for name, value in zip(var_names, var_values):
        namespace[name] = value
    # numpy сам подавляет предупреждения о log(-1), 1/0 → выдаст nan/inf
    with np.errstate(all="ignore"):
        return eval(code, {"__builtins__": {}}, namespace)
 
 
def _has_any_valid_point(arrays):
    """
    True если есть хотя бы одна точка, где ВСЕ массивы конечны одновременно.
    Для поверхности arrays=[Z], для параметрики arrays=[X, Y, Z].
    """
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return bool(np.any(mask))
 
 
# === Подготовка данных для Plotly ===
 
def _compute_data(expr: str, x_min, x_max, y_min, y_max, resolution):
    """Считает массивы для отрисовки. Возвращает dict с traces или {empty: True}."""
    parts = [p.strip() for p in expr.split(";") if p.strip()]
 
    if len(parts) == 1:
        # z = f(x, y)
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        Z = _evaluate(parts[0], ("x", "y"), (X, Y))
        Z = np.broadcast_to(np.asarray(Z, dtype=float), X.shape).copy()
 
        if not _has_any_valid_point([Z]):
            return {"empty": True, "x_range": [x_min, x_max], "y_range": [y_min, y_max]}
 
        # Заменяем inf на nan — Plotly корректно делает в них дыры
        Z[~np.isfinite(Z)] = np.nan
        return {
            "empty": False,
            "traces": [{
                "type": "surface",
                "x": X.tolist(), "y": Y.tolist(),
                "z": [[None if math.isnan(v) else v for v in row] for row in Z],
                "colorscale": "Viridis", "showscale": True,
            }],
        }
 
    if len(parts) == 3:
        # (x, y, z) = f(u, v)
        u = np.linspace(x_min, x_max, resolution)
        v = np.linspace(y_min, y_max, resolution)
        U, V = np.meshgrid(u, v)
        X = np.broadcast_to(np.asarray(_evaluate(parts[0], ("u", "v"), (U, V)), dtype=float), U.shape).copy()
        Y = np.broadcast_to(np.asarray(_evaluate(parts[1], ("u", "v"), (U, V)), dtype=float), U.shape).copy()
        Z = np.broadcast_to(np.asarray(_evaluate(parts[2], ("u", "v"), (U, V)), dtype=float), U.shape).copy()
 
        if not _has_any_valid_point([X, Y, Z]):
            return {"empty": True, "x_range": [-1, 1], "y_range": [-1, 1]}
 
        for arr in (X, Y, Z):
            arr[~np.isfinite(arr)] = np.nan
 
        return {
            "empty": False,
            "traces": [{
                "type": "surface",
                "x": [[None if math.isnan(v) else v for v in row] for row in X],
                "y": [[None if math.isnan(v) else v for v in row] for row in Y],
                "z": [[None if math.isnan(v) else v for v in row] for row in Z],
                "colorscale": "Plasma", "showscale": True,
            }],
        }
 
    raise ValueError(
        f"Ожидается 1 выражение (z = f(x, y)) или 3 через ';' "
        f"((x, y, z) = f(u, v)), получено: {len(parts)}"
    )
 
 
# === HTML с полем ввода и стрелочными осями ===
 
_HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>viz3d — 3D визуализация</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f1419; color: #e6e6e6;
            display: flex; flex-direction: column;
            height: 100vh; overflow: hidden;
        }
        .header {
            padding: 14px 20px;
            background: #1a1f2e;
            border-bottom: 1px solid #2a3142;
            display: flex; align-items: flex-end; gap: 12px; flex-wrap: wrap;
        }
        .title {
            font-size: 16px; font-weight: 600; margin: 0;
            margin-right: 8px; color: #e6e6e6; white-space: nowrap;
            padding-bottom: 8px;
        }
        .field { display: flex; flex-direction: column; gap: 3px; }
        .field label {
            font-size: 11px; color: #8b95a7;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        input[type="text"], input[type="number"] {
            background: #0f1419; border: 1px solid #2a3142;
            color: #e6e6e6; padding: 7px 10px; border-radius: 5px;
            font-family: "SF Mono", Monaco, Consolas, monospace;
            font-size: 13px; outline: none;
            transition: border-color 0.15s;
        }
        input[type="text"]:focus, input[type="number"]:focus { border-color: #4a90e2; }
        input.expr { min-width: 320px; flex-grow: 1; }
        input.range-input { width: 65px; }
        .range-group { display: flex; align-items: center; gap: 4px; }
        .range-group span { color: #5a6478; font-size: 12px; }
        button {
            background: #4a90e2; color: white; border: none;
            padding: 9px 18px; border-radius: 5px;
            font-size: 13px; font-weight: 500; cursor: pointer;
            transition: background 0.15s;
        }
        button:hover { background: #5aa0f2; }
        button:disabled { background: #3a4356; cursor: not-allowed; }
        .examples {
            padding: 8px 20px; background: #151a26;
            border-bottom: 1px solid #2a3142;
            font-size: 12px; color: #8b95a7;
            display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
        }
        .examples span.label { color: #5a6478; }
        .chip {
            background: #1a1f2e; padding: 4px 10px; border-radius: 12px;
            cursor: pointer;
            font-family: "SF Mono", Monaco, Consolas, monospace;
            transition: background 0.15s, color 0.15s;
            border: 1px solid #2a3142;
        }
        .chip:hover { background: #2a3142; color: #e6e6e6; }
        .plot-wrap { flex-grow: 1; min-height: 0; position: relative; }
        #plot { width: 100%; height: 100%; }
        .error {
            background: #4a1d1d; color: #ff8888;
            padding: 10px 20px; border-bottom: 1px solid #6a2d2d;
            font-family: "SF Mono", Monaco, Consolas, monospace;
            font-size: 13px; display: none;
        }
        .error.visible { display: block; }
        /* Плашка "графика не существует" — поверх графика снизу */
        .no-graph {
            position: absolute;
            left: 50%;
            bottom: 24px;
            transform: translateX(-50%);
            background: rgba(74, 29, 29, 0.92);
            color: #ffb0b0;
            padding: 12px 24px;
            border-radius: 8px;
            border: 1px solid #6a2d2d;
            font-size: 14px;
            font-weight: 500;
            display: none;
            pointer-events: none;
            backdrop-filter: blur(4px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        }
        .no-graph.visible { display: block; }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">viz3d</h1>
        <div class="field" style="flex-grow: 1;">
            <label>Ваша функция</label>
            <input id="expr" class="expr" type="text" value="sin(sqrt(x**2 + y**2))"
                   placeholder="например: x**2 + y**2  или  cos(u)*sin(v); sin(u)*sin(v); cos(v)">
        </div>
        <div class="field">
            <label>Диапазон 1</label>
            <div class="range-group">
                <input id="xmin" class="range-input" type="number" value="-5" step="0.5">
                <span>—</span>
                <input id="xmax" class="range-input" type="number" value="5" step="0.5">
            </div>
        </div>
        <div class="field">
            <label>Диапазон 2</label>
            <div class="range-group">
                <input id="ymin" class="range-input" type="number" value="-5" step="0.5">
                <span>—</span>
                <input id="ymax" class="range-input" type="number" value="5" step="0.5">
            </div>
        </div>
        <div class="field">
            <label>Разрешение</label>
            <input id="res" class="range-input" type="number" value="80" min="10" max="300" step="10">
        </div>
        <button id="btn" onclick="render()">Построить</button>
    </div>
    <div class="examples">
        <span class="label">Примеры:</span>
        <span class="chip" onclick="setExpr('x**2 + y**2')">параболоид</span>
        <span class="chip" onclick="setExpr('x**2 - y**2')">седло</span>
        <span class="chip" onclick="setExpr('sin(sqrt(x**2 + y**2))')">волны</span>
        <span class="chip" onclick="setExpr('sin(x) * cos(y)')">волны&nbsp;2</span>
        <span class="chip" onclick="setExpr('exp(-(x**2+y**2)/4) * cos(x*y)')">гауссиана</span>
        <span class="chip" onclick="setExpr('cos(u)*sin(v); sin(u)*sin(v); cos(v)', 0, 6.28, 0, 3.14)">сфера</span>
        <span class="chip" onclick="setExpr('(2+cos(v))*cos(u); (2+cos(v))*sin(u); sin(v)', 0, 6.28, 0, 6.28)">тор</span>
        <span class="chip" onclick="setExpr('log(x) + log(y)', -2, -1, -2, -1)">log при&nbsp;x&lt;0&nbsp;(нет&nbsp;графика)</span>
    </div>
    <div id="error" class="error"></div>
    <div class="plot-wrap">
        <div id="plot"></div>
        <div id="no-graph" class="no-graph">⚠ Такого графика не существует</div>
    </div>
 
    <script>
        // === Построение стрелочных осей x, y, z вместо стандартной коробки Plotly ===
        function buildArrowAxes(bounds) {
            // bounds = { xmin, xmax, ymin, ymax, zmin, zmax }
            // Слегка расширяем диапазоны, чтобы стрелки торчали за поверхность
            const pad = 0.15;
            const ext = (lo, hi) => {
                const span = (hi - lo) || 1;
                return [lo - span * pad, hi + span * pad];
            };
            const [xLo, xHi] = ext(bounds.xmin, bounds.xmax);
            const [yLo, yHi] = ext(bounds.ymin, bounds.ymax);
            const [zLo, zHi] = ext(bounds.zmin, bounds.zmax);
 
            // Линии осей: от минимума до максимума по каждой координате через 0,
            // но проще — просто от lo до hi (полная ось)
            const axisLines = [
                {  // X-ось
                    type: 'scatter3d', mode: 'lines',
                    x: [xLo, xHi], y: [0, 0], z: [0, 0],
                    line: { color: '#ff6b6b', width: 4 },
                    hoverinfo: 'skip', showlegend: false,
                },
                {  // Y-ось
                    type: 'scatter3d', mode: 'lines',
                    x: [0, 0], y: [yLo, yHi], z: [0, 0],
                    line: { color: '#51cf66', width: 4 },
                    hoverinfo: 'skip', showlegend: false,
                },
                {  // Z-ось
                    type: 'scatter3d', mode: 'lines',
                    x: [0, 0], y: [0, 0], z: [zLo, zHi],
                    line: { color: '#4dabf7', width: 4 },
                    hoverinfo: 'skip', showlegend: false,
                },
            ];
 
            // Подписи "x", "y", "z" возле наконечников стрелок
            const labels = {
                type: 'scatter3d', mode: 'text',
                x: [xHi, 0, 0],
                y: [0, yHi, 0],
                z: [0, 0, zHi],
                text: ['<b>x</b>', '<b>y</b>', '<b>z</b>'],
                textfont: { size: 18, color: '#e6e6e6' },
                textposition: 'top center',
                hoverinfo: 'skip', showlegend: false,
            };
 
            // Стрелки на концах осей — рисуем через cone (3D конусы вдоль оси)
            const coneSize = Math.max(xHi - xLo, yHi - yLo, zHi - zLo) * 0.04;
            const arrows = {
                type: 'cone',
                x: [xHi, 0, 0],
                y: [0, yHi, 0],
                z: [0, 0, zHi],
                u: [1, 0, 0],   // направление стрелки X
                v: [0, 1, 0],   // направление стрелки Y
                w: [0, 0, 1],   // направление стрелки Z
                sizemode: 'absolute',
                sizeref: coneSize,
                anchor: 'tip',
                colorscale: [[0, '#888'], [1, '#888']],
                showscale: false,
                hoverinfo: 'skip',
                lighting: { ambient: 0.8 },
            };
 
            return [...axisLines, labels, arrows];
        }
 
        function computeBounds(traces) {
            // Находит min/max по всем traces (поверхностям) для x, y, z
            let xmin = Infinity, xmax = -Infinity;
            let ymin = Infinity, ymax = -Infinity;
            let zmin = Infinity, zmax = -Infinity;
            for (const t of traces) {
                for (const row of t.x) for (const v of row)
                    if (v !== null && isFinite(v)) { xmin = Math.min(xmin, v); xmax = Math.max(xmax, v); }
                for (const row of t.y) for (const v of row)
                    if (v !== null && isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
                for (const row of t.z) for (const v of row)
                    if (v !== null && isFinite(v)) { zmin = Math.min(zmin, v); zmax = Math.max(zmax, v); }
            }
            // Если что-то осталось бесконечностью (нет данных) — дефолт
            if (!isFinite(xmin)) { xmin = -1; xmax = 1; }
            if (!isFinite(ymin)) { ymin = -1; ymax = 1; }
            if (!isFinite(zmin)) { zmin = -1; zmax = 1; }
            // Гарантируем включение нуля (чтобы стрелочные оси не висели в воздухе)
            xmin = Math.min(xmin, 0); xmax = Math.max(xmax, 0);
            ymin = Math.min(ymin, 0); ymax = Math.max(ymax, 0);
            zmin = Math.min(zmin, 0); zmax = Math.max(zmax, 0);
            return { xmin, xmax, ymin, ymax, zmin, zmax };
        }
 
        function drawPlot(surfaceTraces, bounds) {
            const axisTraces = buildArrowAxes(bounds);
            const layout = {
                paper_bgcolor: '#0f1419',
                plot_bgcolor: '#0f1419',
                font: { color: '#e6e6e6' },
                margin: { l: 0, r: 0, t: 0, b: 0 },
                showlegend: false,
                scene: {
                    // Прячем стандартную "коробку" — оставляем только наши стрелки
                    xaxis: { visible: false, range: [bounds.xmin * 1.2, bounds.xmax * 1.2] },
                    yaxis: { visible: false, range: [bounds.ymin * 1.2, bounds.ymax * 1.2] },
                    zaxis: { visible: false, range: [bounds.zmin * 1.2, bounds.zmax * 1.2] },
                    aspectmode: 'cube',
                    bgcolor: '#0f1419',
                    camera: { eye: { x: 1.7, y: 1.7, z: 1.3 } }
                }
            };
            Plotly.newPlot('plot', [...surfaceTraces, ...axisTraces], layout, {
                responsive: true,
                displaylogo: false,
            });
        }
 
        function setExpr(e, xmin, xmax, ymin, ymax) {
            document.getElementById('expr').value = e;
            if (xmin !== undefined) {
                document.getElementById('xmin').value = xmin;
                document.getElementById('xmax').value = xmax;
                document.getElementById('ymin').value = ymin;
                document.getElementById('ymax').value = ymax;
            }
            render();
        }
 
        function showError(msg) {
            const el = document.getElementById('error');
            el.textContent = '⚠ ' + msg;
            el.classList.add('visible');
        }
        function hideError() { document.getElementById('error').classList.remove('visible'); }
 
        function showNoGraph() { document.getElementById('no-graph').classList.add('visible'); }
        function hideNoGraph() { document.getElementById('no-graph').classList.remove('visible'); }
 
        async function render() {
            const btn = document.getElementById('btn');
            btn.disabled = true;
            btn.textContent = '...';
            hideError();
            hideNoGraph();
 
            const xmin = parseFloat(document.getElementById('xmin').value);
            const xmax = parseFloat(document.getElementById('xmax').value);
            const ymin = parseFloat(document.getElementById('ymin').value);
            const ymax = parseFloat(document.getElementById('ymax').value);
 
            const payload = {
                expr: document.getElementById('expr').value,
                x_min: xmin, x_max: xmax, y_min: ymin, y_max: ymax,
                resolution: parseInt(document.getElementById('res').value),
            };
 
            try {
                const resp = await fetch('/data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();
 
                if (data.error) {
                    // Ошибка парсинга — рисуем пустые оси и красную плашку сверху
                    showError(data.error);
                    drawPlot([], { xmin: -1, xmax: 1, ymin: -1, ymax: 1, zmin: -1, zmax: 1 });
                    return;
                }
 
                if (data.empty) {
                    // Все значения NaN/Inf — оси по запрошенному диапазону, плашка снизу
                    drawPlot([], {
                        xmin: Math.min(xmin, 0), xmax: Math.max(xmax, 0),
                        ymin: Math.min(ymin, 0), ymax: Math.max(ymax, 0),
                        zmin: -1, zmax: 1
                    });
                    showNoGraph();
                    return;
                }
 
                const bounds = computeBounds(data.traces);
                drawPlot(data.traces, bounds);
            } catch (e) {
                showError('Ошибка сети: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Построить';
            }
        }
 
        document.getElementById('expr').addEventListener('keydown', e => {
            if (e.key === 'Enter') render();
        });
 
        render();
    </script>
</body>
</html>
"""
 
 
# === Запуск сервера ===
 
def _find_free_port(start: int = 5000) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start
 
 
def plot(port: Optional[int] = None, open_browser: bool = True):
    """
    Запускает локальный сервер с веб-интерфейсом для построения 3D-графиков.
 
    На странице пользователь вводит функцию и нажимает «Построить».
    Поддерживаемый синтаксис:
      • z = f(x, y):                'sin(x) * cos(y)'
      • (x, y, z) = f(u, v):        'cos(u)*sin(v); sin(u)*sin(v); cos(v)'
                                    (три выражения через точку с запятой)
 
    Доступные функции: sin, cos, tan, asin, acos, atan, atan2,
    sinh, cosh, tanh, exp, log, log2, log10, sqrt, cbrt, abs, sign,
    floor, ceil, round, min, max, pow.
    Константы: pi, e, tau.
 
    Если функция нигде в области не определена (все значения NaN/Inf),
    оси остаются на месте, а внизу появляется надпись «Такого графика не существует».
 
    Параметры
    ---------
    port : int, optional
        Порт сервера. Если None — выбирается автоматически.
    open_browser : bool
        Автоматически открывать браузер.
    """
    app = Flask(__name__)
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
 
    @app.route("/")
    def index():
        return render_template_string(_HTML_TEMPLATE)
 
    @app.route("/data", methods=["POST"])
    def get_data():
        try:
            params = request.get_json(force=True)
            data = _compute_data(
                params["expr"],
                float(params["x_min"]), float(params["x_max"]),
                float(params["y_min"]), float(params["y_max"]),
                int(params["resolution"]),
            )
            return jsonify(data)
        except ValueError as e:
            return jsonify({"error": str(e)})
        except ZeroDivisionError:
            return jsonify({"error": "Деление на ноль"})
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"})
 
    if port is None:
        port = _find_free_port(5000)
 
    url = f"http://127.0.0.1:{port}"
    print(f"\n  viz3d → сервер запущен: {url}")
    print(f"  Откройте браузер и введите свою функцию")
    print(f"  Ctrl+C для остановки\n")
 
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
 
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
