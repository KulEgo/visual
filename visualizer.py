"""
Главный модуль библиотеки viz3d.

Запускает локальный сервер с веб-интерфейсом, где пользователь
вводит математическую функцию и видит её 3D-визуализацию.

Поддерживаются:
- z = f(x, y)         — обычная поверхность
- (x, y, z) = f(u, v) — параметрическая поверхность (3 выражения через ;)
"""

import ast
import webbrowser
import threading
import socket
from typing import Optional

import numpy as np
from flask import Flask, render_template_string, request, jsonify


# === Безопасный калькулятор выражений ===

# Разрешённые функции и константы
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

# Разрешённые узлы AST
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
    return eval(code, {"__builtins__": {}}, namespace)


# === Подготовка данных для Plotly ===

def _compute_data(expr: str, x_min, x_max, y_min, y_max, resolution):
    """Определяет тип графика и считает массивы для отрисовки."""
    parts = [p.strip() for p in expr.split(";") if p.strip()]

    if len(parts) == 1:
        # z = f(x, y)
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        Z = _evaluate(parts[0], ("x", "y"), (X, Y))
        Z = np.broadcast_to(np.asarray(Z, dtype=float), X.shape)
        return {
            "traces": [{
                "type": "surface",
                "x": X.tolist(), "y": Y.tolist(), "z": Z.tolist(),
                "colorscale": "Viridis", "showscale": True,
            }],
        }

    if len(parts) == 3:
        # (x, y, z) = f(u, v)
        u = np.linspace(x_min, x_max, resolution)
        v = np.linspace(y_min, y_max, resolution)
        U, V = np.meshgrid(u, v)
        X = np.broadcast_to(np.asarray(_evaluate(parts[0], ("u", "v"), (U, V)), dtype=float), U.shape)
        Y = np.broadcast_to(np.asarray(_evaluate(parts[1], ("u", "v"), (U, V)), dtype=float), U.shape)
        Z = np.broadcast_to(np.asarray(_evaluate(parts[2], ("u", "v"), (U, V)), dtype=float), U.shape)
        return {
            "traces": [{
                "type": "surface",
                "x": X.tolist(), "y": Y.tolist(), "z": Z.tolist(),
                "colorscale": "Plasma", "showscale": True,
            }],
        }

    raise ValueError(
        f"Ожидается 1 выражение (z = f(x, y)) или 3 через ';' "
        f"((x, y, z) = f(u, v)), получено: {len(parts)}"
    )


# === HTML с полем ввода ===

_HTML_TEMPLATE = """
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
        #plot { flex-grow: 1; min-height: 0; }
        .error {
            background: #4a1d1d; color: #ff8888;
            padding: 10px 20px; border-bottom: 1px solid #6a2d2d;
            font-family: "SF Mono", Monaco, Consolas, monospace;
            font-size: 13px; display: none;
        }
        .error.visible { display: block; }
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
    </div>
    <div id="error" class="error"></div>
    <div id="plot"></div>

    <script>
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
        function hideError() {
            document.getElementById('error').classList.remove('visible');
        }

        async function render() {
            const btn = document.getElementById('btn');
            btn.disabled = true;
            const oldText = btn.textContent;
            btn.textContent = '...';
            hideError();

            const payload = {
                expr: document.getElementById('expr').value,
                x_min: parseFloat(document.getElementById('xmin').value),
                x_max: parseFloat(document.getElementById('xmax').value),
                y_min: parseFloat(document.getElementById('ymin').value),
                y_max: parseFloat(document.getElementById('ymax').value),
                resolution: parseInt(document.getElementById('res').value),
            };

            try {
                const resp = await fetch('/data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();
                if (data.error) { showError(data.error); return; }

                const layout = {
                    paper_bgcolor: '#0f1419',
                    plot_bgcolor: '#0f1419',
                    font: { color: '#e6e6e6' },
                    margin: { l: 0, r: 0, t: 0, b: 0 },
                    scene: {
                        xaxis: { gridcolor: '#2a3142', zerolinecolor: '#3a4356' },
                        yaxis: { gridcolor: '#2a3142', zerolinecolor: '#3a4356' },
                        zaxis: { gridcolor: '#2a3142', zerolinecolor: '#3a4356' },
                        camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
                    }
                };
                Plotly.newPlot('plot', data.traces, layout, { responsive: true });
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

        // Стартовый график
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
