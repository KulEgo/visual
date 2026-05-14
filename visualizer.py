"""
Главный модуль библиотеки viz3d.

Поддерживает два режима:
- 2 параметра (x, y) -> z = f(x, y): строит поверхность z = f(x, y)
- 2 параметра (u, v) -> (x, y, z): параметрическая поверхность в 3D
"""

import inspect
import json
import webbrowser
import threading
import socket
from typing import Callable, Optional, Tuple

import numpy as np
from flask import Flask, render_template_string, jsonify


# HTML-шаблон с Plotly для интерактивной 3D-визуализации
_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>viz3d — 3D визуализация</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f1419;
            color: #e6e6e6;
        }
        .header {
            padding: 16px 24px;
            background: #1a1f2e;
            border-bottom: 1px solid #2a3142;
        }
        .header h1 {
            margin: 0;
            font-size: 18px;
            font-weight: 600;
        }
        .header p {
            margin: 4px 0 0 0;
            font-size: 13px;
            color: #8b95a7;
            font-family: "SF Mono", Monaco, Consolas, monospace;
        }
        #plot {
            width: 100vw;
            height: calc(100vh - 70px);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>3D Визуализация</h1>
        <p>{{ title }}</p>
    </div>
    <div id="plot"></div>
    <script>
        fetch('/data')
            .then(r => r.json())
            .then(data => {
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
            });
    </script>
</body>
</html>
"""


def _find_free_port(start: int = 5000) -> int:
    """Находит свободный порт начиная со start."""
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def _compute_surface(
    func: Callable,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int,
) -> dict:
    """Вычисляет поверхность z = f(x, y)."""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Пробуем сначала векторизованный вызов, иначе поэлементно
    try:
        Z = func(X, Y)
        Z = np.asarray(Z, dtype=float)
        if Z.shape != X.shape:
            raise ValueError("shape mismatch")
    except Exception:
        Z = np.array([[float(func(xi, yi)) for xi in x] for yi in y])

    return {
        "traces": [{
            "type": "surface",
            "x": X.tolist(),
            "y": Y.tolist(),
            "z": Z.tolist(),
            "colorscale": "Viridis",
            "showscale": True,
        }]
    }


def _compute_parametric(
    func: Callable,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    resolution: int,
) -> dict:
    """Вычисляет параметрическую поверхность (x, y, z) = f(u, v)."""
    u = np.linspace(u_range[0], u_range[1], resolution)
    v = np.linspace(v_range[0], v_range[1], resolution)
    U, V = np.meshgrid(u, v)

    X = np.zeros_like(U)
    Y = np.zeros_like(U)
    Z = np.zeros_like(U)

    try:
        result = func(U, V)
        X, Y, Z = (np.asarray(r, dtype=float) for r in result)
    except Exception:
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                x, y, z = func(U[i, j], V[i, j])
                X[i, j], Y[i, j], Z[i, j] = x, y, z

    return {
        "traces": [{
            "type": "surface",
            "x": X.tolist(),
            "y": Y.tolist(),
            "z": Z.tolist(),
            "colorscale": "Plasma",
            "showscale": True,
        }]
    }


def plot(
    func: Callable,
    x_range: Tuple[float, float] = (-5, 5),
    y_range: Tuple[float, float] = (-5, 5),
    resolution: int = 80,
    port: Optional[int] = None,
    open_browser: bool = True,
    title: Optional[str] = None,
):
    """
    Визуализирует функцию в 3D и запускает локальный веб-сервер.

    Параметры
    ---------
    func : callable
        Функция от 2 или 3 параметров.
        - 2 параметра: f(x, y) -> z   (строит поверхность)
        - 2 параметра: f(u, v) -> (x, y, z)  (параметрическая поверхность)
          Тип определяется автоматически по возвращаемому значению.
    x_range, y_range : tuple(float, float)
        Диапазоны входных параметров (по умолчанию (-5, 5)).
    resolution : int
        Число точек по каждой оси (по умолчанию 80).
    port : int, optional
        Порт сервера. Если None — выбирается автоматически.
    open_browser : bool
        Автоматически открывать браузер.
    title : str, optional
        Заголовок графика (по умолчанию — исходник функции).
    """
    sig = inspect.signature(func)
    n_params = len(sig.parameters)

    if n_params != 2:
        raise ValueError(
            f"Функция должна принимать 2 параметра, получено: {n_params}. "
            "Для z = f(x, y) или (x, y, z) = f(u, v) используется 2 входных параметра."
        )

    # Определяем тип функции по тестовому вызову
    try:
        test_val = func(0.5, 0.5)
        is_parametric = (
            hasattr(test_val, "__len__") and len(test_val) == 3
            and not isinstance(test_val, (str, bytes))
        )
    except Exception:
        is_parametric = False

    if is_parametric:
        data = _compute_parametric(func, x_range, y_range, resolution)
        mode = "параметрическая поверхность (x, y, z) = f(u, v)"
    else:
        data = _compute_surface(func, x_range, y_range, resolution)
        mode = "поверхность z = f(x, y)"

    # Заголовок
    if title is None:
        try:
            src = inspect.getsource(func).strip()
            title = f"{mode} | {src[:120]}"
        except (OSError, TypeError):
            title = mode

    # Flask-приложение
    app = Flask(__name__)
    # Отключаем шумные логи запросов
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return render_template_string(_HTML_TEMPLATE, title=title)

    @app.route("/data")
    def get_data():
        return jsonify(data)

    if port is None:
        port = _find_free_port(5000)

    url = f"http://127.0.0.1:{port}"
    print(f"\n  viz3d → сервер запущен: {url}")
    print(f"  Режим:  {mode}")
    print(f"  Ctrl+C для остановки\n")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
