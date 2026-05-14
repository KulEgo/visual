"""
viz3d — главный модуль.
 
Веб-интерфейс с двумя режимами:
  • 2D: y = f(x) — обычная кривая на плоскости
  • 3D: z = f(x, y) или параметрическая поверхность (x, y, z) = f(u, v)
 
Снизу под графиком выводится автоматически определённый тип функции
(линейная, парабола, синусоида, параболоид, седло, сфера, тор и т. д.).
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
 
 
def _safe_compile(expr: str, allowed_vars: set):
    """Безопасно компилирует выражение. allowed_vars — какие имена-переменные разрешены."""
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
            allowed = set(_SAFE_NAMES.keys()) | allowed_vars
            if node.id not in allowed:
                raise ValueError(f"Неизвестное имя: {node.id}")
 
    return tree, compile(tree, "<expr>", "eval")
 
 
def _evaluate(expr: str, var_names: tuple, var_values: tuple):
    """Безопасно вычисляет выражение."""
    allowed_vars = set(var_names)
    _, code = _safe_compile(expr, allowed_vars)
    namespace = dict(_SAFE_NAMES)
    for name, value in zip(var_names, var_values):
        namespace[name] = value
    with np.errstate(all="ignore"):
        return eval(code, {"__builtins__": {}}, namespace)
 
 
def _has_any_valid_point(arrays):
    """True если есть точка, где ВСЕ массивы конечны одновременно."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return bool(np.any(mask))
 
 
# === Распознавание типа функции (символический анализ AST) ===
 
def _used_functions(tree: ast.AST) -> set:
    """Множество имён функций, использованных в выражении (sin, cos, log, ...)."""
    funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            funcs.add(node.func.id)
    return funcs
 
 
def _total_polynomial_degree(tree: ast.AST, variables: tuple) -> Optional[int]:
    """
    Полная степень многочлена от заданных переменных (по сумме показателей в мономе).
    None — если не многочлен. Например: x*y → 2, x**2 + y → 2, x + y → 1.
    """
    def deg(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return 0
        if isinstance(node, ast.Name):
            if node.id in variables:
                return 1
            if node.id in _SAFE_NAMES or node.id in {"x", "y", "u", "v"}:
                return 0
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return deg(node.operand)
        if isinstance(node, ast.BinOp):
            l, r = deg(node.left), deg(node.right)
            if l is None or r is None:
                return None
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return max(l, r)
            if isinstance(node.op, ast.Mult):
                return l + r
            if isinstance(node.op, ast.Div):
                if r == 0:
                    return l
                return None
            if isinstance(node.op, ast.Pow):
                if r != 0:
                    return None
                exp_val = _const_value(node.right)
                if exp_val is None or exp_val < 0 or exp_val != int(exp_val):
                    return None
                return l * int(exp_val)
            return None
        if isinstance(node, ast.Call):
            return None
        return None
 
    return deg(tree.body if isinstance(tree, ast.Expression) else tree)
 
 
def _polynomial_degree(tree: ast.AST, var: str) -> Optional[int]:
    """
    Если выражение — многочлен от переменной var, возвращает его степень.
    Если не многочлен — None. Степень 0 = константа (от var не зависит).
    """
    def deg(node):
        # Число / константа → степень 0
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return 0
        # Имя переменной
        if isinstance(node, ast.Name):
            if node.id == var:
                return 1
            # Любое другое известное имя (другая переменная или константа pi/e/tau)
            # относительно var является константой → степень 0
            if node.id in _SAFE_NAMES or node.id in {"x", "y", "u", "v"}:
                return 0
            return None
        # Унарные + и - не меняют степень
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return deg(node.operand)
        # Бинарные операции
        if isinstance(node, ast.BinOp):
            l = deg(node.left)
            r = deg(node.right)
            if l is None or r is None:
                return None
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return max(l, r)
            if isinstance(node.op, ast.Mult):
                return l + r
            if isinstance(node.op, ast.Div):
                # Деление полиномиально только если знаменатель — константа
                if r == 0:
                    return l
                return None
            if isinstance(node.op, ast.Pow):
                # Степень полиномиальна, если показатель — целая неотрицательная константа
                if r != 0:
                    return None
                # Извлекаем числовое значение показателя
                exp_val = _const_value(node.right)
                if exp_val is None or exp_val < 0 or exp_val != int(exp_val):
                    return None
                return l * int(exp_val)
            return None
        # Вызовы функций — точно не полином
        if isinstance(node, ast.Call):
            return None
        return None
 
    return deg(tree.body if isinstance(tree, ast.Expression) else tree)
 
 
def _const_value(node):
    """Пробует извлечь числовое значение из узла-константы (включая -число)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        v = _const_value(node.operand)
        if v is None:
            return None
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
    return None
 
 
def _classify_2d(expr: str) -> str:
    """Определяет тип функции y = f(x) и возвращает описание."""
    try:
        tree, _ = _safe_compile(expr, {"x"})
    except ValueError:
        return "Функция одной переменной"
 
    funcs = _used_functions(tree)
    degree = _polynomial_degree(tree, "x")
 
    # Если многочлен — называем по степени
    if degree is not None:
        names = {
            0: "Постоянная функция (горизонтальная прямая)",
            1: "Линейная функция (прямая)",
            2: "Квадратичная функция (парабола)",
            3: "Кубическая функция",
            4: "Многочлен 4-й степени",
        }
        if degree in names:
            return names[degree]
        return f"Многочлен {degree}-й степени"
 
    # Не многочлен — определяем по использованным функциям
    trig = funcs & {"sin", "cos", "tan", "asin", "acos", "atan",
                    "sinh", "cosh", "tanh"}
    if trig:
        return f"Тригонометрическая функция ({', '.join(sorted(trig))})"
    if funcs & {"exp"}:
        return "Экспоненциальная функция"
    if funcs & {"log", "log2", "log10"}:
        return "Логарифмическая функция"
    if funcs & {"sqrt", "cbrt"}:
        return "Степенная функция (корень)"
    if funcs & {"abs"}:
        return "Функция с модулем"
 
    return "Функция одной переменной"
 
 
def _classify_3d_surface(expr: str) -> str:
    """Определяет тип поверхности z = f(x, y)."""
    try:
        tree, _ = _safe_compile(expr, {"x", "y"})
    except ValueError:
        return "Поверхность z = f(x, y)"
 
    funcs = _used_functions(tree)
    total_deg = _total_polynomial_degree(tree, ("x", "y"))
 
    # Многочлен от x и y
    if total_deg is not None:
        if total_deg == 0:
            return "Горизонтальная плоскость"
        if total_deg == 1:
            return "Плоскость"
        if total_deg == 2:
            kind = _classify_quadratic_surface(tree)
            if kind:
                return kind
            return "Квадратичная поверхность 2-й степени"
        return f"Поверхность {total_deg}-й степени"
 
    # Тригонометрия
    trig = funcs & {"sin", "cos", "tan", "asin", "acos", "atan",
                    "sinh", "cosh", "tanh"}
    if trig:
        return f"Волнообразная поверхность ({', '.join(sorted(trig))})"
    if funcs & {"exp"}:
        return "Экспоненциальная поверхность"
    if funcs & {"log", "log2", "log10"}:
        return "Логарифмическая поверхность"
    if funcs & {"sqrt", "cbrt"}:
        return "Поверхность с корнем"
 
    return "Поверхность z = f(x, y)"
 
 
def _classify_quadratic_surface(tree: ast.AST) -> Optional[str]:
    """
    Численная проверка по тестовым точкам: что за квадратичная поверхность.
    Параболоид: f(x,y) = a·x² + b·y² + ... с одинаковыми знаками при x², y²
    Седло: разные знаки при x², y²
    Параболический цилиндр: один из коэффициентов = 0
    """
    code = compile(tree, "<expr>", "eval")
 
    def f(x, y):
        ns = dict(_SAFE_NAMES)
        ns["x"] = x
        ns["y"] = y
        try:
            with np.errstate(all="ignore"):
                return float(eval(code, {"__builtins__": {}}, ns))
        except Exception:
            return float("nan")
 
    # Коэффициенты при x², y², xy через конечные разности второго порядка
    # ∂²/∂x² ≈ f(h,0) - 2f(0,0) + f(-h,0)  → 2·a
    h = 0.01
    f00 = f(0, 0)
    a2 = (f(h, 0) - 2 * f00 + f(-h, 0)) / (h * h)   # 2·коэф при x²
    b2 = (f(0, h) - 2 * f00 + f(0, -h)) / (h * h)   # 2·коэф при y²
    xy = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)  # коэф при xy
 
    def near_zero(v):
        return abs(v) < 1e-6
 
    if not all(math.isfinite(v) for v in [a2, b2, xy]):
        return None
 
    a_zero, b_zero, xy_zero = near_zero(a2), near_zero(b2), near_zero(xy)
 
    if a_zero and b_zero and xy_zero:
        return None  # не квадратичная по факту
    if xy_zero:
        if a_zero or b_zero:
            return "Параболический цилиндр"
        if (a2 > 0) == (b2 > 0):
            return "Эллиптический параболоид"
        return "Гиперболический параболоид (седло)"
    # есть смешанный член
    # Детерминант квадратичной формы a·x² + 2c·xy + b·y² → ab - c²
    det = (a2 / 2) * (b2 / 2) - (xy / 2) ** 2
    if det > 0:
        return "Эллиптический параболоид"
    if det < 0:
        return "Гиперболический параболоид (седло)"
    return "Квадратичная поверхность"
 
 
def _classify_3d_parametric(parts: list) -> str:
    """Распознаёт стандартные параметрические поверхности."""
    if len(parts) != 3:
        return "Параметрическая поверхность"
 
    # Нормализуем — убираем пробелы
    norm = [p.replace(" ", "") for p in parts]
 
    # Сфера: (cos(u)*sin(v), sin(u)*sin(v), cos(v)) с возможными радиусами
    if (all("sin" in p or "cos" in p for p in norm)
            and "cos(u)" in norm[0] and "sin(v)" in norm[0]
            and "sin(u)" in norm[1] and "sin(v)" in norm[1]
            and "cos(v)" in norm[2]):
        return "Сфера"
 
    # Тор: первые две координаты — (R + r·cos(v))·cos(u) и ·sin(u), третья — sin(v)
    has_torus_pattern = (
        ("cos(v)" in norm[0] and "cos(u)" in norm[0])
        and ("cos(v)" in norm[1] and "sin(u)" in norm[1])
        and ("sin(v)" in norm[2])
    )
    if has_torus_pattern:
        return "Тор"
 
    # Цилиндр: x = r·cos(u), y = r·sin(u), z = v (или линейная по v)
    if ("cos(u)" in norm[0] and "sin(u)" in norm[1]
            and "u" not in norm[2] and "sin" not in norm[2] and "cos" not in norm[2]):
        return "Цилиндр"
 
    # Геликоид: u·cos(v), u·sin(v), v
    if ("cos(v)" in norm[0] and "sin(v)" in norm[1]
            and norm[2] in ("v", "-v")):
        return "Геликоид"
 
    # Конус: u·cos(v), u·sin(v), u
    if ("cos" in norm[0] and "sin" in norm[1]
            and norm[2] in ("u", "v", "-u", "-v")):
        return "Конус"
 
    return "Параметрическая поверхность"
 
 
# === Подготовка данных ===
 
def _compute_2d(expr: str, x_min, x_max, resolution):
    """y = f(x) на отрезке [x_min, x_max]."""
    x = np.linspace(x_min, x_max, resolution)
    y = _evaluate(expr, ("x",), (x,))
    y = np.broadcast_to(np.asarray(y, dtype=float), x.shape).astype(float).copy()
 
    if not _has_any_valid_point([y]):
        return {"empty": True, "description": _classify_2d(expr)}
 
    y[~np.isfinite(y)] = np.nan
    return {
        "empty": False,
        "mode": "2d",
        "traces": [{
            "type": "scatter",
            "mode": "lines",
            "x": x.tolist(),
            "y": [None if math.isnan(v) else v for v in y],
            "line": {"color": "#4a90e2", "width": 3},
        }],
        "description": _classify_2d(expr),
    }
 
 
def _compute_3d(expr: str, x_min, x_max, y_min, y_max, resolution):
    """3D-режим: либо z = f(x, y), либо параметрика."""
    parts = [p.strip() for p in expr.split(";") if p.strip()]
 
    if len(parts) == 1:
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        Z = _evaluate(parts[0], ("x", "y"), (X, Y))
        Z = np.broadcast_to(np.asarray(Z, dtype=float), X.shape).copy()
 
        if not _has_any_valid_point([Z]):
            return {"empty": True, "description": _classify_3d_surface(parts[0])}
 
        Z[~np.isfinite(Z)] = np.nan
        return {
            "empty": False,
            "mode": "3d",
            "traces": [{
                "type": "surface",
                "x": X.tolist(), "y": Y.tolist(),
                "z": [[None if math.isnan(v) else v for v in row] for row in Z],
                "colorscale": "Viridis", "showscale": True,
            }],
            "description": _classify_3d_surface(parts[0]),
        }
 
    if len(parts) == 3:
        u = np.linspace(x_min, x_max, resolution)
        v = np.linspace(y_min, y_max, resolution)
        U, V = np.meshgrid(u, v)
        X = np.broadcast_to(np.asarray(_evaluate(parts[0], ("u", "v"), (U, V)), dtype=float), U.shape).copy()
        Y = np.broadcast_to(np.asarray(_evaluate(parts[1], ("u", "v"), (U, V)), dtype=float), U.shape).copy()
        Z = np.broadcast_to(np.asarray(_evaluate(parts[2], ("u", "v"), (U, V)), dtype=float), U.shape).copy()
 
        if not _has_any_valid_point([X, Y, Z]):
            return {"empty": True, "description": _classify_3d_parametric(parts)}
 
        for arr in (X, Y, Z):
            arr[~np.isfinite(arr)] = np.nan
 
        return {
            "empty": False,
            "mode": "3d",
            "traces": [{
                "type": "surface",
                "x": [[None if math.isnan(v) else v for v in row] for row in X],
                "y": [[None if math.isnan(v) else v for v in row] for row in Y],
                "z": [[None if math.isnan(v) else v for v in row] for row in Z],
                "colorscale": "Plasma", "showscale": True,
            }],
            "description": _classify_3d_parametric(parts),
        }
 
    raise ValueError(
        "В 3D-режиме: 1 выражение (z = f(x, y)) или 3 через ';' "
        "((x, y, z) = f(u, v))"
    )
 
 
# === HTML ===
 
_HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>viz3d — визуализация</title>
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
            margin-right: 4px; color: #e6e6e6; white-space: nowrap;
            padding-bottom: 8px;
        }
        /* Переключатель 2D / 3D */
        .mode-switch {
            display: flex;
            background: #0f1419;
            border: 1px solid #2a3142;
            border-radius: 6px;
            overflow: hidden;
            align-self: flex-end;
        }
        .mode-switch button {
            background: transparent; border: none;
            color: #8b95a7; padding: 8px 16px;
            font-size: 13px; font-weight: 600; cursor: pointer;
            transition: background 0.15s, color 0.15s;
        }
        .mode-switch button.active {
            background: #4a90e2; color: white;
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
        input.expr { min-width: 280px; flex-grow: 1; }
        input.range-input { width: 65px; }
        .range-group { display: flex; align-items: center; gap: 4px; }
        .range-group span { color: #5a6478; font-size: 12px; }
        button.action {
            background: #4a90e2; color: white; border: none;
            padding: 9px 18px; border-radius: 5px;
            font-size: 13px; font-weight: 500; cursor: pointer;
            transition: background 0.15s;
        }
        button.action:hover { background: #5aa0f2; }
        button.action:disabled { background: #3a4356; cursor: not-allowed; }
        .examples {
            padding: 8px 20px; background: #151a26;
            border-bottom: 1px solid #2a3142;
            font-size: 12px; color: #8b95a7;
            display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
            min-height: 38px;
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
        /* Плашка описания внизу графика (тип функции) */
        .description {
            background: #1a1f2e;
            border-top: 1px solid #2a3142;
            padding: 12px 20px;
            font-size: 14px;
            color: #e6e6e6;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .description .badge {
            background: #4a90e2;
            color: white;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        /* Скрываем 3D-поля в 2D-режиме */
        body.mode-2d .only-3d { display: none; }
        body.mode-2d .label-x { content: "Диапазон x"; }
        /* Плашка «такого графика не существует» */
        .no-graph {
            position: absolute;
            left: 50%; bottom: 24px;
            transform: translateX(-50%);
            background: rgba(74, 29, 29, 0.92);
            color: #ffb0b0;
            padding: 12px 24px;
            border-radius: 8px;
            border: 1px solid #6a2d2d;
            font-size: 14px; font-weight: 500;
            display: none; pointer-events: none;
            backdrop-filter: blur(4px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        }
        .no-graph.visible { display: block; }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">viz3d</h1>
        <div class="mode-switch">
            <button id="btn-2d" onclick="switchMode('2d')">2D</button>
            <button id="btn-3d" class="active" onclick="switchMode('3d')">3D</button>
        </div>
        <div class="field" style="flex-grow: 1;">
            <label id="expr-label">Ваша функция</label>
            <input id="expr" class="expr" type="text" value="sin(sqrt(x**2 + y**2))">
        </div>
        <div class="field">
            <label id="range1-label">Диапазон 1</label>
            <div class="range-group">
                <input id="xmin" class="range-input" type="number" value="-5" step="0.5">
                <span>—</span>
                <input id="xmax" class="range-input" type="number" value="5" step="0.5">
            </div>
        </div>
        <div class="field only-3d">
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
        <button class="action" id="btn" onclick="render()">Построить</button>
    </div>
    <div id="examples" class="examples"></div>
    <div id="error" class="error"></div>
    <div class="plot-wrap">
        <div id="plot"></div>
        <div id="no-graph" class="no-graph">⚠ Такого графика не существует</div>
    </div>
    <div id="description" class="description" style="display:none">
        <span class="badge" id="desc-badge">Тип</span>
        <span id="desc-text"></span>
    </div>
 
    <script>
        let mode = '3d';
 
        // === Примеры для каждого режима ===
        const EXAMPLES_2D = [
            { label: 'константа', expr: '3' },
            { label: 'линейная', expr: '2*x + 1' },
            { label: 'парабола', expr: 'x**2 - 4' },
            { label: 'кубическая', expr: 'x**3 - 3*x' },
            { label: 'синусоида', expr: 'sin(x)' },
            { label: 'экспонента', expr: 'exp(x)' },
            { label: 'логарифм', expr: 'log(x)' },
            { label: 'модуль', expr: 'abs(x) - 2' },
        ];
        const EXAMPLES_3D = [
            { label: 'параболоид', expr: 'x**2 + y**2' },
            { label: 'седло', expr: 'x**2 - y**2' },
            { label: 'плоскость', expr: '2*x + 3*y + 1' },
            { label: 'волны', expr: 'sin(sqrt(x**2 + y**2))' },
            { label: 'гауссиана', expr: 'exp(-(x**2+y**2)/4)' },
            { label: 'сфера', expr: 'cos(u)*sin(v); sin(u)*sin(v); cos(v)', xmin: 0, xmax: 6.28, ymin: 0, ymax: 3.14 },
            { label: 'тор', expr: '(2+cos(v))*cos(u); (2+cos(v))*sin(u); sin(v)', xmin: 0, xmax: 6.28, ymin: 0, ymax: 6.28 },
            { label: 'log(x)<0 — нет графика', expr: 'log(x) + log(y)', xmin: -2, xmax: -1, ymin: -2, ymax: -1 },
        ];
 
        function renderExamples() {
            const list = mode === '2d' ? EXAMPLES_2D : EXAMPLES_3D;
            const box = document.getElementById('examples');
            box.innerHTML = '<span class="label">Примеры:</span>';
            for (const ex of list) {
                const chip = document.createElement('span');
                chip.className = 'chip';
                chip.textContent = ex.label;
                chip.onclick = () => applyExample(ex);
                box.appendChild(chip);
            }
        }
 
        function applyExample(ex) {
            document.getElementById('expr').value = ex.expr;
            if (ex.xmin !== undefined) {
                document.getElementById('xmin').value = ex.xmin;
                document.getElementById('xmax').value = ex.xmax;
            }
            if (ex.ymin !== undefined) {
                document.getElementById('ymin').value = ex.ymin;
                document.getElementById('ymax').value = ex.ymax;
            }
            render();
        }
 
        function switchMode(newMode) {
            mode = newMode;
            document.getElementById('btn-2d').classList.toggle('active', mode === '2d');
            document.getElementById('btn-3d').classList.toggle('active', mode === '3d');
            document.body.classList.toggle('mode-2d', mode === '2d');
 
            if (mode === '2d') {
                document.getElementById('expr-label').textContent = 'Ваша функция y = f(x)';
                document.getElementById('range1-label').textContent = 'Диапазон x';
                document.getElementById('expr').placeholder = 'например: sin(x), x**2 - 4';
                // дефолтное выражение для 2D
                document.getElementById('expr').value = 'sin(x)';
            } else {
                document.getElementById('expr-label').textContent = 'Ваша функция';
                document.getElementById('range1-label').textContent = 'Диапазон 1';
                document.getElementById('expr').placeholder = 'например: x**2 + y**2  или  cos(u)*sin(v); sin(u)*sin(v); cos(v)';
                document.getElementById('expr').value = 'sin(sqrt(x**2 + y**2))';
            }
            renderExamples();
            render();
        }
 
        // === Стрелочные оси для 3D ===
        function buildArrowAxes3D(bounds) {
            const pad = 0.15;
            const ext = (lo, hi) => {
                const span = (hi - lo) || 1;
                return [lo - span * pad, hi + span * pad];
            };
            const [xLo, xHi] = ext(bounds.xmin, bounds.xmax);
            const [yLo, yHi] = ext(bounds.ymin, bounds.ymax);
            const [zLo, zHi] = ext(bounds.zmin, bounds.zmax);
 
            const axisLines = [
                { type: 'scatter3d', mode: 'lines',
                  x: [xLo, xHi], y: [0, 0], z: [0, 0],
                  line: { color: '#ff6b6b', width: 4 },
                  hoverinfo: 'skip', showlegend: false },
                { type: 'scatter3d', mode: 'lines',
                  x: [0, 0], y: [yLo, yHi], z: [0, 0],
                  line: { color: '#51cf66', width: 4 },
                  hoverinfo: 'skip', showlegend: false },
                { type: 'scatter3d', mode: 'lines',
                  x: [0, 0], y: [0, 0], z: [zLo, zHi],
                  line: { color: '#4dabf7', width: 4 },
                  hoverinfo: 'skip', showlegend: false },
            ];
 
            const labels = {
                type: 'scatter3d', mode: 'text',
                x: [xHi, 0, 0], y: [0, yHi, 0], z: [0, 0, zHi],
                text: ['<b>x</b>', '<b>y</b>', '<b>z</b>'],
                textfont: { size: 18, color: '#e6e6e6' },
                textposition: 'top center',
                hoverinfo: 'skip', showlegend: false,
            };
 
            const coneSize = Math.max(xHi - xLo, yHi - yLo, zHi - zLo) * 0.04;
            const arrows = {
                type: 'cone',
                x: [xHi, 0, 0], y: [0, yHi, 0], z: [0, 0, zHi],
                u: [1, 0, 0], v: [0, 1, 0], w: [0, 0, 1],
                sizemode: 'absolute', sizeref: coneSize,
                anchor: 'tip',
                colorscale: [[0, '#888'], [1, '#888']],
                showscale: false, hoverinfo: 'skip',
                lighting: { ambient: 0.8 },
            };
 
            return [...axisLines, labels, arrows];
        }
 
        function computeBounds3D(traces) {
            let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity,zmin=Infinity,zmax=-Infinity;
            for (const t of traces) {
                for (const row of t.x) for (const v of row)
                    if (v !== null && isFinite(v)) { xmin=Math.min(xmin,v); xmax=Math.max(xmax,v); }
                for (const row of t.y) for (const v of row)
                    if (v !== null && isFinite(v)) { ymin=Math.min(ymin,v); ymax=Math.max(ymax,v); }
                for (const row of t.z) for (const v of row)
                    if (v !== null && isFinite(v)) { zmin=Math.min(zmin,v); zmax=Math.max(zmax,v); }
            }
            if (!isFinite(xmin)) { xmin=-1; xmax=1; }
            if (!isFinite(ymin)) { ymin=-1; ymax=1; }
            if (!isFinite(zmin)) { zmin=-1; zmax=1; }
            xmin=Math.min(xmin,0); xmax=Math.max(xmax,0);
            ymin=Math.min(ymin,0); ymax=Math.max(ymax,0);
            zmin=Math.min(zmin,0); zmax=Math.max(zmax,0);
            return { xmin, xmax, ymin, ymax, zmin, zmax };
        }
 
        function drawPlot3D(surfaceTraces, bounds) {
            const axisTraces = buildArrowAxes3D(bounds);
            Plotly.newPlot('plot', [...surfaceTraces, ...axisTraces], {
                paper_bgcolor: '#0f1419', plot_bgcolor: '#0f1419',
                font: { color: '#e6e6e6' },
                margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
                scene: {
                    xaxis: { visible: false, range: [bounds.xmin*1.2, bounds.xmax*1.2] },
                    yaxis: { visible: false, range: [bounds.ymin*1.2, bounds.ymax*1.2] },
                    zaxis: { visible: false, range: [bounds.zmin*1.2, bounds.zmax*1.2] },
                    aspectmode: 'cube', bgcolor: '#0f1419',
                    camera: { eye: { x: 1.7, y: 1.7, z: 1.3 } }
                }
            }, { responsive: true, displaylogo: false });
        }
 
        function drawPlot2D(traces, xRange, yRange) {
            // Берём фактические границы данных, расширяем чтобы включить 0
            let xmin = xRange[0], xmax = xRange[1];
            let ymin = Infinity, ymax = -Infinity;
            for (const t of traces) {
                for (const v of t.y) {
                    if (v !== null && isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
                }
            }
            if (!isFinite(ymin)) { ymin = -1; ymax = 1; }
            // включаем 0 в видимую область
            xmin = Math.min(xmin, 0); xmax = Math.max(xmax, 0);
            ymin = Math.min(ymin, 0); ymax = Math.max(ymax, 0);
            const xPad = (xmax - xmin) * 0.1 || 1;
            const yPad = (ymax - ymin) * 0.1 || 1;
            const xLo = xmin - xPad, xHi = xmax + xPad;
            const yLo = ymin - yPad, yHi = ymax + yPad;
 
            // Линии осей через начало координат
            const axisX = {
                type: 'scatter', mode: 'lines',
                x: [xLo, xHi], y: [0, 0],
                line: { color: '#ff6b6b', width: 2 },
                hoverinfo: 'skip', showlegend: false,
            };
            const axisY = {
                type: 'scatter', mode: 'lines',
                x: [0, 0], y: [yLo, yHi],
                line: { color: '#51cf66', width: 2 },
                hoverinfo: 'skip', showlegend: false,
            };
 
            // Стрелки на концах осей (через annotations)
            const arrows = [
                {  // стрелка X (вправо)
                    x: xHi, y: 0, ax: xHi - (xHi - xLo) * 0.04, ay: 0,
                    xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                    showarrow: true, arrowhead: 3, arrowsize: 1.5, arrowwidth: 2,
                    arrowcolor: '#ff6b6b', text: '',
                },
                {  // стрелка Y (вверх)
                    x: 0, y: yHi, ax: 0, ay: yHi - (yHi - yLo) * 0.04,
                    xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                    showarrow: true, arrowhead: 3, arrowsize: 1.5, arrowwidth: 2,
                    arrowcolor: '#51cf66', text: '',
                },
                // подписи x, y
                { x: xHi, y: 0, xref: 'x', yref: 'y', text: '<b>x</b>',
                  showarrow: false, xanchor: 'left', yanchor: 'middle',
                  font: { color: '#e6e6e6', size: 16 }, xshift: 8 },
                { x: 0, y: yHi, xref: 'x', yref: 'y', text: '<b>y</b>',
                  showarrow: false, xanchor: 'middle', yanchor: 'bottom',
                  font: { color: '#e6e6e6', size: 16 }, yshift: 8 },
            ];
 
            Plotly.newPlot('plot', [...traces, axisX, axisY], {
                paper_bgcolor: '#0f1419', plot_bgcolor: '#0f1419',
                font: { color: '#e6e6e6' },
                margin: { l: 40, r: 40, t: 20, b: 40 },
                showlegend: false,
                xaxis: {
                    range: [xLo, xHi], gridcolor: '#2a3142',
                    zeroline: false, showline: false,
                    color: '#8b95a7',
                },
                yaxis: {
                    range: [yLo, yHi], gridcolor: '#2a3142',
                    zeroline: false, showline: false,
                    color: '#8b95a7',
                },
                annotations: arrows,
            }, { responsive: true, displaylogo: false });
        }
 
        function showError(msg) {
            const el = document.getElementById('error');
            el.textContent = '⚠ ' + msg;
            el.classList.add('visible');
        }
        function hideError() { document.getElementById('error').classList.remove('visible'); }
        function showNoGraph() { document.getElementById('no-graph').classList.add('visible'); }
        function hideNoGraph() { document.getElementById('no-graph').classList.remove('visible'); }
 
        function showDescription(text, badge) {
            const block = document.getElementById('description');
            document.getElementById('desc-badge').textContent = badge;
            document.getElementById('desc-text').textContent = text;
            block.style.display = 'flex';
        }
        function hideDescription() {
            document.getElementById('description').style.display = 'none';
        }
 
        async function render() {
            const btn = document.getElementById('btn');
            btn.disabled = true; btn.textContent = '...';
            hideError(); hideNoGraph();
 
            const xmin = parseFloat(document.getElementById('xmin').value);
            const xmax = parseFloat(document.getElementById('xmax').value);
            const ymin = parseFloat(document.getElementById('ymin').value);
            const ymax = parseFloat(document.getElementById('ymax').value);
            const res = parseInt(document.getElementById('res').value);
            const expr = document.getElementById('expr').value;
 
            const payload = mode === '2d'
                ? { mode, expr, x_min: xmin, x_max: xmax, resolution: res }
                : { mode, expr, x_min: xmin, x_max: xmax, y_min: ymin, y_max: ymax, resolution: res };
 
            try {
                const resp = await fetch('/data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();
 
                if (data.error) {
                    showError(data.error);
                    if (mode === '3d') drawPlot3D([], { xmin: -1, xmax: 1, ymin: -1, ymax: 1, zmin: -1, zmax: 1 });
                    else drawPlot2D([], [xmin, xmax]);
                    hideDescription();
                    return;
                }
 
                if (data.empty) {
                    if (mode === '3d') {
                        drawPlot3D([], {
                            xmin: Math.min(xmin, 0), xmax: Math.max(xmax, 0),
                            ymin: Math.min(ymin, 0), ymax: Math.max(ymax, 0),
                            zmin: -1, zmax: 1
                        });
                    } else {
                        drawPlot2D([], [xmin, xmax]);
                    }
                    showNoGraph();
                    if (data.description) showDescription(data.description, mode === '2d' ? '2D' : '3D');
                    return;
                }
 
                if (mode === '3d') {
                    const bounds = computeBounds3D(data.traces);
                    drawPlot3D(data.traces, bounds);
                } else {
                    drawPlot2D(data.traces, [xmin, xmax]);
                }
                if (data.description) showDescription(data.description, mode === '2d' ? '2D' : '3D');
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
 
        // Старт
        renderExamples();
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
    Запускает локальный сервер с веб-интерфейсом для построения графиков.
 
    На странице есть переключатель 2D / 3D:
      • 2D: y = f(x)         — например 'sin(x)', 'x**2 - 4'
      • 3D: z = f(x, y)      — например 'x**2 + y**2'
        или (x, y, z) = f(u, v) через ';' — например
        'cos(u)*sin(v); sin(u)*sin(v); cos(v)'  (сфера)
 
    Снизу под графиком автоматически выводится тип функции (линейная,
    парабола, параболоид, седло, сфера, тор и т. д.).
 
    Доступные функции: sin, cos, tan, asin, acos, atan, atan2,
    sinh, cosh, tanh, exp, log, log2, log10, sqrt, cbrt, abs, sign,
    floor, ceil, round, min, max, pow.
    Константы: pi, e, tau.
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
            req_mode = params.get("mode", "3d")
            if req_mode == "2d":
                data = _compute_2d(
                    params["expr"],
                    float(params["x_min"]), float(params["x_max"]),
                    int(params["resolution"]),
                )
            else:
                data = _compute_3d(
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
    print(f"  Откройте браузер, выберите режим 2D/3D и введите функцию")
    print(f"  Ctrl+C для остановки\n")
 
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
 
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
