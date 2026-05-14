"""
Примеры использования библиотеки viz3d.
Раскомментируйте нужный пример и запустите: python example.py
"""

import numpy as np
from viz3d import plot


# === Пример 1: Поверхность z = f(x, y) ===
# Параболоид
# plot(lambda x, y: x**2 + y**2)

# Седло
# plot(lambda x, y: x**2 - y**2)

# Волны
plot(
    lambda x, y: np.sin(np.sqrt(x**2 + y**2)),
    x_range=(-10, 10),
    y_range=(-10, 10),
    resolution=100,
)


# === Пример 2: Параметрическая поверхность ===
# Сфера
# plot(
#     lambda u, v: (np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)),
#     x_range=(0, 2 * np.pi),
#     y_range=(0, np.pi),
# )

# Тор
# def torus(u, v, R=2, r=1):
#     return (
#         (R + r * np.cos(v)) * np.cos(u),
#         (R + r * np.cos(v)) * np.sin(u),
#         r * np.sin(v),
#     )
# plot(torus, x_range=(0, 2 * np.pi), y_range=(0, 2 * np.pi))
