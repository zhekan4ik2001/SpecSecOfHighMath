import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def F(x, y):
    f1 = 2 * y - np.cos(x + 1)
    f2 = x + np.sin(y) + 0.4
    return [f1, f2]

def J(x, y):
    return [
        [np.sin(x + 1), 2],
        [1, np.cos(y)]
    ]

def vector_norm(vector):
    return np.sqrt(sum(x ** 2 for x in vector))

def solve_2x2_system(A, b):
    a11, a12 = A[0]
    a21, a22 = A[1]
    b1, b2 = b
    det = a11 * a22 - a12 * a21
    x1 = (b1 * a22 - b2 * a12) / det
    x2 = (a11 * b2 - a21 * b1) / det
    return [x1, x2]

def newton_system(x0, y0, eps=0.0001, max_iter=100):
    x, y = x0, y0

    for iteration in range(max_iter):
        # Вычисляем вектор функций
        F_val = F(x, y)

        # Проверяем условие остановки: норма вектора функций меньше eps
        norm_F = vector_norm(F_val)
        if norm_F < eps:
            break

        # Вычисляем матрицу Якоби
        J_val = J(x, y)

        # Формируем вектор -F
        minus_F = [-f for f in F_val]

        # Решаем систему линейных уравнений J * delta = -F
        delta = solve_2x2_system(J_val, minus_F)

        # Обновляем решение
        x += delta[0]
        y += delta[1]

    return x, y, iteration + 1

# Начальное приближение (подбрано эмпирически)
x0, y0 = 0.0, 0.0

# Решаем систему методом Ньютона
solution_x, solution_y, iterations = newton_system(x0, y0)

# Выводим результаты
print("\nРешение системы уравнений преобразованным итерационным методом:")
if(iterations >= 100):
    print("Корни не найдены")
else:
    print("Корни найдены")
    print(f"Решение найдено за {iterations} итераций:")
    print(f"x = {solution_x:.6f}")
    print(f"y = {solution_y:.6f}")

def system_for_scipy(vars):
    x, y = vars
    f1 = 2 * y - np.cos(x + 1)
    f2 = x + np.sin(y) + 0.4
    return [f1, f2]


scipy_solution = fsolve(system_for_scipy, [x0, y0])
scipy_x, scipy_y = scipy_solution

print("Корени найденные с помощью функции scipy.optimize.fsolve:")
print(f"x = {scipy_x:.6f}")
print(f"y = {scipy_y:.6f}")

x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_range, y_range)

F1 = 2 * Y - np.cos(X + 1)
F2 = X + np.sin(Y) + 0.4

plt.figure(figsize=(12, 8))


# Рисуем контуры без label
plt.contour(X, Y, F1, colors="lightblue", levels=[0], linewidths=2)
plt.contour(X, Y, F2, colors="orange", levels=[0], linewidths=2)

# Создаём фиктивные линии для легенды
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='lightblue', linewidth=2, label='2y - cos(x+1) = 0'),
    Line2D([0], [0], color='orange', linewidth=2, label='x + sin(y) + 0.4 = 0'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Найденные корни')
]

plt.scatter([solution_x, scipy_x], [solution_y, scipy_y], color='red', zorder=5)

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.legend(handles=legend_elements)

plt.show()