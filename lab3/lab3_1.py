import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4th_order(f, x0, y0, y_prime0, x_end, h):
    """
    Метод Рунге-Кутты 4‑го порядка для ДУ 2‑го порядка.
    f — функция, задающая ДУ в виде y'' = f(x, y, y')
    """
    x = np.arange(x0, x_end + h, h)
    n = len(x)
    y = np.zeros(n)
    y_prime = np.zeros(n)

    y[0] = y0
    y_prime[0] = y_prime0

    for i in range(n - 1):
        k1_y = h * y_prime[i]
        k1_yp = h * f(x[i], y[i], y_prime[i])

        k2_y = h * (y_prime[i] + 0.5 * k1_yp)
        k2_yp = h * f(x[i] + 0.5 * h, y[i] + 0.5 * k1_y, y_prime[i] + 0.5 * k1_yp)

        k3_y = h * (y_prime[i] + 0.5 * k2_yp)
        k3_yp = h * f(x[i] + 0.5 * h, y[i] + 0.5 * k2_y, y_prime[i] + 0.5 * k2_yp)

        k4_y = h * (y_prime[i] + k3_yp)
        k4_yp = h * f(x[i] + h, y[i] + k3_y, y_prime[i] + k3_yp)

        y[i + 1] = y[i] + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        y_prime[i + 1] = y_prime[i] + (k1_yp + 2 * k2_yp + 2 * k3_yp + k4_yp) / 6

    return x, y


# Задача 1: ДУ 2-го порядка
def f1(x, y, y_prime):
    return 2 * y / (x ** 2)

x0 = 1
y0 = 5 / 6
y_prime0 = 2 / 3
x_end = 2
h_values = [0.1, 0.01, 0.001]
results = {}

for h in h_values:
    x, y = runge_kutta_4th_order(f1, x0, y0, y_prime0, x_end, h)
    results[h] = (x, y)

# Визуализация решений для разных шагов (3 отдельных подграфика)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Решение ДУ $x^2 y\'\' - 2y = 0$ методом Рунге-Кутты для разных шагов', fontsize=14)

for idx, h in enumerate(h_values):
    x, y = results[h]
    axes[idx].plot(x, y, 'b-', linewidth=2)
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('y(x)')
    axes[idx].set_title(f'Шаг h = {h}')
    axes[idx].grid(True)

plt.tight_layout()
plt.show()

# Сравнение скорости сходимости
plt.figure(figsize=(12, 8))
reference_h = 0.001
x_ref, y_ref = results[reference_h]

for h, (x, y) in results.items():
    if h != reference_h:
        # Интерполяция для сравнения
        y_interp = np.interp(x_ref, x, y)
        error = np.abs(y_interp - y_ref)
        plt.plot(x_ref, error, label=f'Ошибка h = {h} относительно h = {reference_h}')

plt.xlabel('x')
plt.ylabel('Абсолютная ошибка')
plt.title('Сравнение скорости сходимости метода')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

print("\nРешение дифференциального уравнения методом Рунге-Кутты:")

# Вывод только финальных решений для задачи 1
for h, (x, y) in results.items():
    final_x = x[-1]
    final_y = y[-1]
    print(f"Шаг h = {h}: y({final_x:.3f}) = {final_y:.6f}")

print("\nРешение через встроенную функцию scipy.integrate.solve_ivp")
from scipy.integrate import solve_ivp

# Преобразуем ДУ 2‑го порядка в систему ДУ 1‑го порядка
# Исходное уравнение: y'' = 2y / x²
# Вводим замену: z = y', тогда z' = y'' = 2y / x²
def system_ode(x, state):
    y, z = state  # y — значение функции, z — её производная
    dydx = z
    dzdx = 2 * y / (x ** 2)
    return [dydx, dzdx]

# Начальные условия
initial_state = [y0, y_prime0]  # [y(x0), y'(x0)]

# Решаем с помощью solve_ivp (метод RK45 — адаптивный шаг)
solution_scipy = solve_ivp(
    system_ode,
    [x0, x_end],
    initial_state,
    method='RK45',
    dense_output=True,  # Позволяет получить решение в любых точках
    rtol=1e-8,      # Относительная точность
    atol=1e-10       # Абсолютная точность
)

# Получаем решение на той же сетке, что и в методе Рунге‑Кутты
x_scipy = np.arange(x0, x_end + 0.001, 0.001)  # Используем мелкий шаг для сравнения
y_scipy, y_prime_scipy = solution_scipy.sol(x_scipy)


# Вывод финальных значений для сравнения
for h, (x, y) in results.items():
    final_x = x[-1]
    final_y = y[-1]
    print(f"Шаг h = {h}: y({final_x:.3f}) = {final_y:.6f}")
