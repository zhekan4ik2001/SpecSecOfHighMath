import numpy as np
import matplotlib.pyplot as plt

def adaptive_runge_kutta(f, x0, y0, x_end, tol=1e-6, h_initial=0.1, max_steps=10000):
    """
    Адаптивный метод Рунге-Кутты (метод Фельберга) с индикацией прогресса
    """
    x_values = [x0]
    y_values = [y0]
    h_values = []
    current_x = x0
    current_y = y0
    h = h_initial
    step_count = 0  # счётчик шагов
    last_progress = 0  # последний выведенный процент прогресса

    while current_x < x_end and step_count < max_steps:
        step_count += 1

        # Индикация прогресса каждые 10%
        progress = (current_x - x0) / (x_end - x0) * 100
        if progress - last_progress >= 10:
            print(f"Прогресс: {progress:.1f}% (x = {current_x:.4f}, шаг = {h:.6f})")
            last_progress = progress

        if current_x + h > x_end:
            h = x_end - current_x

        # Расчёт с шагом h (4‑й порядок)
        k1 = h * f(current_x, current_y)
        k2 = h * f(current_x + h/4, current_y + k1/4)
        k3 = h * f(current_x + 3*h/8, current_y + 3*k1/32 + 9*k2/32)
        k4 = h * f(current_x + 12*h/13, current_y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
        k5 = h * f(current_x + h, current_y + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
        k6 = h * f(current_x + h/2, current_y - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)

        y4 = current_y + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
        y5 = current_y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55

        error = abs(y5 - y4)

        if error < tol:
            current_x += h
            current_y = y4
            x_values.append(current_x)
            y_values.append(current_y)
            h_values.append(h)

            # Адаптация шага
            if error == 0:
                h *= 2
            else:
                h_new = 0.9 * h * (tol / error) ** (1/5)
                h = min(h_new, 2 * h)  # Ограничение роста шага
        else:
            # Уменьшение шага
            h *= 0.5

        # Защита от слишком малого шага
        if h < 1e-12:
            print(f"\nПРЕДУПРЕЖДЕНИЕ: шаг стал слишком малым (h = {h:.2e}) на x = {current_x:.4f}")
            print("Прерывание расчёта для предотвращения бесконечного цикла.")
            break

    if step_count >= max_steps:
        print(f"\nПРЕДУПРЕЖДЕНИЕ: достигнуто максимальное число шагов ({max_steps}). Расчёт прерван.")

    return np.array(x_values), np.array(y_values), np.array(h_values)

# Задача 2: ДУ 1-го порядка с адаптивным шагом
def f2(x, y):
    return x + y ** 2

x0_2 = 0
y0_2 = 1
x_end_2 = 1.5  # Ограничиваем интервал, так как решение быстро растёт

# Решаем уравнение с адаптивным шагом
print("Запуск расчёта адаптивного метода Рунге-Кутты...")
x_adapt, y_adapt, h_adapt = adaptive_runge_kutta(f2, x0_2, y0_2, x_end_2, tol=1e-6, h_initial=0.1)

# Визуализация решения для адаптивного метода
plt.figure(figsize=(12, 8))
plt.plot(x_adapt, y_adapt, 'b-', linewidth=2, label='Адаптивный метод Рунге-Кутты')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Решение ДУ $y\' = x + y^2$, $y(0) = 1$ с адаптивным шагом')
plt.legend()
plt.grid(True)
plt.show()

# График зависимости шага от x
plt.figure(figsize=(12, 8))
plt.step(x_adapt[:-1], h_adapt, where='post', linewidth=2)
plt.xlabel('x')
plt.ylabel('Шаг h')
plt.title('Зависимость шага от координаты x (адаптивный метод)')
plt.grid(True)
plt.show()

print("\nРешение дифференциального уравнения методом Рунге-Кутты с адаптивным шагом:")
# Вывод финального решения для задачи 2
final_x_2 = x_adapt[-1]
final_y_2 = y_adapt[-1]
print(f"y({final_x_2:.3f}) = {final_y_2:.6f}")
print(f"\nВсего точек: {len(x_adapt)}")
print(f"Минимальный шаг: {np.min(h_adapt):.8f}")
print(f"Максимальный шаг: {np.max(h_adapt):.6f}")


print("\nРешение через встроенную функцию scipy.integrate.solve_ivp")
    
from scipy.integrate import solve_ivp

def f(x, y):
    return x + y**2

# Начальные условия
x0, y0 = 0, 1
x_end = 1.3  # Ограничиваем интервал из‑за быстрого роста решения

# Решаем численно
solution = solve_ivp(f, [x0, x_end], [y0], method='RK45', rtol=1e-6)

# Вывод результата
print(f"y({solution.t[-1]:.3f}) ≈ {solution.y[0][-1]:.6f}")
