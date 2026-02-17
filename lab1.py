import numpy as np 
from scipy import optimize
import matplotlib.pyplot as plt

def equation_1_fx(x):
    return 8 * np.cos(x) - x

def equation_1__(x):
    return 8 * np.cos(x) - x - 6

def equation_2_fx(x):
    return x**3 + x**2 + 9

def equation_1_gx(x):
    return 8 * np.cos(x) - 6

def equation_2_gx(x):
    return - (x**2 + 9) ** (1/3)

def simple_iteration(equation, x0, eps):
    x_n = x0
    x_n1 = equation(x_n)
    iteration = 0
    
    while (abs(x_n1 - x_n) > eps) and (iteration < 1000): 
        x_n = x_n1 
        x_n1 = equation(x_n)
        iteration += 1
        
    return x_n1, iteration


x0_1 = -5
eps_equation_1 = 0.0001
x0_2 = -3
eps_equation_2 = 0.01

# Нахождение корней
root_1_simple, i_1 = simple_iteration(equation_1_gx, x0_1, eps_equation_1)
root_1_scipy = optimize.root(equation_1__, x0_1).x[0]

root_2_simple, i_2 = simple_iteration(equation_2_gx, x0_2, eps_equation_2)
root_2_scipy = optimize.root(equation_2_fx, x0_2).x[0]

# Вывод результатов
if(i_1 >= 1000):
    print("Корень 1 уравнения не найден, график рассходится")
else:
    print("Корень 1 уравнения найденный самостоятельно: ", root_1_simple)

print("Корень 1 уравнения найденный с помощью функции optimize.root: ", root_1_scipy)

if(i_2 >= 1000):
    print("Корень 2 уравнения не найден, график рассходится")
else:
    print("Корень 2 уравнения найденный самостоятельно: ", root_2_simple)

print("Корень 2 уравнения найденный с помощью функции optimize.root: ", root_2_scipy)

# Построение графиков
x_values_1 = np.linspace(-10, 10, 400)
y_values_1 = equation_1_fx(x_values_1)

x_values_2 = np.linspace(-5, 5, 400)
y_values_2 = equation_2_fx(x_values_2)

# График для equation_1
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_values_1, y_values_1, label='y = f(x)')
plt.axhline(0, color='red', linewidth=0.5, linestyle='--')
plt.axvline(0, color='red', linewidth=0.5, linestyle='--')
plt.scatter([root_1_simple, root_1_scipy], [equation_1_fx(root_1_simple), equation_1_fx(root_1_scipy)], color='red', label='Найденные корни', zorder=5)
plt.title('График функции y = 8 * cos(x) - x')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()

# График для equation_2
plt.subplot(1, 2, 2)
plt.plot(x_values_2, y_values_2, label='y = f(x)')
plt.axhline(0, color='red', linewidth=0.5, linestyle='--')
plt.axvline(0, color='red', linewidth=0.5, linestyle='--')
plt.scatter([root_2_simple, root_2_scipy], [equation_2_fx(root_2_simple), equation_2_fx(root_2_scipy)], color='red', label='Найденные корни', zorder=5)
plt.title('График функции y = x^3 + x^2 + 9')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

