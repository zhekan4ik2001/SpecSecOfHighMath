import numpy as np

def matrix_multiply(A, B):
    # Если B — вектор, преобразуем в столбец
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    result = np.zeros((rows_A, cols_B))
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]

    if result.shape[1] == 1:
        return result.flatten()
    return result

def createidentity(n):
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.0
    return I

def vector_norm(v):
    return np.sqrt(np.sum(v ** 2))

# Метод степенных итераций для нахождения максимального собственного значения.
# Возвращает максимальное собственное значение матрицы A.
def power_iteration(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / vector_norm(x)

    lambda_old = 0
    for _ in range(max_iter):
        x_new = matrix_multiply(A, x)
        lambda_new = np.dot(x, x_new)

        if abs(lambda_new - lambda_old) < tol:
            break

        x = x_new / vector_norm(x_new)
        lambda_old = lambda_new

    return lambda_new

# Обратная степенная итерация для нахождения минимального собственного значения.
# Использует метод Гаусса для решения системы на каждой итерации.
def inverse_power_iteration(A, max_iter=1000, tol=1e-6):

    n = A.shape[0]
    x = np.random.rand(n)
    x = x / vector_norm(x)

    # Добавляем небольшое смещение, чтобы избежать вырожденности
    A_shifted = A + 1e-8 * createidentity(n)

    min_eigenvalue = 0
    for _ in range(max_iter):
        # Решаем систему A_shifted * y = x
        y = np.linalg.solve(A_shifted, x)
        min_eigenvalue_new = 1 / vector_norm(y)  # Упрощённая оценка

        if abs(min_eigenvalue_new - min_eigenvalue) < tol:
            break

        x = y / vector_norm(y)
        min_eigenvalue = min_eigenvalue_new

    return min_eigenvalue

#Метод простой итерации с предварительным преобразованием с адаптивным tau
def solve_by_transformed_iteration(A, b, eps=0.0001, max_iterations=2000):
    n = len(b)
    x = np.zeros(n)  # начальное приближение

    # Преобразуем систему Ax=b к виду x=Bx+c,
    # где B = (I - tau * A^T * A)
    # c = tau * A^T * b
    A_T = A.T
    A_T_A = matrix_multiply(A_T, A)
    A_T_b = matrix_multiply(A_T, b)

    # Находим максимальное и минимальное собственные значения A^T A
    lambda_max = power_iteration(A_T_A)
    lambda_min = inverse_power_iteration(A_T_A)

    # Оптимальный tau для минимизации спектрального радиуса
    tau = 2 / (lambda_min + lambda_max)

    # x = (I - tau * A^T A) x + tau * A^T b
    I = createidentity(n)
    B = I - tau * A_T_A
    c = tau * A_T_b
    
    spectral_radius = power_iteration(np.abs(B))

    for iteration in range(max_iterations):
        x_new = matrix_multiply(B, x) + c

        # Проверяем условие сходимости
        if np.max(np.abs(x_new - x)) < eps:
            return x_new, iteration + 1

        x = x_new.copy()

    return x, max_iterations

# Исходные данные
A = np.array([
    [0.26, -0.14, 0.02, 0.24],
    [0.15, 0.12, -0.17, 0.26],
    [0.35, 0.22, -0.03, -0.27],
    [0.12, 0.24, -0.15, 0.23]
])

b = np.array([-1.73, 0.65, -2.26, 1.17])

# Решаем систему преобразованным итерационным методом
solution, num_iterations = solve_by_transformed_iteration(A, b, eps=0.0001)

# Выводим результаты
print("\nРешение системы уравнений преобразованным итерационным методом:")
if(num_iterations >= 2000):
    print("Корни не найдены")
else:
    print("Корни найдены")
    for i, val in enumerate(solution, start=1):
        print(f"x_{i} = {val:.6f}")
    print(f"Количество итераций: {num_iterations}")
    print()

# Сравнение с точным решением (метод Гаусса)
solution_direct = np.linalg.solve(A, b)
print("Корени найденные с помощью функции np.linalg.solve:")
for i, val in enumerate(solution_direct, start=1):
    print(f"x_{i} = {val:.6f}")
