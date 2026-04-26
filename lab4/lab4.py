import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AutoScalingSimulator:
    def __init__(self, K=100, r=0.1, tau=60, theta=10, A=80, k=5):
        """
        Инициализация параметров системы
        K: максимальная ёмкость (макс. RPS)
        r: скорость реакции автоскейлера
        tau: задержка масштабирования (с)
        theta: порог эффекта Олли
        A: порог загрузки CPU (%)
        k: количество добавляемых серверов за раз
        """
        self.K = K
        self.r = r
        self.tau = tau
        self.theta = theta
        self.A = A
        self.k = k
        self.history = []

    def logistic_growth_with_delay(self, N, t, history_func):
        """Уравнение Хатчинсона с запаздыванием"""
        if t <= self.tau:
            N_delayed = N
        else:
            N_delayed = history_func(t - self.tau)

        # Эффект Олли: если серверов меньше порога, эффективность падает
        olli_factor = max(0, (N - self.theta) / self.K) if self.theta > 0 else 1

        dNdt = self.r * N * olli_factor * (1 - N_delayed / self.K)
        return dNdt

    def simulate(self, T=1000, dt=1, initial_load=30, spike_time=200, spike_magnitude=150):
        """Моделирование автоскейлинга с улучшенной обработкой задержек"""
        time_points = np.arange(0, T, dt)
        N = np.zeros(len(time_points))
        N[0] = initial_load

        load = np.full(len(time_points), initial_load)
        load[spike_time:] = spike_magnitude

        # Гарантируем, что tau — корректное число
        tau = max(1, int(self.tau)) if not np.isnan(self.tau) else 60
        delay_steps = int(tau / dt)

        for i in range(1, len(time_points)):
            t = time_points[i]
            current_load = load[i]

            # Расчёт с учётом задержки
            if i >= delay_steps:
                N_delayed = N[i - delay_steps]
            else:
                N_delayed = initial_load

            # Эффект Олли
            olli_factor = max(0, (N[i-1] - self.theta) / self.K) if self.theta > 0 else 1

            # Уравнение Хатчинсона
            dN = self.r * N[i-1] * olli_factor * (1 - N_delayed / self.K) * dt
            N[i] = max(1, N[i-1] + dN)

            # Автоскейлинг по загрузке CPU
            cpu_usage = (current_load / N[i]) * 100 if N[i] > 0 else 100
            if cpu_usage > self.A and N[i] < self.K:
                N[i] += self.k
            elif cpu_usage < self.A * 0.7 and N[i] > initial_load:
                N[i] = max(initial_load, N[i] - self.k)

        self.history = list(zip(time_points, N, load))
        return time_points, N, load

    def optimize_parameters(self, target_stability_time=400):
        """Автоматическая оптимизация параметров с использованием scipy.optimize"""

        # Целевая функция для минимизации — комбинация стабильности и перелёта
        def objective_function(params):
            # Распаковываем параметры
            r, tau, theta, A, k = params

            # Сохраняем текущие параметры для восстановления
            original_params = {
                'r': self.r,
                'tau': self.tau,
                'theta': self.theta,
                'A': self.A,
                'k': self.k
            }

            # Устанавливаем тестируемые параметры
            self.r = r
            self.tau = tau
            self.theta = theta
            self.A = A
            self.k = k

            try:
                # Запускаем симуляцию с текущими параметрами
                time_points, N, load = self.simulate(
                    T=1500, dt=1,
            initial_load=30,
            spike_time=200,
            spike_magnitude=180
        )

                # Рассчитываем метрики
                stability = self.calculate_stability(N, target_stability_time)
                overshoot = self.calculate_overshoot(N)

                # Комбинация метрик для оптимизации (веса можно настроить)
                # Веса подобраны так, чтобы оба параметра влияли на результат
                objective_value = stability * 0.7 + overshoot * 300 * 0.3

            except:
                # В случае ошибки возвращаем большое значение
                objective_value = 1e6

            # Восстанавливаем оригинальные параметры
            for param, value in original_params.items():
                setattr(self, param, value)

            return objective_value

        # Начальные значения для оптимизации (текущие параметры)
        x0 = [self.r, self.tau, self.theta, self.A, self.k]

        # Границы для параметров
        bounds = [
            (0.05, 0.3),   # r: скорость роста (минимум 0.05, максимум 0.3)
            (10, 100),     # tau: задержка (10–100 секунд)
            (5, 25),       # theta: эффект Олли (5–25 серверов)
            (70, 95),     # A: порог CPU (70–95%)
            (3, 12)       # k: серверов за шаг (3–12)
        ]

        # Запускаем оптимизацию
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',  # метод для ограниченной оптимизации
            bounds=bounds,
            options={'maxiter': 50, 'disp': False}
        )

        if result.success:
            optimal_r, optimal_tau, optimal_theta, optimal_A, optimal_k = result.x
            print("Оптимизация успешна! Найдены оптимальные параметры:")
            print(f"  r (скорость роста): {optimal_r:.3f}")
            print(f"  τ (задержка): {optimal_tau:.0f} с")
            print(f"  θ (эффект Олли): {optimal_theta:.0f}")
            print(f"  A (порог CPU): {optimal_A:.0f}%")
            print(f"  k (серверов за шаг): {optimal_k:.0f}")
            return {
                'r': optimal_r,
                'tau': optimal_tau,
                'theta': optimal_theta,
                'A': optimal_A,
                'k': optimal_k
            }
        else:
            print("Оптимизация не удалась, используем начальные параметры")
            return None

    def calculate_stability(self, N, target_time):
        """Расчёт стабильности системы — дисперсия в установившемся режиме"""
        end_idx = min(int(target_time), len(N) - 1)
        start_idx = max(0, end_idx - 100)

        if start_idx >= end_idx:
            return 1e6

        segment = N[start_idx:end_idx]

        # Фильтруем некорректные значения
        segment_clean = [x for x in segment if not np.isnan(x) and not np.isinf(x) and x >= 0]
        if len(segment_clean) < 10:
            return 1e6

        variance = np.var(segment_clean)
        return min(variance, 1e6)

    def calculate_overshoot(self, N):
        """Расчёт «перелёта» системы"""
        if len(N) < 250:  # ждём достаточно времени после всплеска
            return 0.0

        # Ищем пик после всплеска (после 200 с)
        post_spike = N[200:]
        if len(post_spike) == 0:
            return 0.0
        peak = np.max(post_spike)

        # Установившееся значение — последние 50 точек
        steady = np.mean(N[-50:]) if len(N) >= 50 else np.mean(N)

        if steady <= 0 or peak <= steady or np.isnan(peak) or np.isnan(steady):
            return 0.0

        overshoot = (peak - steady) / steady
        return max(0.0, min(overshoot, 5.0))  # максимум 500 %

    def analyze_olli_effect(self):
        """Анализ воздействия эффекта Олли"""
        if self.theta <= 0:
            return "Эффект Олли не активирован (θ ≤ 0)"

        impact_score = self.theta / self.K * 100
        analysis = f"""
Анализ эффекта Олли:
- Порог θ = {self.theta} серверов
- Влияние на систему: {impact_score:.1f}% от максимальной ёмкости
- Воздействие:
  * При количестве серверов < θ система работает неэффективно
  * Замедляет реакцию на всплески трафика
  * Предотвращает чрезмерное масштабирование при малых нагрузках
  * Создаёт «зону устойчивости» вокруг минимального эффективного размера

Рекомендации:
- Если θ слишком высок: система будет медленно реагировать на всплески
- Если θ слишком низок: эффект незначителен
- Оптимально: θ ≈ 10–20% от K
"""
        return analysis

    def plot_results(self):
        """Визуализация результатов"""
        time_points, N, load = zip(*self.history)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(time_points, N, 'b-', linewidth=2, label='Количество серверов')
        ax1.axhline(y=self.K, color='r', linestyle='--', label='Максимальная ёмкость')
        ax1.set_ylabel('Количество серверов')
        ax1.set_title('Динамика автоскейлинга с задержкой и эффектом Олли')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time_points, load, 'g-', linewidth=2, label='Нагрузка (RPS)')
        ax2.set_ylabel('Нагрузка (RPS)')
        ax2.set_xlabel('Время (с)')
        ax2.set_title('Нагрузка на систему')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# Основной блок запуска
if __name__ == "__main__":
    # Начальные параметры — заведомо нестабильная система
    simulator = AutoScalingSimulator(
        K=100,
        r=0.2,   
        tau=5, 
        theta=3, 
        A=100,   
        k=15
    )
    print()
    print(f"1. Исходные параметры системы")
    print(f"  K (максимальная ёмкость): {simulator.K}")
    print(f"  r (скорость роста): {simulator.r:.3f}")
    print(f"  τ (задержка): {simulator.tau:.0f} с")
    print(f"  θ (эффект Олли): {simulator.theta:.0f}")
    print(f"  A (порог CPU): {simulator.A:.0f}%")
    print(f"  k (серверов за шаг): {simulator.k:.0f}")

    time_points, N, load = simulator.simulate(
        T=1500, dt=1,
        initial_load=30,
        spike_time=200,
        spike_magnitude=180
    )
    simulator.plot_results()
    print()
    print(f"2. Эффект Олли")
    print(simulator.analyze_olli_effect())

    print(f"3. Параметры после оптимизации системы")\

     # Автоматическая оптимизация параметров
    optimal_params = simulator.optimize_parameters(target_stability_time=400)

    if optimal_params is not None:
        # Применяем оптимизированные параметры
        simulator.r = optimal_params['r']
        simulator.tau = optimal_params['tau']
        simulator.theta = optimal_params['theta']
        simulator.A = optimal_params['A']
        simulator.k = optimal_params['k']
    else:
        print("Используем начальные параметры для второго запуска")

    time_points_opt, N_opt, load_opt = simulator.simulate(
        T=1500, dt=1,
        initial_load=30,
        spike_time=200,
        spike_magnitude=180
    )

    # Визуализация сравнения
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time_points, N, 'b--', linewidth=1.5, label='До оптимизации')
    ax1.plot(time_points_opt, N_opt, 'r-', linewidth=2, label='После оптимизации')
    ax1.axhline(y=simulator.K, color='k', linestyle='--', label='Максимальная ёмкость')
    ax1.set_ylabel('Количество серверов')
    ax1.set_title('Сравнение динамики автоскейлинга')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_points, load, 'g-', linewidth=2, label='Нагрузка (RPS)')
    ax2.set_ylabel('Нагрузка (RPS)')
    ax2.set_xlabel('Время (с)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    print()
    print(f"4. Вывод")

    try:
        stability_before = simulator.calculate_stability(N, 400)
        overshoot_before = simulator.calculate_overshoot(N)
        stability_after = simulator.calculate_stability(N_opt, 400)
        overshoot_after = simulator.calculate_overshoot(N_opt)

        print(f"Стабильность до оптимизации: {stability_before:.2f}")
        print(f"Перелёт до оптимизации: {overshoot_before:.2%}")
        print(f"Стабильность после оптимизации: {stability_after:.2f}")
        print(f"Перелёт после оптимизации: {overshoot_after:.2%}")

        # Расчёт улучшения cтабильности
        improvement_stability = (stability_before - stability_after) / stability_before * 100
        # Ограничиваем отрицательные улучшения нулём (улучшения нет)
        improvement_stability = max(0, improvement_stability)
        print(f"Улучшение стабильности: {improvement_stability:.1f}%")

        # Расчёт снижения перелёта
        improvement_overshoot = (overshoot_before - overshoot_after) / overshoot_before * 100
        improvement_overshoot = max(0, improvement_overshoot)
        print(f"Снижение перелёта: {improvement_overshoot:.1f}%")

    except Exception as e:
        print(f"Ошибка при расчёте метрик: {e}")
        print("Стабильность до оптимизации: ошибка расчёта")
        print("Перелёт до оптимизации: ошибка расчёта")
        print("Стабильность после оптимизации: ошибка расчёта")
        print("Перелёт после оптимизации: ошибка расчёта")
        print("Улучшение стабильности: ошибка расчёта")
        print("Снижение перелёта: ошибка расчёта")