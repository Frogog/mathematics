import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = np.array([
    20.3, 15.4, 17.2, 19.2, 23.3,
    18.1, 21.9, 15.3, 16.8, 13.2,
    20.4, 16.5, 19.7, 20.5, 14.3,
    18.4, 16.8, 14.7, 20.8, 19.5,
    15.3, 19.3, 17.8, 16.2, 15.7,
    22.8, 21.9, 12.5, 10.1, 21.1,
    18.3, 14.7, 14.5, 18.2, 18.7,
    13.9, 19.1, 18.5, 20.2, 23.8,
    16.7, 20.4, 19.5, 11.8, 19.6,
    17.8, 21.3, 17.5, 19.4, 13.5,
    18.2, 19.3, 16.2, 16.4, 17.6
])

# 1. Первичная обработка: группировка с шагом h = 2, количество интервалов s = 7
# По данным: min = 10.1, max = 23.8, шаг 2 даёт интервалы: [10,12), [12,14), ..., [22,24]
bins = np.arange(10, 25, 2)  # [10,12,14,16,18,20,22,24] — 7 интервалов
freq, bin_edges = np.histogram(data, bins=bins)

midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

# Построение гистограммы и полигона
plt.figure(figsize=(10, 5))
plt.hist(data, bins=bins, edgecolor='black', alpha=0.6, label='Гистограмма частот')
plt.plot(midpoints, freq, 'o-', color='red', label='Полигон частот', linewidth=2, markersize=8)
plt.title('Рис. 1. Гистограмма и полигон частот')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

mean = np.mean(data)
median = np.median(data)
mode_result = stats.mode(data, keepdims=True)
mode = mode_result.mode[0]
mode_count = mode_result.count[0]

skewness = stats.skew(data, bias=False)
kurtosis = stats.kurtosis(data, bias=False)

std_dev = np.std(data, ddof=1)
cv = (std_dev / mean) * 100

print("\n=== Характеристики ===")
print(f"Среднее арифметическое: {mean:.4f}")
print(f"Медиана Me: {median:.4f}")
print(f"Мода: {mode:.4f} (встречается {mode_count} раз)")
print(f"Стандартное отклонение: {std_dev:.4f}")
print(f"Дисперсия: {np.var(data, ddof=1):.4f}")
print(f"Коэффициент асимметрии: {skewness:.4f}")
print(f"Эксцесс: {kurtosis:.4f}")
print(f"Коэффициент вариации: {cv:.2f}%")


if cv < 33:
    print(f"✓ Коэффициент вариации = {cv:.2f}% < 33% → выборка однородна.")
else:
    print(f"✗ Коэффициент вариации = {cv:.2f}% ≥ 33% → выборка неоднородна.")

mean_std = np.mean(data)
std_std = np.std(data, ddof=1)
lower_bound = mean_std - 3 * std_std
upper_bound = mean_std + 3 * std_std
outliers = data[(data < lower_bound) | (data > upper_bound)]

print(f"\nПравило трёх сигм: интервал [{lower_bound:.2f}; {upper_bound:.2f}]")
if len(outliers) == 0:
    print("✓ Выбросов нет. Данные однородны по правилу трёх сигм.")
else:
    print(f"✗ Найдены выбросы: {outliers} → данные могут быть неоднородны.")