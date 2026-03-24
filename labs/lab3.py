import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

data = np.array([
    12.428786354828626, 0.503639618071029, -4.400173914948246, -3.9447899177705406,
    -4.0508028939459475, 0.43219880924676546, 1.0210852599609643, -8.467809387841262,
    -2.2818470676813742, 0.16015189092082438, -3.4013213587261273, 0.24397888071485796,
    1.6014338291622698, 11.391507647456601, -9.562977432981134, -0.14818386600585653,
    9.556829460305162, 1.8411425264982972, 1.4496505321201403, -4.547705324259587,
    -3.307313712614123, 0.9155042937688995, 7.070793905111495, 3.0143281640246276,
    8.648335202704184, -4.0630469663685655, -0.45059653868724125, 2.0497578737174624,
    3.5283120417059397, -4.502116902333219, 5.79962726649479, 8.85826931724092,
    0.34971900847973303, 7.272428880494553, -3.132008608847391, 2.098489736706833,
    2.184363089579856, -3.148334038744215, 0.18579395716893488, 0.14369572116062046,
    -4.159748990534572, 4.462658686204813, -3.6423601920634976, 3.9805014387436675,
    -0.2216824065928813, -0.562504177398514, -0.29292994779301806, 3.336852038298966,
    8.734668987269979, 1.8817003058554838
])

n = len(data)
print(f"Объём выборки n = {n}")
print(f"Минимум: {np.min(data):.4f}")
print(f"Максимум: {np.max(data):.4f}")

k = int(np.ceil(1 + 3.322 * np.log10(n)))
print(f"\nКоличество интервалов (Стерджесс): k = {k}")

x_min, x_max = np.min(data), np.max(data)
h = (x_max - x_min) / k
print(f"Длина интервала h = {h:.4f}")

bins = np.linspace(x_min, x_max, k + 1)
freq, bin_edges = np.histogram(data, bins=bins)

midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

print("\n=== Дискретный вариационный ряд ===")
print("Интервал\t\tСередина\tЧастота")
for i in range(len(freq)):
    print(f"[{bin_edges[i]:.2f}; {bin_edges[i+1]:.2f})\t{midpoints[i]:.2f}\t\t{freq[i]}")

plt.figure(figsize=(10, 6))
plt.hist(data, bins=bins, edgecolor='black', alpha=0.6, label='Гистограмма частот')
plt.plot(midpoints, freq, 'o-', color='red', label='Полигон частот', linewidth=2, markersize=8)
plt.title('Гистограмма и полигон частот (Вариант 25)')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


mean = np.mean(data)
var_biased = np.var(data, ddof=0)
var_unbiased = np.var(data, ddof=1)
std_biased = np.sqrt(var_biased)
std_unbiased = np.sqrt(var_unbiased)
max_freq_idx = np.argmax(freq)
modal_interval = [bin_edges[max_freq_idx], bin_edges[max_freq_idx + 1]]
mode_approx = midpoints[max_freq_idx]
median = np.median(data)
skewness = stats.skew(data, bias=False)
kurtosis = stats.kurtosis(data, bias=False)
cv = (std_unbiased / abs(mean)) * 100

print("\n=== Основные числовые характеристики ===")
print(f"Выборочное среднее x̄ = {mean:.4f}")
print(f"Выборочная дисперсия Dв = {var_biased:.4f}")
print(f"Исправленная дисперсия s² = {var_unbiased:.4f}")
print(f"Выборочное СКО σв = {std_biased:.4f}")
print(f"Исправленное СКО s = {std_unbiased:.4f}")
print(f"Мода Mo (интервал): [{modal_interval[0]:.2f}; {modal_interval[1]:.2f}) ≈ {mode_approx:.2f}")
print(f"Медиана Me = {median:.4f}")
print(f"Коэффициент асимметрии A = {skewness:.4f}")
print(f"Эксцесс E = {kurtosis:.4f}")
print(f"Коэффициент вариации v = {cv:.2f}%")

print("\n=== Проверка соответствия нормальному закону (экспресс-метод) ===")

# 1) По коэффициенту вариации (для нормального распределения v ≈ 30-35%)
if 20 < cv < 40:
    print(f"✓ Коэффициент вариации v = {cv:.2f}% находится в диапазоне 20-40% → возможно нормальное распределение.")
else:
    print(f"✗ Коэффициент вариации v = {cv:.2f}% выходит за пределы типичного для нормального диапазона (20-40%).")

# 2) По асимметрии и эксцессу (для нормального A ≈ 0, E ≈ 0)
if abs(skewness) < 0.5:
    print(f"✓ Асимметрия A = {skewness:.4f} близка к 0 → распределение симметрично.")
else:
    print(f"✗ Асимметрия A = {skewness:.4f} ≠ 0 → распределение асимметрично.")

if abs(kurtosis) < 0.5:
    print(f"✓ Эксцесс E = {kurtosis:.4f} близок к 0 → форма близка к нормальной.")
else:
    print(f"✗ Эксцесс E = {kurtosis:.4f} отличается от 0 → форма отличается от нормальной.")

# 3) Правило трёх сигм
lower_bound = mean - 3 * std_unbiased
upper_bound = mean + 3 * std_unbiased
outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"\nПравило трёх сигм: [{lower_bound:.2f}; {upper_bound:.2f}]")
print(f"Выбросов: {len(outliers)} ({(len(outliers)/n*100):.1f}%)")
if len(outliers) == 0:
    print("✓ Выбросы отсутствуют → признак нормальности.")
else:
    print(f"✗ Найдены выбросы: {outliers[:5]}... → возможны отклонения от нормальности.")

# ==================== 4. МЕТОД МОМЕНТОВ (М-ОЦЕНКИ) ====================
print("\n=== М-оценки параметров нормального закона (метод моментов) ===")
print("Для нормального распределения параметры оцениваются как:")
print(f"μ̂ = x̄ = {mean:.4f}")
print(f"σ̂ = σв = {std_biased:.4f} (или s = {std_unbiased:.4f})")
print("\nМ-оценки для нормального закона:")
print(f"Оценка математического ожидания: μ̂ = {mean:.4f}")
print(f"Оценка среднеквадратического отклонения: σ̂ = {std_biased:.4f}")

# ==================== 5. ПОСТРОЕНИЕ ФУНКЦИИ ПЛОТНОСТИ НОРМАЛЬНОГО ЗАКОНА ====================
# Интервал для построения (правило трёх сигм)
x_norm = np.linspace(mean - 3.5 * std_biased, mean + 3.5 * std_biased, 500)
pdf_norm = stats.norm.pdf(x_norm, loc=mean, scale=std_biased)

# Гистограмма с наложенной нормальной кривой
plt.figure(figsize=(10, 6))
plt.hist(data, bins=bins, density=True, edgecolor='black', alpha=0.6, label='Гистограмма (нормированная)')
plt.plot(x_norm, pdf_norm, 'r-', linewidth=2, label=f'Нормальное распределение\nμ={mean:.2f}, σ={std_biased:.2f}')
plt.title('Функция плотности нормального закона (М-оценки параметров)')
plt.xlabel('x')
plt.ylabel('Плотность f(x)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Отдельно график плотности
plt.figure(figsize=(10, 5))
plt.plot(x_norm, pdf_norm, 'b-', linewidth=2)
plt.fill_between(x_norm, pdf_norm, alpha=0.3)
plt.title('Функция плотности нормального распределения\n(по М-оценкам параметров)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(alpha=0.3)
plt.show()

print("\n=== Вопросы для самопроверки ===")
print("1. Статистическая оценка — приближённое значение параметра распределения по выборке.")
print("2. Точность оценки — величина, характеризующая отклонение оценки от истинного параметра.")
print("3. Несмещённая: M(θ̂)=θ; состоятельная: θ̂→θ при n→∞; эффективная: минимальная дисперсия.")
print("4. Надёжность оценки — вероятность, с которой доверительный интервал покрывает параметр.")
print("5. Выборочное среднее — несмещённая и состоятельная оценка мат. ожидания.")
print("6. Исправленная выборочная дисперсия s² — несмещённая и состоятельная оценка дисперсии.")
print("7. Доверительный интервал — интервал, покрывающий параметр с заданной вероятностью (надежностью).")
print("8. М-оценки — оценки параметров, полученные методом моментов (приравнивание выборочных и теоретических моментов).")