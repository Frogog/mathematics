import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


x_norm = np.linspace(-8, 8, 500)
sigmas = [1, 2, 3]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for sigma, color in zip(sigmas, colors):
    pdf = stats.norm.pdf(x_norm, loc=0, scale=sigma)
    cdf = stats.norm.cdf(x_norm, loc=0, scale=sigma)
    plt.plot(x_norm, pdf, color=color, label=f'σ = {sigma}')
plt.title('Плотность.\nНормальное распределение')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for sigma, color in zip(sigmas, colors):
    cdf = stats.norm.cdf(x_norm, loc=0, scale=sigma)
    plt.plot(x_norm, cdf, color=color, label=f'σ = {sigma}')
plt.title('Функция распределения.\nНормальное распределение')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Б) a = 0, 1, 2, при σ = 1
plt.figure(figsize=(12, 5))
means = [0, 1, 2]
colors = ['blue', 'green', 'red']

plt.subplot(1, 2, 1)
for mu, color in zip(means, colors):
    pdf = stats.norm.pdf(x_norm, loc=mu, scale=1)
    plt.plot(x_norm, pdf, color=color, label=f'a = {mu}')
plt.title('Плотность.\nНормальное распределение')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for mu, color in zip(means, colors):
    cdf = stats.norm.cdf(x_norm, loc=mu, scale=1)
    plt.plot(x_norm, cdf, color=color, label=f'a = {mu}')
plt.title('Функция распределения.\nНормальное распределение')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


x_exp = np.linspace(0, 5, 500)
lambdas = [0.5, 1, 2]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for lam, color in zip(lambdas, colors):
    pdf = stats.expon.pdf(x_exp, scale=1/lam)
    plt.plot(x_exp, pdf, color=color, label=f'λ = {lam}')
plt.title('Плотность.\nПоказательное распределение')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for lam, color in zip(lambdas, colors):
    cdf = stats.expon.cdf(x_exp, scale=1/lam)
    plt.plot(x_exp, cdf, color=color, label=f'λ = {lam}')
plt.title('Функция распределения.\nПоказательное распределение')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


x_weibull = np.linspace(0, 4, 500)
weibull_params = [
    (5, 1, 'a=5, β=1'),
    (0.5, 1, 'a=0.5, β=1'),
    (1.5, 1, 'a=1.5, β=1'),
    (1, 1, 'a=1, β=1')
]
colors = ['blue', 'green', 'red', 'orange']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for (a, beta, label), color in zip(weibull_params, colors):
    pdf = stats.weibull_min.pdf(x_weibull, c=beta, scale=a)
    plt.plot(x_weibull, pdf, color=color, label=label)
plt.title('Плотность.\nРаспределение Вейбулла')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for (a, beta, label), color in zip(weibull_params, colors):
    cdf = stats.weibull_min.cdf(x_weibull, c=beta, scale=a)
    plt.plot(x_weibull, cdf, color=color, label=label)
plt.title('Функция распределения.\nРаспределение Вейбулла')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


x_lognorm = np.linspace(0.01, 10, 500)
lognorm_params = [
    (0.7, 1, 'a=0.7, σ²=1'),
    (0.5, 0.3, 'a=0.5, σ²=0.3'),
    (1, 1, 'a=1, σ²=1')
]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for (a, sigma2, label), color in zip(lognorm_params, colors):
    sigma = np.sqrt(sigma2)
    pdf = stats.lognorm.pdf(x_lognorm, s=sigma, scale=np.exp(a))
    plt.plot(x_lognorm, pdf, color=color, label=label)
plt.title('Плотность.\nЛогнормальное распределение')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for (a, sigma2, label), color in zip(lognorm_params, colors):
    sigma = np.sqrt(sigma2)
    cdf = stats.lognorm.cdf(x_lognorm, s=sigma, scale=np.exp(a))
    plt.plot(x_lognorm, cdf, color=color, label=label)
plt.title('Функция распределения.\nЛогнормальное распределение')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


x_poisson = np.arange(0, 12, 1)
poisson_lambdas = [0.5, 1, 2, 3.5]
colors = ['blue', 'green', 'red', 'orange']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for lam, color in zip(poisson_lambdas, colors):
    pmf = stats.poisson.pmf(x_poisson, mu=lam)
    plt.plot(x_poisson, pmf, 'o-', color=color, label=f'λ = {lam}')
plt.title('Закон Пуассона')
plt.xlabel('x')
plt.ylabel('P(X=x)')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for lam, color in zip(poisson_lambdas, colors):
    cdf = stats.poisson.cdf(x_poisson, mu=lam)
    plt.plot(x_poisson, cdf, 'o-', color=color, label=f'λ = {lam}')
plt.title('Закон Пуассона')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


x_binom = np.arange(0, 21, 1)
binom_params = [
    (10, 0.1, 'n=10, p=0.1'),
    (10, 0.3, 'n=10, p=0.3'),
    (20, 0.1, 'n=20, p=0.1'),
    (20, 0.3, 'n=20, p=0.3')
]
colors = ['blue', 'green', 'red', 'orange']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for (n, p, label), color in zip(binom_params, colors):
    pmf = stats.binom.pmf(x_binom, n=n, p=p)
    plt.plot(x_binom, pmf, 'o-', color=color, label=label, markersize=4)
plt.title('Биномиальное распределение')
plt.xlabel('Число успехов')
plt.ylabel('P(X=x)')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
for (n, p, label), color in zip(binom_params, colors):
    cdf = stats.binom.cdf(x_binom, n=n, p=p)
    plt.plot(x_binom, cdf, 'o-', color=color, label=label, markersize=4)
plt.title('Биномиальное распределение')
plt.xlabel('Число успехов')
plt.ylabel('F(x)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()