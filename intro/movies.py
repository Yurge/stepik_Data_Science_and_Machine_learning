import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time


movie = pd.read_csv('/Users/yvvyds/PycharmProjects/stepik_Data_Science_and_Machine_learning/'
                    '/intro/data/movie_metadata.csv')

# print(movie.columns)

genres = movie[['movie_title', 'genres']]
budget = movie[['budget', 'duration']]
# print(budget.head())

budget = budget.map(lambda x: x + 1)    # можно использовать apply вместо map
# print(budget.head())


# Найдем среднее значение в каждом столбике
# print(budget.apply(np.mean))

# Найдем среднее значение в каждой строке
# print(budget.apply(np.mean, axis=1))

# без функции apply код будет быстрее:
# print(budget.mean() + 1000)

# Но ещё быстрее будет через numpy array, но сначала удалим значения NaN
# print(np.mean(budget['budget'].dropna().values))

# Давайте посмотрим, какое время выполнения имеют разные методы:
before = time()
# movie.budget.mean(axis=0)     # 0.0002
# movie.budget.apply('mean')    # 0.0003
# np.mean(movie.budget.dropna().values)   # 0.0004
# movie.budget.apply(np.mean)   # 0.02
# movie.budget.describe().loc['mean']    # 0.03
after = time()
# print(after - before)


# ----------------------------- Поработаем с данными биржи ----------------------------
stock = pd.read_csv('/Users/yvvyds/PycharmProjects/stepik_Data_Science_and_Machine_learning/'
                    '/intro/data/amzn_stock.csv',
                    index_col='Date', parse_dates=True)
# Переведя parse_dates=True, мы теперь сможем правильно работать с датами (как с датами, а не как с цифрами)
# Теперь мы можем отфильтровать данные только за 2010 год или взять определенный временной отрезок
# print(stock['2010':'2010'])
# print(stock['2010-02':'2011-03'])

# Существует метод resample, при помощи которого можно сагрегировать данные в меньшую или большую сторону,
# например разделить на 2 часа или разбить по недельно
# print(stock.resample('2h').asfreq())
# print(stock.resample('1w').mean())

# Есть функция rolling, которая высчитывает скажем среднее значение за 3 дня (за текущий день и два дня передним).
# Это плавающее окно. Когда текущий день сменяется, то среднее значение за последние 3 дня будет пересчитано.
# print(stock.rolling(window=3).mean())
# print(stock.rolling(3, min_periods=1).mean())   # Заполняем NaN

# Есть отличная функция expanding, которая может считать на всех текущих данных например среднее значение,
# т.е. если всего 10 строк данных, то посчитает среднюю за 10 дней, а завтра пересчитает среднюю уже за 11 дней.
# print(stock.expanding().mean())
# print(stock.expanding(min_periods=3).mean())    # Начинаем с третьего значения

# Есть среднее экспоненциальное, т.е. текущее значение имеет наибольший вес, а предыдущие значения с угасающими весами
# print(stock.ewm(alpha=0.7).mean())

# Посмотрим на данные в индексе
print(stock.index.day_name().value_counts())


# --------- Сглаживание кривых -----------
# посмотрим на график:
stock['Open'].plot()
# plt.show()
# теперь сформируем график из средних значений за 10 дней
stock['Open'].rolling(21, min_periods=1).mean().plot()
# plt.show()  # Видно, что исчезли шиповидные локальные максимумы и минимумы
