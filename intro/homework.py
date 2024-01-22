import pandas as pd
import numpy as np

# 1.
# Создать простейший датафрейм
my_data = pd.DataFrame({
    'type': ['A', 'A', 'B', 'B'],
    'value': [10, 14, 12 ,23]
})

# Загружаем и считываем файл с данными
my_stat = pd.read_csv('intro/data/my_stat.csv')

# 2.
# В переменную subset_1 сохраните только первые 10 строк и только 1 и 3 колонку.
# В переменную subset_2 сохраните все строки кроме 1 и 5 и только 2 и 4 колонку
subset_1 = my_stat.iloc[:10, [0, 2]]
subset_2 = my_stat.iloc[:, [1, 3]].drop([0, 4])

# 3.
# В переменную subset_3 сохраните только те наблюдения, у которых значения переменной V1  строго больше 0, и значение переменной V3  равняется 'A'.
# В переменную  subset_4  сохраните только те наблюдения, у которых значения переменной V2  не равняются 10, или значения переменной V4 больше или равно 1
subset_3 = my_stat.query('V1 > 0 & V3 == "A"')
subset_4 = my_stat.query('V2 != 10 | V4 >= 1')

# 4.
# В данных (my_stat) создайте две новые переменных: V5 = V1 + V4, V6 = натуральный логарифм переменной V2
my_stat['V5'] = my_stat.V1 + my_stat.V4
my_stat['V6'] = np.log(my_stat.V2)

# 5.
# Переименовать колонки в данных  my_stat
my_stat = my_stat.rename(columns={'V1': 'session_value',
                                  'V2': 'time',
                                  'V3': 'group',
                                  'V4': 'n_users'})


# 6.
stat = pd.read_csv('intro/data/my_stat_1.csv')
# В переменной session_value замените все пропущенные значения на нули
stat = stat.fillna(0)
# В переменной n_users замените все отрицательные значения на медианное значение переменной n_users (без учета отрицательных значений).
n_users_median = stat.query('n_users >= 0').n_users.median()
stat.n_users = stat.n_users.mask(stat.n_users < 0, n_users_median)


# 7.
# Рассчитайте среднее значение переменной session_value для каждой группы.
mean_session_value_data = my_stat \
    .groupby('group', as_index=False) \
    .agg({'session_value': 'mean'}) \
    .rename(columns={'session_value': 'mean_session_value'})