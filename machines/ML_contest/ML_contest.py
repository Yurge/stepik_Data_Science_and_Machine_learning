import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (12, 6)})     # сразу зададим размер будущих графиков


# ЗАДАЧА: ПОЧЕМУ ПОЛЬЗОВАТЕЛИ НЕ ЗАКАНЧИВАЮТ КУРС НА СТЕПИКЕ?  КАК ПРЕДСКАЗАТЬ, ЧТО ПОЛЬЗОВАТЕЛЬ УЙДЕТ С КУРСА?


events_data = pd.read_csv('machines/ML contest/event_data_train.csv')
submissions_data = pd.read_csv('machines/ML contest/submissions_data_train.csv')
# Для того чтобы на будущих этапах во время фильтрации и группировки не удалить часть данных, создадим переменную с
# исходным количеством юзеров и будем всегда с ним сравнивать:
all_users = events_data.user_id.nunique()   # 19234


# добавим нужный столбик с датой
events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
# добавим нужный столбик с датой без времени
events_data['day'] = events_data.date.dt.date
submissions_data['day'] = submissions_data.date.dt.date
# print(events_data.columns)
# print(submissions_data.columns)
# print(events_data)
# print(submissions_data)


# Интересно посмотреть, сколько было уникальных пользователей в день
unic_id_in_day = events_data \
    .groupby('day') \
    .user_id.nunique()

# Посмотрим на графике на активность пользователей
# unic_id_in_day.plot()
# plt.show()


# посчитаем Сколько степов закончил каждый юзер (не уникальных степов) и посмотрим на графике
passed_steps_by_users = events_data[events_data.action == 'passed'] \
    .groupby('user_id', as_index=False) \
    .agg({'step_id': 'count'}) \
    .rename(columns={'step_id': 'passed_steps'})
# Данный способ не правильный, т.к. не берёт во внимание 2000 юзеров, которые не закончили ни одного степа,

# поэтому воспользуемся pivot_tables, чтобы развернуть наши данные и сформировать юзеров, у которых passed == 0
users_event_data = events_data \
    .pivot_table(index='user_id',
                 columns='action',
                 values='step_id',
                 aggfunc='count',
                 fill_value=0) \
    .reset_index()

# users_event_data.user_id.nunique() == 19234 (ни кого не потеряли)
# print(users_event_data)
# users_event_data.passed.hist()
# plt.show()


# Создадим табличку с данными сколько было правильных и не правильных попыток решений
users_scores = submissions_data \
    .pivot_table(index='user_id',
                 columns='submission_status',
                 values='step_id',
                 aggfunc='count',
                 fill_value=0) \
    .reset_index()


# Посмотрим, какие перерывы в днях были у пользователей:
# что делаем:
# оставили только три колонки,
# убрали дубликаты в двух колонках,
# сгруппировали по юзерам и выбрали колонку,
# занесли в список по каждому юзеру таймстэмпы когда он был на курсе,
# посчитали разницу между этими таймстэмпами
gap_data = events_data[['timestamp', 'user_id', 'day']] \
    .drop_duplicates(subset=['user_id', 'day']) \
    .groupby('user_id')['timestamp'] \
    .apply(list) \
    .apply(np.diff).values

gap_data = pd.Series(np.concatenate(gap_data, axis=0))
# переведем timestamp в дни
gap_data = round(gap_data / (24 * 60 * 60), 0)
# print(gap_data)

# посмотрим на графике, какие у пользователей перерывы в обучении
# gap_data[gap_data < 100].hist()
# plt.show()

gap_data.quantile(0.95)     # 60
gap_data.quantile(0.9)       # 18
# Квантиль 0,95 == 60 означает, что 95% данных из нашей серии имеют значения <= 60 и только 5% имеют значения более 60,
# но в данном случае нам не совсем важны точные определения и формулировки. Нам нужно от чего-то оттолкнуться, чтобы
# как-то определять дропнувшихся пользователей. Возьмём что-то среднее и это будет значение в 30 дней.


# Для того чтобы вычислить дропнувшихся юзеров, давайте по каждому юзеру посмотрим, последний день посещения курса
users_data = events_data \
    .groupby('user_id', as_index=False) \
    .agg({'timestamp': 'max'}) \
    .rename(columns={'timestamp': 'last_timestamp'})


# Создадим колонку со значением дропнулся юзер или нет
# только имеем в виду, что в этой колонке не будет учитываться, может юзер уже полностью закончил курс, поэтому
# не заходит более 30 дней
now = events_data.timestamp.max()           # это timestamp самого последнего дня наших наблюдений
drop_out_threshold = 24 * 60 * 60 * 30      # это 30 дней, переведенные в секунды, т.е. в ед.изм timestamp
users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold


# Посчитаем количество уникальных дней у каждого пользователя
users_count_uniq_days = events_data \
    .groupby('user_id')['day'] \
    .nunique() \
    .reset_index() \
    .rename(columns={'day': 'uniq_days'})


# соединим все таблички, чтобы получить одну информативную (не забудем проверить количество юзеров)
users_data = users_data.merge(users_scores, how='outer', on='user_id').fillna(0)
users_data = users_data.merge(users_event_data, how='outer', on='user_id')
users_data = users_data.merge(users_count_uniq_days, how='outer', on='user_id')

# Теперь нужно проверить, прошел ли пользователь наш курс? Чтобы получить сертификат, нужно набрать 171 бал
users_data['passed_course'] = users_data.passed > 170

# ради интереса посмотрим процент прошедших и не прошедших курс
# print(users_data['passed_course'].value_counts(normalize=True))


# НА ДАННОМ ЭТАПЕ НАВЕРНОЕ МОЖНО УЖЕ ПРОВЕСТИ ML, НО ПОТОМ У НАС БУДЕТ ЗАДАЧА ПРЕДСКАЗЫВАТЬ КАК МОЖНО БЫСТРЕЕ
# ДРОПНЕТСЯ ЛИ ЮЗЕР, НАПРИМЕР ЧЕРЕЗ НЕДЕЛЮ ПОСЛЕ НАЧАЛА КУРСА, ЧТОБЫ ЕГО ПОЙМАТЬ, ЧТОБЫ ОН НЕ УШЁЛ

