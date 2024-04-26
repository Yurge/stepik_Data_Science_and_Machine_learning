import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (12, 6)})     # сразу зададим размер будущих графиков
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


# ЗАДАЧА: ПОЧЕМУ ПОЛЬЗОВАТЕЛИ НЕ ЗАКАНЧИВАЮТ КУРС НА СТЕПИКЕ?  КАК ПРЕДСКАЗАТЬ, ЧТО ПОЛЬЗОВАТЕЛЬ УЙДЕТ С КУРСА?


events_data = pd.read_csv('machines/ML_contest/event_data_train.csv')
submissions_data = pd.read_csv('machines/ML_contest/submissions_data_train.csv')
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
# print(submissions_data.step_id.nunique())
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

# ПЕРЕФОРМУЛИРУЕМ ЗАДАЧУ: МОЖНО ЛИ ЧЕРЕЗ n ДНЕЙ ПРЕДСКАЗАТЬ, ДРОПНИТСЯ ЛИ ЮЗЕР ?


# Добавим информацию о первом вхождении пользователя на курс, чтобы от неё отталкиваться
user_min_time = events_data \
    .groupby('user_id', as_index=False) \
    .agg({'timestamp': 'min'}) \
    .rename(columns={'timestamp': 'min_timestamp'})

# Добавим эти данные в нашу сводную таблицу
users_data = users_data.merge(user_min_time, how='outer', on='user_id')

# мы попробуем предсказать по первым 3 дням, дропнется ли юзер,
# но для этого нужно по каждому юзеру вытащить из таблицы первые 3 дня, чтобы получить тренировочный датафрэйм
events_data = events_data.merge(user_min_time, how='outer', on='user_id')
learning_time = 24 * 60 * 60 * 3        # 3 дня в секундах (можно менять на другое кол-во дней)
events_data_train = events_data[events_data.timestamp <= events_data.min_timestamp + learning_time]
# Проверим, что отобрали не более 3х дней
events_data_train.groupby('user_id').day.nunique().max()   # 4 (если решать 3 дня (72 ч), то можно заскочить на 4 день)

submissions_data = submissions_data.merge(user_min_time, how='outer', on='user_id')
learning_time = 24 * 60 * 60 * 4        # 3 дня в секундах (можно менять на другое кол-во дней)
submissions_data_train = submissions_data[submissions_data.timestamp <= submissions_data.min_timestamp + learning_time]
# Проверим, что отобрали не более 3х дней
submissions_data_train.groupby('user_id').day.nunique().max()   # 4


# Теперь попробуем сформировать переменную X, добавляя в неё разные фичи
# и посчитаем scores, чтобы понять, возможно ли обучение на этих данных за 3 дня


X = submissions_data_train \
    .groupby('user_id').day.nunique().to_frame().reset_index() \
    .rename(columns={'day': 'uniq_days'})

uniq_steps = submissions_data_train \
    .groupby('user_id').step_id.nunique().to_frame().reset_index() \
    .rename(columns={'step_id': 'uniq_steps'})
# Смерджим с данными из uniq_steps
X = X.merge(uniq_steps, how='outer', on='user_id')
# Смерджим с данными из submissions_data_train
X = X.merge(submissions_data_train.pivot_table(index='user_id', columns='submission_status', values='step_id',
                                               aggfunc='count', fill_value=0).reset_index(),
            how='outer', on='user_id')
# Добавим столбик: отношение правильных ответов к сумме всех ответов. Возможно это улучшит обучение
X['correct_ratio'] = X.correct / (X.correct + X.wrong)
# Смерджим с данными из events_data_train
X = X.merge(events_data_train.pivot_table(index='user_id', columns='action', values='step_id',
                                          aggfunc='count', fill_value=0).reset_index(),
            how='outer', on='user_id')
X = X.fillna(0)
# Нам ещё очень нужны два столбика из users_data, дропнулся ли юзер и закончил ли он курс
X = X.merge(users_data[['user_id', 'is_gone_user', 'passed_course']], how='outer', on='user_id')


# Посмотрев на данные X, мы понимаем, что нам НЕ нужны юзеры, которые зашли на курс и спокойно учатся,
# т.е. у них passed_course == False  and  is_gone_user == False. Таких юзеров нужно убрать из выборки и вот как мы
# это сделаем:
X = X.query('passed_course == True or is_gone_user == True')
# Давайте проверим себя, чтобы не получилось вариантов False - False
# print(X.groupby(['passed_course', 'is_gone_user']).user_id.count())     # Всё верно

# Теперь можно сформировать переменные X, y
y = X.passed_course.map(int)
X = X.drop(['correct', 'started_attempt','is_gone_user', 'passed_course'], axis=1)
# Т.к. колонка user_id нам тоже не нужна, просто ради приличия можно оставить её в качестве индекса строк
#X = X.set_index('user_id')

# Посмотрим на р-значения колонок. Если оно сильно больше 0.05, то удалим их из X
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
# print(result.summary2())


# ----------------------------------------  ПЕРЕХОДИМ К ОБУЧЕНИЮ МОДЕЛИ  ---------------------------------------


# Разобьём данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Найдём лучшие параметры дерева
clf = DecisionTreeClassifier()
tree_params = {'criterion': ['gini', 'entropy'],
               'max_depth': range(2,20),
               'max_features': range(3,10)}
tree_grid = GridSearchCV(clf, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(X_train, y_train)
best_clf = tree_grid.best_estimator_
# print(tree_grid.best_params_, '\n')
print()


y_predict = best_clf.predict(X_test)
# print(y_predict, '\n')
# Сохраним в переменную все данные о вероятностях отнесения к одному из 2х классов, чтобы ими можно было манипулировать
y_predicted_prob = best_clf.predict_proba(X_test)[:, 1]
# print(y_predicted_prob, '\n')

# Чтобы менять метрикиы, можем менять какую вероятность мы будем относить к TruePositive. Для этого пропишем в
# программе, чтобы относила пользователей к тем, кто закончит курс, если вероятность > 0.68
# y_predicted_prob = np.where(y_predicted_prob[:, 1] > 0.68, 1, 0)
# print(y_predicted_prob, '\n')
# Посмотрим на гистограмме распределение положительной вероятности
# pd.Series(y_predicted_prob).hist()
# plt.show()


# ----------------------------------------------------  МЕТРИКИ  --------------------------------------------------
# Вычислим значения основных метрик
def all_metrics(y_test, y_pred, y_pred_prob):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return (f'cnf_matrix = \n{cnf_matrix} \n{report} \nroc_auc = {roc_auc:.3f}\n')


print(f'DecisionTree metrics: \n\n{all_metrics(y_test, y_predict, y_predicted_prob)}\n')


# Построим график Precision-Recall
display = PrecisionRecallDisplay.from_estimator(best_clf, X_test, y_test, name="LinearSVC", plot_chance_level=True)
_ = display.ax_.set_title("2-class Precision-Recall curve")


# Обязательно строим график ROC-кривая
# Чем больше площадь AUC - тем больше правильных предсказанных значений и меньше ложно-правильных
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc: .3f})')
plt.plot([0, 1], [0, 1] , 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



# Ещё можно применить логистическую регрессию. Давайте посмотрим на её метрики
log_regression = LogisticRegression(max_iter=6000)
log_regression.fit(X_train, y_train)
y_predict_log_reg = log_regression.predict(X_test)
y_predicted_prob_log_reg = log_regression.predict_proba(X_test)[:, 1]
# y_predicted_prob_log_reg = np.where(y_predicted_prob_log_reg[:, 1] > 0.5, 1, 0)
print(f'LogisticRegression metrics: \n\n{all_metrics(y_test, y_predict_log_reg, y_predicted_prob_log_reg)}\n')
# Метрики Логистической Регрессии оказались чуть лучше.




