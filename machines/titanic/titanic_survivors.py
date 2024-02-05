from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from graphviz import Source
from IPython.display import SVG
from IPython.display import display
from IPython.display import HTML
style = "<style>svg{width:70% !important;height:70% !important;}</style>"
HTML(style)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# СТОИТ ЗАДАЧА: ОБУЧИТЬ МАШИНУ, ЧТОБЫ ОНА ПРЕДСКАЗЫВАЛА, ПОГИБНЕТ ПАССАЖИР ИЛИ НЕТ

titanic_data = pd.read_csv('machines/titanic/train.csv')

# Передадим в переменную Х данные для обучения, предварительно удалив не нужные фичи,
# а также таргет фич "Survived" с данными о выживших
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']


# В некоторых колонках значения имеют тип строк, а не чисел. На этот случай в pandas есть метод
X = pd.get_dummies(X)
# Также есть пропущенные значения в колонке Age. Для простоты заменим их медианным возрастом всех пассажиров
X = X.fillna({'Age': X.Age.median()})

# Можно убедиться, что больше нет Nan
#print(X.isna().sum())

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Посмотрим на получившееся дерево решений
#plt.figure(figsize=(100, 25))
#tree.plot_tree(clf, fontsize=10, feature_names=list(X), filled=True)
#plt.show()

# Разобъём наши данные на обучающие и тестовые. При тестировании мы увидим на сколько эффективно
# предсказывает наша модель. Для того, чтобы не переобучить или недообучить нашу модель (маленькая эфф-ть пред-
# сказания), мы будем добавлять параметры при обучении.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)

train = clf.score(X_train, y_train)
test = clf.score(X_test, y_test)
#print(train, '\n', test, '\n')

# Добавим в дерево параметр максимальной глубины и, варьируя им, посмотрим как меняется способность предсказывать
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
clf.fit(X_train, y_train)
train = clf.score(X_train, y_train)
test = clf.score(X_test, y_test)
#print(train, '\n', test)

#plt.figure(figsize=(25, 7))
#tree.plot_tree(clf, fontsize=7, feature_names=list(X), filled=True)
#plt.show()

scores_max_depth = pd.DataFrame()
max_depth_values = range(1, 50)
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train = clf.score(X_train, y_train)
    test = clf.score(X_test, y_test)
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    temp_score = pd.DataFrame({'max_depth': [max_depth],
                               'train_score': [round(train, 2)],
                               'test_score': [round(test, 2)],
                               'cross_val_score': [round(mean_cross_val_score, 2)]
                               })
    scores_max_depth = pd.concat([scores_max_depth, temp_score])

# перегруппируем данные для построения графика
scores_max_depth_long = pd.melt(scores_max_depth, id_vars=['max_depth'], value_vars=['train_score', 'test_score', 'cross_val_score'], ignore_index=False)
# можно на графике посмотреть лучшее соотношение глубины дерева с обучением
#sns.lineplot(scores_max_depth_long[scores_max_depth_long.max_depth < 50],
#             x='max_depth', y='value', hue='variable')
sns.lineplot(scores_max_depth_long[scores_max_depth_long.max_depth < 16],
             x='max_depth', y='value', hue='variable')
#plt.show()
# Видно, что max_depth == 3 или может 4 является оптимальным для ML

# После того как мы добавили в наш цикл переменную mean_cross_val_score и посмотрели на новый график,
# мы видим, что для кросс валидации лучшая глубина дерева получилась равной 10.
# Создаем классификатор best_clf, указывая в max_depth = 10. Обучаем на тренировочной выборке и замеряем точность на тестовой (см. в самом низу)


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
scores = cross_val_score(clf, X_train, y_train, cv=5)
#print(scores)
#print(f"Accuracy = {scores.mean(): 0.3f}, with a standard deviation of {scores.std(): 0.2f}")

best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
best_clf.fit(X_train, y_train)
best_scores = best_clf.score(X_test, y_test)
print(best_scores)