from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
from IPython.display import SVG
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score

# СТОИТ ЗАДАЧА: ОБУЧИТЬ МАШИНУ, ЧТОБЫ ОНА ПРЕДСКАЗЫВАЛА, ПОГИБНЕТ ПАССАЖИР ТИТАНИКА ИЛИ НЕТ

titanic_data = pd.read_csv('machines/titanic/train.csv')

# Передадим в переменную Х данные для обучения, предварительно удалив ненужные фичи,
# а также удалим таргет фич "Survived" с данными о выживших
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
scores_max_depth_long = pd.melt(scores_max_depth,
                                id_vars=['max_depth'],
                                value_vars=['train_score', 'test_score', 'cross_val_score'],
                                ignore_index=False)
# print(scores_max_depth_long)
# можно на графике посмотреть лучшее соотношение глубины дерева с обучением
# sns.lineplot(scores_max_depth_long[scores_max_depth_long.max_depth < 50],
#             x='max_depth', y='value', hue='variable')

# Сделаем график "поближе"
#sns.lineplot(scores_max_depth_long[scores_max_depth_long.max_depth < 16],
#             x='max_depth', y='value', hue='variable')
# plt.show()
# Видно, что max_depth == 3 или может 4 является оптимальным для ML

# После того как мы добавили в наш цикл переменную mean_cross_val_score и посмотрели на новый график,
# мы увидели, что для кросс валидации лучшая глубина дерева получилась равной 10.
# Создаем классификатор best_clf, указывая в max_depth = 10. Обучаем на тренировочной выборке и замеряем точность на
# тестовой (см. ниже)

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
# print(cv_scores)
# print(f"Accuracy = {cv_scores.mean(): 0.3f}, with a standard deviation of {cv_scores.std(): 0.2f}")

better_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
better_clf.fit(X_train, y_train)
better_scores = better_clf.score(X_test, y_test)
# print(round(better_scores, 3))


# При помощи цикла мы "вручную" смогли выявить наилучшую глубину дерева, но конечно же в Pandas уже есть способы для
# того, чтобы определять наилучшие параметры построения дерева (обучения)
clf = tree.DecisionTreeClassifier()
tree_params = {'criterion': ['gini', 'entropy'],
               'max_depth': range(1,20),
               'max_features': range(4,10)}
tree_grid = GridSearchCV(clf, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(X_train, y_train)
best_clf = tree_grid.best_estimator_
# print(tree_grid.best_params_, '\n', tree_grid.best_score_)
y_predict = best_clf.predict(X_test)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)


# на самом деле программа не просто предсказывает выжил человек или не выжил, а программа вычисляет вероятность. Всё что
# больше 0.5 - это выжившие
y_predicted_prob = best_clf.predict_proba(X_test)
# print(y_predicted_prob)


# Там, где значения вероятности 0.9-1.0 или 0-0.1 - там понятно почти точно, что человек выжил или нет.
# но что делать с вероятностями в промежутке например 0.3-0.7 ?
# Мы можем сказать программе, чтобы она относила к выжившим только вероятность > 0.8, остальных отнести к погибшим,
# таким обазом будут меняться и значения precision и recall
y_predict = np.where(y_predicted_prob[:, 1] > 0.82, 1, 0)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
# print(precision, recall)


# Построим график Precision-Recall
from sklearn.metrics import PrecisionRecallDisplay
display = PrecisionRecallDisplay.from_estimator(
    best_clf, X_test, y_test, name="LinearSVC", plot_chance_level=True
                                                )
_ = display.ax_.set_title("2-class Precision-Recall curve")


# Обязательно строим график ROC-кривая
# Чем больше площадь AUC - тем больше правильных предсказанных значений и меньше ложно-правильных
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])
roc_auc= auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1] , 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
