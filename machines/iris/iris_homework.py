import pandas as pd
from sklearn import tree
import seaborn as sns

# ЗАДАЧА: ИСПОЛЬЗУЯ ТРЕНИРОВОЧНЫЕ И ТЕСТОВЫЕ ДАННЫЕ, НАЙТИ ОПТИМАЛЬНЫЙ ПОКАЗАТЕЛЬ ГЛУБИНЫ ДЕРЕВА, ПОСЛЕ ЧЕГО ПРОВЕСТИ ML

iris_data_train = pd.read_csv('machines/iris/train_iris.csv')
iris_data_test = pd.read_csv('machines/iris/test_iris.csv')

X_train = iris_data_train.drop(['Unnamed: 0', 'species'], axis=1)
y_train = iris_data_train.species
X_test = iris_data_test.drop(['Unnamed: 0', 'species'], axis=1)
y_test = iris_data_test.species

scores_max_depth = pd.DataFrame()
depth_value = range(1, 100)
for depth in depth_value:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    scores = pd.DataFrame({'max_depth': [depth],
                           'train_score': [train_score],
                           'test_score': [test_score]})
    scores_max_depth = pd.concat([scores_max_depth, scores])


scores_max_depth_long = pd.melt(scores_max_depth, id_vars=['max_depth'], value_vars=['train_score', 'test_score'])

sns.lineplot(scores_max_depth_long[scores_max_depth_long.max_depth < 6], x='max_depth', y='value', hue='variable')
# plt.show()

# На графике видим, что наилучшая глубина в районе max_depth=4. Создадим финальный классификатор:
best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
best_clf.fit(X_train, y_train)
best_train_score = best_clf.score(X_train, y_train)
best_test_score = best_clf.score(X_test, y_test)

# print(best_train_score)     # 1.0
# print(best_test_score)      # 0.92
