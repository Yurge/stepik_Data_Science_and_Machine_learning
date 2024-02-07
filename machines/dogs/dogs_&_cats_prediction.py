import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

# ЗАДАЧА: НАЙТИ ОПТИМАЛЬНУЮ ГЛУБИНУ ДЕРЕВА, ПРОВЕСТИ ML, ПРЕДСКАЗАТЬ СТОЛБИК "ВИД" ДЛЯ ТЕСТОВЫХ ДАННЫХ

df_train = pd.read_csv('machines/dogs/dogs_n_cats.csv')
df_test = pd.read_json('machines/dogs/dataset_209691_15.txt')

X = df_train.drop(['Длина', 'Высота', 'Вид'], axis=1)
y = df_train['Вид']
# print(X.isna().sum())

X_TEST = df_test[X.columns]      # Оставляем такие же колонки и в том же порядке, как в тренировочном 'X'.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

scores_max_depth = pd.DataFrame()
max_depths = range(1, 51)
for depth in max_depths:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    clf.fit(X_train, y_train)
    train_scores = clf.score(X_train, y_train)
    test_scores = clf.score(X_test, y_test)
    scores = pd.DataFrame({'max_depth': [depth],
                           'train_scores': [train_scores],
                           'test_scores': [test_scores]})
    scores_max_depth = pd.concat([scores_max_depth, scores])

scores_max_depth_long = pd.melt(scores_max_depth, id_vars='max_depth', value_vars=['train_scores', 'test_scores'])

sns.lineplot(scores_max_depth_long[scores_max_depth_long.max_depth < 6], x='max_depth', y='value', hue='variable')
# plt.show()

best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
best_clf.fit(X_train, y_train)
best_train_scores = best_clf.score(X_train, y_train)
best_test_scores = best_clf.score(X_test, y_test)

# print(best_train_scores)    # 0.97
# print(best_test_scores)     # 0.98

Y_TEST = best_clf.predict(X_TEST)
df_test['Вид'] = pd.Series(Y_TEST)
# print(df_test['Вид'].value_counts())

plt.figure(figsize=(25, 7))
tree.plot_tree(best_clf, fontsize=7, feature_names=list(X), filled=True)
# plt.show()
