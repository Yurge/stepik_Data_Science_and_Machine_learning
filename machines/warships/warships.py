import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

warships_train = pd.read_csv('/Users/yvvyds/PycharmProjects/stepik_Data_Science_and_Machine_learning/'
                       'machines/warships/invasion.csv')

warships_X_test = pd.read_csv('/Users/yvvyds/PycharmProjects/stepik_Data_Science_and_Machine_learning/'
                       'machines/warships/operative_information.csv')

# print(warships_train)

# ['transport' 'fighter' 'cruiser'] = 2, 1, 0
# warships_train['answer'] = warships_train['class'].apply(lambda x: x.count('t'))
X_train = warships_train.drop(['class'], axis=1)
y_train = warships_train['class']

# print(y_train.head(20))

# Построим RandomForest, сохраним в переменную лучшие параметры
# clf_rf = RandomForestClassifier()
tree_params_rf = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': range(3, 12, 2),
    'max_depth': range(2, 4, 1),
    'max_features': range(1, 6),
    'min_samples_leaf': range(1, 3),
    'min_samples_split': range(0, 10, 2)
}
# tree_grid_rf = GridSearchCV(clf_rf, tree_params_rf, cv=5, n_jobs=-1, verbose=True)
# tree_grid_rf.fit(X_train, y_train)
# print(tree_grid_rf.best_params_, sep='\n')
best_clf_rf = RandomForestClassifier(criterion='gini', max_depth=2, max_features=2,
                                     min_samples_leaf=1, min_samples_split=2, n_estimators=3)
best_clf_rf.fit(X_train, y_train)

# предскажем значения на тестовых данных и выведем их количество
y_predict_rf = best_clf_rf.predict(warships_X_test)
print(pd.Series(y_predict_rf).value_counts())

# Посмотрим на значимость параметров в деревьях, т.е. как часто используется параметр при сплите
imp = pd.DataFrame(best_clf_rf.feature_importances_, index=X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='bar')
print(imp.sort_values('importance'))
