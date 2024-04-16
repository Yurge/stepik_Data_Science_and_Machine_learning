import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

predicted = dt.predict(X_test)


# Задание - осуществите перебор всех деревьев (GridSearchCV) на данных ириса по следующим параметрам:
params = {
    'max_depth': range(1, 11),
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 11)
}
search = GridSearchCV(dt, param_grid=params, cv=5)
search.fit(X, y)
best_tree = search.best_estimator_         # (max_depth=3, min_samples_split=4)



# Задание - осуществим поиск по тем же параметрам что и в предыдущем задании с помощью RandomizedSearchCV
search2 = RandomizedSearchCV(dt, param_distributions=params, cv=5)
search2.fit(X, y)
best_tree2 = search2.best_estimator_        # (max_depth=6, min_samples_split=3)
