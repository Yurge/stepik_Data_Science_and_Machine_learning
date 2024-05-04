import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', rc={'figure.figsize': (12, 6)})
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

# СТОИТ ЗАДАЧА: ПРЕДСКАЗАТЬ ПО ИМЕЮЩИМСЯ ФИЧАМ, СЪЕДОБЕН ЛИ ГРИБ

mushrooms_train = pd.read_csv('machines/mushrooms/training_mush.csv')
mushrooms_X_test = pd.read_csv('machines/mushrooms/testing_mush.csv')
mushrooms_y_test = pd.read_csv('machines/mushrooms/testing_y_mush.csv')

# print(mushrooms_train.columns)
# print(mushrooms_X_test.shape)

X_train = mushrooms_train.drop(['class'], axis=1)
y_train = mushrooms_train['class']


# Построим RandomForest, сохраним в переменную лучшие параметры
# clf_rf = RandomForestClassifier(random_state=0)
tree_params_rf = {
        'n_estimators': range(10, 50, 10),
        'max_depth': range(1, 12, 2),
        'min_samples_leaf': range(1, 7),
        'min_samples_split': range(2, 9, 2)
}
# tree_grid_rf = GridSearchCV(clf_rf, tree_params_rf, cv=3, n_jobs=-1, verbose=True)
# tree_grid_rf.fit(X_train, y_train)
# print(tree_grid_rf.best_params_, sep='\n')
best_clf_rf = RandomForestClassifier(max_depth=9,  min_samples_leaf=1, min_samples_split=2, n_estimators=10)
best_clf_rf.fit(X_train, y_train)

# Посмотрим на значимость параметров в деревьях, т.е. как часто используется параметр при сплите
# imp = pd.DataFrame(best_clf_rf.feature_importances_, index=X_train.columns, columns=['importance'])
# imp.sort_values('importance').plot(kind='barh')
# print(imp.sort_values('importance'))
# plt.show()

# предскажем значения на тестовых данных и выведем их количество
y_predict_rf = best_clf_rf.predict(mushrooms_X_test)
# print(pd.Series(y_predict_rf).value_counts())
y_predict_prob_rf = best_clf_rf.predict_proba(mushrooms_X_test)[:, 1]

# рассчитаем confusion matrix по предсказаниям и посмотрим на их значения и на график
conf_matrix = confusion_matrix(mushrooms_y_test, y_predict_rf)
# print(conf_matrix)
sns.heatmap(conf_matrix, annot=True, cmap="Blues")
plt.show()


def all_metrics(y_test, y_pred, y_pred_prob):
    report = classification_report(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return (f'{report}     roc_auc       {roc_auc:.3f}\n'
            f'________________________  END  ______________________')


print(f'RandomForest metrics: \n\n{all_metrics(mushrooms_y_test, y_predict_rf, y_predict_prob_rf)}\n')
