import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import scipy.stats as ss


# Задача: укажите, чему будет равняться значение Information Gain для переменной,  которая будет помещена в корень дерева.


data_heart = pd.read_csv('machines/practice/heart_disease.csv')
print(data_heart.columns)

X = data_heart.drop(['num'], axis=1)
y = data_heart.num

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

plt.figure(figsize=(40, 20), dpi=80)
p = tree.plot_tree(clf, fontsize=18, filled=True, feature_names=list(X))
# plt.show()


def entrop(data, target):
    answer = ss.entropy(data[target].value_counts() / len(data), base=2)
    return round(answer, 3)


def information_gain(data, feature):
    target = data.columns[-1]
    starting_entropy = entrop(data, target)
    num = data[feature].unique()
    ###
    n1 = sum(data[data[feature] == num[0]][target].value_counts().values.tolist())
    entr1 = entrop(data[data[feature] == num[0]], target)
    n2 = sum(data[data[feature] == num[1]][target].value_counts().values.tolist())
    entr2 = entrop(data[data[feature] == num[1]], target)
    N = n1 + n2
    IG = starting_entropy - ((n1 / N * entr1) + (n2 / N * entr2))
    return round(IG, 3)


features = data_heart.columns[1:-1]
target = data_heart.columns[-1]
for feature in features:
    ig = information_gain(data_heart, feature)
    print(f'IG по фиче "{feature}" = {ig} \n')