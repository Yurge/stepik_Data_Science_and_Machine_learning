import pandas as pd
import scipy.stats as ss

# Посмотрим на значение Энтропии и Прироста Информации  по каждому столбику файла cats.csv
cats = pd.read_csv('machines/data/cats.csv')


def entrop(data, target):
    answer = ss.entropy(data[target].value_counts() / len(data), base=2)
    return round(answer, 2)


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
    return round(IG, 2)


features = cats.columns[1:-1]
target = cats.columns[-1]
for feature in features:
    for value in cats[feature].unique():
        data = cats[cats[feature] == value]
        print(f'Энтропия по фиче "{feature}" со значением {value} = {entrop(data, target)}')
    ig = information_gain(cats, feature)
    print(f'IG по фиче "{feature}" = {ig} \n')
