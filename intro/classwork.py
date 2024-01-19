import matplotlib.pyplot as plt
import pandas as pd
import numpy
import seaborn as sns


students = pd.read_csv('intro/data/StudentsPerformance.csv')
dota = pd.read_csv('intro/data/dota_hero_stats.csv')
acc = pd.read_csv('intro/data/accountancy.csv')
concentrations = pd.read_csv('intro/data/algae.csv')
salary = pd.read_csv('intro/data/income.csv')
df = pd.read_csv('intro/data/dataset_209770_6-2.txt', sep=' ')
genome = pd.read_csv('intro/data/genome_matrix.csv')
iris = pd.read_csv('intro/data/iris.csv', index_col=0)


students.columns = students.columns.str.replace(' ', '_')
students.columns = students.columns.str.lower()


# Посмотрим на стандартные статистические данные студентов со стандартным ланчем и урезанным
lunch_standard = students.query('lunch == "standard"') # 1й способ
lunch_reduced = students[students.lunch == 'free/reduced']   # 2й способ
#print(f'lunch_standard: \n {lunch_standard.describe()} \n '
#      f'lunch_reduced: \n {lunch_reduced.describe()}')


# Посчитаем средние оценки в зависимости от пола
mean_score_by_gender = students.groupby('gender', as_index=False) \
    .agg({'math_score': 'mean', 'reading_score': 'mean', 'writing_score': 'mean'}) \
    .rename(columns={'math_score': 'mean_math_score', 'reading_score': 'mean_reading_score', 'writing_score': 'mean_writing_score'})


# Отберем по 5 лучших парней и девушек по предмету Математика
best_score_in_math = students[['gender', 'math_score']] \
    .sort_values(['gender', 'math_score'], ascending=False) \
    .groupby('gender').head(5)

# Сгруппируйте по колонкам attack_type и primary_attr и выберите самый распространённый набор характеристик
max_type_attr = dota \
    .groupby(['attack_type', 'primary_attr'], as_index=False) \
    .agg({'id': 'count'}) \
    .sort_values('id', ascending=False) \
    .head(1)

# Найти среднюю концентрацию каждого из веществ в каждом из родов (колонка genus)
mean_concentrations = concentrations.groupby('genus').mean(numeric_only=True)


# укажите через пробел чему равны мин-я, средняя и макс-я концентрации аланина (alanin) среди видов рода Fucus.
# Округлите до 2-ого знака
alanin_by_fucus = concentrations \
    .query('genus == "Fucus"')['alanin'] \
    .describe().round(2) \
    .filter(items=['min', 'mean', 'max'], axis='rows')


# Какое количество ролей в dota самое распространённое?
dota['count_roles'] = dota.roles.str.split(', ').apply(len)
main_role = dota.count_roles.value_counts()

#print(main_role)
#sns.barplot(data=main_role)

# Визуализация:

#students.math_score.hist() # Встроенный в pandas метод построения гистограммы
#students.plot.scatter(x='math_score', y='reading_score') # Тоже встроенный

#ax = sns.scatterplot(data=students, x='math_score', y='reading_score', hue='gender')
#ax.set_xlabel('Баллы по Математике')
#ax.set_ylabel('Баллы по Чтению')

#sns.scatterplot(x=df.x, y=df.y)

# Тепловая карта генома
#g = sns.heatmap(data=genome.iloc[:, 1:], cmap='viridis')
#g.xaxis.set_ticks_position('top')
#g.xaxis.set_tick_params(rotation=90)

# iris - датасэт со значениями параметров ирисов, постройте их распределения
iris.columns = iris.columns.str.replace(' ', '_')
#sns.kdeplot(data=iris.iloc[:, :4])
#sns.violinplot(data=iris.petal_length)
#sns.pairplot(data=iris.iloc[:, :4])

plt.show()