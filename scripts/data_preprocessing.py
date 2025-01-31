import pandas as pd
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('data/train.csv')

# Общая информация о данных
print(df.info())

# Проверяем на наличие пропущенных значений
print(df.isnull().sum())

# Описательная статистика
print(df.describe())

# Описание категориальных переменных
print(df['Sentiment'].value_counts())

# Заполнение NaN значениями по умолчанию (например, 'отсутствует') перед конвертацией
df['Text'] = df['Text'].fillna('отсутствует').astype(str)

# Проверка типа данных после конвертации
print(df['Text'].dtype)

sns.countplot(data=df, x='Sentiment')
plt.title('Распределение настроений')
plt.xlabel('Настроение')
plt.ylabel('Количество')
plt.show()