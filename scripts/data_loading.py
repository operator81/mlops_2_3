import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('data/train.csv')

# Посмотрим на первые 5 строк и выведем общую информацию о датафрейме
print(df.head())
print(df.info())