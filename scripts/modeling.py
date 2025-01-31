import pandas as pd
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Загрузка данных
df = pd.read_csv('data/train.csv')

# Отфильтровываем только негативные и положительные отзывы
df = df[df['Sentiment'].isin(['Negative', 'Positive'])]

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

# Создание пайплайна для векторизации и классификации
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование на тестовом наборе
predictions = model.predict(X_test)

# Отчет о классификации
print(classification_report(y_test, predictions))

# Предсказания вероятности на новом тексте
new_text = ["I'm really worried about the shortages in stores during this pandemic."]
probabilities = model.predict_proba(new_text)

# Печать вероятности негативного отзыва
print(f'Вероятность негативного отзыва: {probabilities[0][0]}')  # Вероятность негативного отзыва
print(f'Вероятность положительного отзыва: {probabilities[0][1]}')  # Вероятность положительного отзыва