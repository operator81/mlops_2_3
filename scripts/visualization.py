import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# Загрузка данных
df = pd.read_csv('data/train.csv')

# Объединим все тексты в одну строку
text = ' '.join(df['Text'])

# Генерируем облако слов

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Показать облако слов
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Убираем оси
plt.show()