FROM python:3.11-slim

WORKDIR /app

COPY . .

# Устанавливаем необходимые зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем скрипты в нужном порядке
CMD ["bash", "-c", "python /scripts/data_loading.py && python /scripts/data_preprocessing.py && python /scripts/modeling.py && python /scripts/visualization.py"]