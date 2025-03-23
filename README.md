Установка зависимостей

1. Клонируйте репозиторий:
git clone https://your-repository-url.git
cd your-repository

2. Установите зависимости с помощью pip:

Создайте виртуальное окружение (если необходимо):

python -m venv venv
source venv/bin/activate  # Для Linux/Mac
venv\Scripts\activate  # Для Windows

3. Установите все необходимые библиотеки:

pip install -r requirements.txt

В файле requirements.txt должны быть следующие зависимости:

transformers==4.24.0
torch==1.12.0
datasets==2.10.1
pandas==1.4.2
scikit-learn==1.0.2
kaggle==1.5.12

4. Если у вас еще не установлен kagglehub, установите его:

pip install kagglehub
