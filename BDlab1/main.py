import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

input_folder = 'neg'
output_folder = 'new'

# Функция для удаления всех файлов из папки
def clear_output_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


# Создание папки new, если она не существует
os.makedirs(output_folder, exist_ok=True)
clear_output_folder(output_folder)

# Инициализация лемматизатора и стоп-слов
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление лишних пробелов и других символов, кроме букв и точек
    text = re.sub(r'[^a-z\s.]', '', text)
    # Разбиение текста на слова
    words = text.split()
    # Удаление стоп-слов и лемматизация
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Список для хранения обработанных документов
documents = []

# Обработка каждого файла в папке neg
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Предобработка текста
            processed_text = preprocess_text(text)
            # Запись обработанного текста в новый файл
            new_file_path = os.path.join(output_folder, filename)
            with open(new_file_path, 'w', encoding='utf-8') as new_file:
                new_file.write(processed_text)
            # Добавление обработанного текста в список документов
            documents.append(processed_text)

# Создание модели TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Пример вывода TF-IDF значений
print(tfidf_matrix.toarray())
print(vectorizer.get_feature_names_out())