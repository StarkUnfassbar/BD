import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('spaceship_titanic.csv')
pd.set_option('display.max_columns', None)

# 1. Анализ данных
# print(f"Размерность DataFrame: \n {df.shape}\n")
# print("Информация о DataFrame:")
# print(f"{df.info()} \n")
# print(f"Описание данных: \n {df.describe(include='all')}\n")
# print(f"Количество пустых ячеек в столбцах: \n {df.isnull().sum()}\n")


# 2,3. Замена пропущенных значений, Обработка категориональных признаков
df.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)

df.VIP = df.VIP.replace({'False': 0, 'True': 1}).astype(float)
df.CryoSleep = df.CryoSleep.replace({'False': 0, 'True': 1}).astype(float)
df.Transported = df.Transported.replace({'False': 0, 'True': 1}).astype(float)

names_col_num = ["Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for name in names_col_num:
    df.fillna({name: df[name].median()}, inplace=True)

names_col_categ = ["HomePlanet", "Destination"]
for name in names_col_categ:
    df.fillna({name: df[name].mode()[0]}, inplace=True)

df.loc[
    ((df['RoomService'] == 0.0) | df['RoomService'].isnull()) &
    ((df['FoodCourt'] == 0.0) | df['FoodCourt'].isnull()) &
    ((df['ShoppingMall'] == 0.0) | df['ShoppingMall'].isnull()) &
    ((df['Spa'] == 0.0) | df['Spa'].isnull()) &
    ((df['VRDeck'] == 0.0) | df['VRDeck'].isnull()) &
    (df['CryoSleep'].isnull()),
    'CryoSleep'
] = 1

df.loc[
    ((df['RoomService'] > 0.0) |
     (df['FoodCourt'] > 0.0) |
     (df['ShoppingMall'] > 0.0) |
     (df['Spa'] > 0.0) |
     (df['VRDeck'] > 0.0)) & (df['CryoSleep'].isnull()),
    'CryoSleep'
] = 0

labelencoder = LabelEncoder()
df.HomePlanet = labelencoder.fit_transform(df.HomePlanet)
df.Destination = labelencoder.fit_transform(df.Destination)

correlation_matrix = df.corr()

# 4. Удаление малозначащих данных
df.drop(columns=['ShoppingMall', 'FoodCourt', 'Age', 'VIP'], inplace=True)

# 5. Отделение целевой функции от датасета
y = df['Transported']
X = df.drop(columns=['Transported'])

# 6. Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# --- 1. Random Forest с Grid Search ---
# Определяем диапазоны гиперпараметров для Grid Search
param_grid_rf = {
    'n_estimators': [100, 150, 200],       # Количество деревьев в лесу
    'max_depth': [10, 20, 30],             # Максимальная глубина дерева
    'min_samples_split': [2, 5, 10],       # Минимальное количество образцов для разделения узла
    'min_samples_leaf': [1, 2, 4]          # Минимальное количество образцов в листе
}

print("Grid Search for Random Forest")
# Инициализация модели Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Запуск измерения времени
start_time = time.time()
# Инициализация GridSearchCV для поиска лучших параметров
grid_search_rf = GridSearchCV(estimator=rf_model,
                               param_grid=param_grid_rf,
                               cv=5,  # Количество фолдов для кросс-валидации
                               scoring='accuracy',  # Метрика, по которой будет оцениваться модель
                               n_jobs=-1)  # Использовать все доступные ядра
# Обучение модели с Grid Search
grid_search_rf.fit(X_train_scaled, y_train)
# Общее время поиска
grid_search_time_rf = time.time() - start_time

# Получение лучшей модели и ее параметров
rf_best_model = grid_search_rf.best_estimator_
rf_best_params = grid_search_rf.best_params_
print(f"Best Parameters for Random Forest: {rf_best_params}")

# Оценка модели Random Forest
y_pred_rf = rf_best_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)  # Точность модели
f1_rf = f1_score(y_test, y_pred_rf)  # F1 Score модели
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")  # Вывод точности
print(f"Random Forest F1 Score: {f1_rf:.4f}")  # Вывод F1 Score
print(f"Grid Search Time: {grid_search_time_rf:.2f} seconds\n")  # Время выполнения

# --- 2. Random Forest с Randomized Search ---
# Определяем распределения гиперпараметров для Randomized Search
param_distributions_rf = {
    'n_estimators': [int(x) for x in np.linspace(100, 200, num=10)],  # Линейно распределенные значения количества деревьев
    'max_depth': [int(x) for x in np.linspace(10, 30, num=5)] + [None],  # Глубина деревьев от 10 до 30 и None
    'min_samples_split': [2, 5, 10],   # Минимальное количество образцов для разделения узла
    'min_samples_leaf': [1, 2, 4],      # Минимальное количество образцов в листе
    'bootstrap': [True, False]           # Использование bootstrap
}

print("Randomized Search for Random Forest")
# Запуск измерения времени
start_time = time.time()
# Инициализация RandomizedSearchCV для поиска параметров
random_search_rf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions_rf,
    n_iter=50,  # Количество случайных комбинаций гиперпараметров
    cv=5,  # Кросс-валидация
    scoring='accuracy',  # Метрика, по которой будет оцениваться модель
    random_state=42,  # Фиксация случайного состояния для воспроизводимости
    n_jobs=-1  # Использовать все доступные ядра
)
# Обучение модели с Randomized Search
random_search_rf.fit(X_train_scaled, y_train)
# Общее время поиска
random_search_time_rf = time.time() - start_time

# Получение лучшей модели и ее параметров
rf_best_model_random = random_search_rf.best_estimator_
rf_best_params_random = random_search_rf.best_params_
print(f"Best Parameters for Random Forest (Randomized Search): {rf_best_params_random}")

# Оценка модели Random Forest после Randomized Search
y_pred_rf_random = rf_best_model_random.predict(X_test_scaled)
accuracy_rf_random = accuracy_score(y_test, y_pred_rf_random)  # Точность модели
f1_rf_random = f1_score(y_test, y_pred_rf_random)  # F1 Score модели
print(f"Random Forest Accuracy (Randomized Search): {accuracy_rf_random:.4f}")  # Вывод точности
print(f"Random Forest F1 Score (Randomized Search): {f1_rf_random:.4f}")  # Вывод F1 Score
print(f"Randomized Search Time: {random_search_time_rf:.2f} seconds\n")  # Время выполнения

# --- 3. Random Forest с BayesSearchCV ---
# Определяем параметры для поиска
param_space_rf = {
    'n_estimators': (50, 300),             # Диапазон количества деревьев
    'max_depth': (10, 50),                 # Диапазон глубины деревьев
    'min_samples_split': (2, 20),          # Минимальное количество примеров для разделения узла
    'min_samples_leaf': (1, 20),           # Минимальное количество образцов в листе
    'bootstrap': [True, False]              # Использование bootstrap
}

# Инициализация модели Random Forest
rf_model_bayes = RandomForestClassifier(random_state=42)

print("Bayesian Search for Random Forest")
# Запуск измерения времени
start_time = time.time()
# Инициализация BayesSearchCV для Байесовского поиска
bayes_search_rf = BayesSearchCV(
    estimator=rf_model_bayes,
    search_spaces=param_space_rf,
    n_iter=50,  # Количество итераций Байесовского поиска
    cv=5,  # Кросс-валидация
    scoring='accuracy',  # Метрика оптимизации
    n_jobs=-1,  # Параллелизация
    random_state=42  # Случайное состояние для воспроизводимости
)

# Обучение с Байесовским поиском
bayes_search_rf.fit(X_train_scaled, y_train)
# Общее время поиска
bayes_search_time_rf = time.time() - start_time

# Получение лучшей модели и ее параметров
rf_best_model_bayes = bayes_search_rf.best_estimator_
rf_best_params_bayes = bayes_search_rf.best_params_
print(f"Best Parameters for Random Forest (Bayesian Search): {rf_best_params_bayes}")

# Оценка модели после Байесовского поиска
y_pred_rf_bayes = rf_best_model_bayes.predict(X_test_scaled)
accuracy_rf_bayes = accuracy_score(y_test, y_pred_rf_bayes)  # Точность модели
f1_rf_bayes = f1_score(y_test, y_pred_rf_bayes)  # F1 Score модели
print(f"Random Forest Accuracy (Bayesian Search): {accuracy_rf_bayes:.4f}")  # Вывод точности
print(f"Random Forest F1 Score (Bayesian Search): {f1_rf_bayes:.4f}")  # Вывод F1 Score
print(f"Bayesian Search Time: {bayes_search_time_rf:.2f} seconds\n")  # Время выполнения

# --- Итоги ---
# Сравнение методов
print("Results:")
print(f"Grid Search Accuracy: {accuracy_rf:.4f}, F1 Score: {f1_rf:.4f}, Time: {grid_search_time_rf:.2f} seconds")
print(f"Randomized Search Accuracy: {accuracy_rf_random:.4f}, F1 Score: {f1_rf_random:.4f}, Time: {random_search_time_rf:.2f} seconds")
print(f"Bayesian Search Accuracy: {accuracy_rf_bayes:.4f}, F1 Score: {f1_rf_bayes:.4f}, Time: {bayes_search_time_rf:.2f} seconds")

# Итоги на моем ноуте:
# Grid Search Accuracy: 0.7729, F1 Score: 0.7898, Time: 210.90 seconds
# Randomized Search Accuracy: 0.7734, F1 Score: 0.7904, Time: 146.29 seconds
# Bayesian Search Accuracy: 0.7740, F1 Score: 0.7908, Time: 343.49 seconds

# Я считаю, что метод Randomized Search самый оптимальный так как имеет большое преимущество
# по скорости, при этом не сильно теряя в точности/качестве, находясь на 2 месте. Но если точность
# важнее, то метод Bayesian Search будет лучшим вариантом