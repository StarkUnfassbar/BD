import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

images_path = 'nails_segmentation/images'
masks_path = 'nails_segmentation/labels'


# Функция для загрузки данных
def load_data(images_path, masks_path):
    images = []
    masks = []
    for filename in os.listdir(images_path):
        img = cv2.imread(os.path.join(images_path, filename))
        img = cv2.resize(img, (128, 128))  # Изменение размера до желаемого размера
        images.append(img)

        mask = cv2.imread(os.path.join(masks_path, filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128))
        masks.append(mask)

    return images, masks


# Загрузка данных
images, masks = load_data(images_path, masks_path)

# Преобразование списков в тензоры
images = tf.convert_to_tensor(images, dtype=tf.float32) / 255.0  # Нормализация изображений
masks = tf.convert_to_tensor(masks, dtype=tf.float32)
masks = tf.where(masks > 0, 1.0, 0.0)  # Бинаризация масок

# Добавление нового измерения для каналов
masks = tf.expand_dims(masks, axis=-1)

# Разделение данных на обучающие и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(images.numpy(), masks.numpy(), test_size=0.2, random_state=42)


# Полносвязная нейросеть
# Определяем функцию для создания полносвязной нейросети
def create_fully_connected_model(input_shape):
    # Создаем последовательную модель
    model = keras.Sequential([
        # Плоский слой (Flatten): преобразует входной тензор в одномерный массив (вектор)
        keras.layers.Flatten(input_shape=input_shape),

        # Полносвязный слой (Dense): 128 нейронов с активацией ReLU
        keras.layers.Dense(128, activation='relu'),

        # Полносвязный слой: 128*128 нейронов с сигмоидальной активацией
        # Выходной слой, создающий 128*128 значений, актуально для задачи сегментации
        keras.layers.Dense(128 * 128, activation='sigmoid'),  # Измените размер в соответствии с вашими данными

        # Изменяем форму выходного вектора обратно в форму изображения (128, 128, 1)
        keras.layers.Reshape((128, 128, 1))
    ])

    # Компилируем модель с оптимизатором Adam и бинарной кроссентропий как функцией потерь
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Создаем модель, передавая стартовую форму входных данных (128, 128, 3)
fc_model = create_fully_connected_model((128, 128, 3))
# Выводим информацию о модели, включая количество параметров
fc_model.summary()

# Обучение модели
fc_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

# Оценка модели с использованием метрик
y_pred = (fc_model.predict(X_test) > 0.5).astype(int)  # Прогнозы с порогом 0.5

# Вычисление метрик
precision = precision_score(y_test.flatten(), y_pred.flatten(), average='binary')
recall = recall_score(y_test.flatten(), y_pred.flatten(), average='binary')
f1 = f1_score(y_test.flatten(), y_pred.flatten(), average='binary')

# Вывод метрик
print(f'Accuracy: {fc_model.evaluate(X_test, y_test, verbose=0)[1]}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# Определяем функцию для создания модели U-Net
def create_unet(input_shape):
    # Входной слой принимающий изображения заданной формы
    inputs = keras.layers.Input(input_shape)

    # Этап сжатия (входной путь)
    # Первый блок сверточных слоев с 32 фильтрами
    conv1 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    conv1 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(conv1)
    # Субдискретизация (пулинг) для уменьшения размерности
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Второй блок сверточных слоев с 64 фильтрами
    conv2 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(pool1)
    conv2 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(conv2)
    # Субдискретизация
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Третий блок сверточных слоев с 128 фильтрами
    conv3 = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(pool2)
    conv3 = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(conv3)
    # Субдискретизация
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Боттлнек (наиболее глубокая часть сети)
    # Четвертый блок сверточных слоев с 256 фильтрами
    conv4 = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(pool3)
    conv4 = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(conv4)

    # Этап расширения (выходной путь)
    # Подъем размерности и объединение с соответствующим блоком из этапа сжатия
    up5 = keras.layers.Concatenate()([keras.layers.UpSampling2D(size=(2, 2))(conv4), conv3])
    # Сверточные слои
    conv5 = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(up5)
    conv5 = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(conv5)

    # Подъем размерности и объединение с соответствующим блоком из этапа сжатия
    up6 = keras.layers.Concatenate()([keras.layers.UpSampling2D(size=(2, 2))(conv5), conv2])
    # Сверточные слои
    conv6 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(up6)
    conv6 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(conv6)

    # Подъем размерности и объединение с соответствующим блоком из этапа сжатия
    up7 = keras.layers.Concatenate()([keras.layers.UpSampling2D(size=(2, 2))(conv6), conv1])
    # Сверточные слои
    conv7 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(up7)
    conv7 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(conv7)

    # Выходной слой: одно сверточное преобразование для генерации маски сегментации
    outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv7)  # Выходной слой

    # Создание модели U-Net с указанными входами и выходами
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# Создание модели
model = create_unet((128, 128, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train.reshape(-1, 128, 128, 1), epochs=10, batch_size=16, validation_split=0.1)

# Прогнозирование на тестовом наборе
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(np.uint8)  # Бинаризация предсказаний

# Оценка модели с использованием метрик
precision = precision_score(y_test.flatten(), y_pred_binary.flatten(), average='binary')
recall = recall_score(y_test.flatten(), y_pred_binary.flatten(), average='binary')
f1 = f1_score(y_test.flatten(), y_pred_binary.flatten(), average='binary')

# Вывод метрик
print(f'Accuracy: {model.evaluate(X_test, y_test.reshape(-1, 128, 128, 1), verbose=0)[1]}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')