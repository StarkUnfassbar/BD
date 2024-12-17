import os
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2

def create_output_directory(output_dir):
    # Создание папки для сохранения выходных изображений, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def clear_output_directory(output_dir):
    # Очистка папки от существующих файлов
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            # Удаление файла
            if os.path.isfile(file_path):
                os.remove(file_path)

def process_image(image_path):
    # Папка для сохранения обработанных изображений
    output_dir = 'img/output'
    create_output_directory(output_dir)
    clear_output_directory(output_dir)

    # Открываем изображение
    img = Image.open(image_path)

    # 1. Повернуть картинку
    img_rotated = img.rotate(90)  # Поворот на 90 градусов
    img_rotated.save(os.path.join(output_dir, 'img1.png'))

    # 2. Изменить размер
    img_resized = img_rotated.resize((800, 600))  # Изменение размера до 800x600
    img_resized.save(os.path.join(output_dir, 'img2.png'))

    # 3. Изменить цветовую палитру на чёрно-белую
    img_bw = img_resized.convert('L')  # Преобразование в черно-белый режим
    img_bw.save(os.path.join(output_dir, 'img3.png'))

    # 4. Гауссово размытие и билатеральное размытие
    img_gaussian = img_bw.filter(ImageFilter.GaussianBlur(5))  # Гауссово размытие
    img_gaussian.save(os.path.join(output_dir, 'img4_gaussian.png'))

    # Конвертация в массив numpy для более сложной обработки
    img_array = np.array(img_gaussian)

    # Билатеральное размытие (используя OpenCV)
    img_bilateral = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)

    # Преобразуем обратно в изображение Pillow
    img_bilateral = Image.fromarray(img_bilateral)
    img_bilateral.save(os.path.join(output_dir, 'img4_bilateral.png'))

    # 5. Добавить рамку
    img_with_border = ImageOps.expand(img_bilateral, border=10, fill='black')  # Черная рамка
    img_with_border.save(os.path.join(output_dir, 'img5.png'))

    # 6. Наложение картинки (например, полупрозрачный квадрат)
    overlay = Image.new('RGBA', img_with_border.size, (255, 0, 0, 128))  # Красный полупрозрачный квадрат
    img_final = Image.alpha_composite(img_with_border.convert('RGBA'), overlay)
    img_final.save(os.path.join(output_dir, 'img6_final.png'))

    print(f'Все изображения успешно обработаны и сохранены в {output_dir}')

# Путь к исходному изображению
image_path = 'img/img0.jpg'
process_image(image_path)