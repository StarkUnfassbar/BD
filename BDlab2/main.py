import pandas as pd

pd.set_option('display.max_rows', None)

# Чтение таблицы из Excel файла
data = pd.read_csv('data.csv')
data = data.iloc[:, 1:]
data.iloc[:, :5] = data.iloc[:, :5].replace({r'[\s,]': ''}, regex=True).apply(pd.to_numeric, errors='coerce')
print(data.to_string(index=False, header=False), "\n")

# Заполнение пустых значений
for col in data.columns:
    if col == "Округ":  # Если столбец с текстом
        data[col].fillna(data[col].mode()[0], inplace=True)  # Заполнение модой
    else:  # Если столбец с числовыми значениями
        data[col].fillna(data[col].median(), inplace=True)  # Заполнение медианой

# Функция для нахождения выбросов и их удаления
def remove_outliers(column):
    Q1, Q3 = column.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    # Удаляем выбросы
    filtered_column = column[(column >= lower_bound) & (column <= upper_bound)]
    return filtered_column

for col in data.columns[:5]:
    # Замена выбросов на медиану
    data[col] = remove_outliers(data[col])
    data[col].fillna(data[col].median(), inplace=True)

print("Очистка данных завершена.")
print(data)
