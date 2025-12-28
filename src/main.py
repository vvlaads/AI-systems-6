import pandas as pd
import matplotlib.pyplot as plt
from config import get_resource_path

# Настройка, чтобы показывались все колонки dataframe
pd.set_option('display.max_columns', None)


def print_sep():
    """Вывести разделитель в консоль"""
    print('=' * 50)


def read_csv_file(filename):
    """Прочесть CSV файл с указанным названием"""
    filepath = get_resource_path(filename)
    return pd.read_csv(filepath)


# === Считываем датафреймы ===
train_df = read_csv_file("train.csv")
test_df = read_csv_file("test.csv")
test_result_df = read_csv_file("gender_submission.csv")

print_sep()
print("До обработки:")
print_sep()
print("Тренировочная выборка:")
print(train_df.info())
print_sep()
print("Тестовая выборка:")
print(test_df.info())
print_sep()

# === Предварительная обработка данных ===
# --- Заполняем пропуски ---
# Age - заполняем медианой
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# Embarked - заполняем модой
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Fare (в тестовых данных есть 1 пропуск) - заполняем медианой
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# --- Нормировка категориальных признаков (кодирование) ---
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

train_df['Embarked'] = train_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_df['Embarked'] = test_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# --- Удаляем ненужные столбцы ---
columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
train_df = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns])
test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])

print("После обработки:")
print_sep()
print("Тренировочная выборка:")
print(train_df.info())
print_sep()
print("Тестовая выборка:")
print(test_df.info())
print_sep()

print("Статистика (тренировочная выборка):")
print(train_df.describe())
print_sep()

# Гистограммы
train_df.hist(bins=30,
              figsize=(10, 5),
              edgecolor='k',
              layout=(2, 4))
plt.suptitle('Гистограммы распределения признаков', fontsize=14)
plt.tight_layout()

# Box plot
train_df.plot(kind='box', subplots=True, layout=(2, 4), figsize=(10, 5),
              showfliers=False,  # Не отображать выбросы
              showmeans=True,  # Показывать среднее значение
              meanline=True,  # Отображать среднее значение линией
              meanprops={'linestyle': '--', 'linewidth': 2, 'color': 'red'})
plt.suptitle('Box-plots', fontsize=14)
plt.tight_layout()

# === Логическая регрессия ===
# TODO реализовать метод

# Вывод графиков
plt.show()
