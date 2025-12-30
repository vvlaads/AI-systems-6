import pandas as pd
import matplotlib.pyplot as plt

from optimization_type import OptimizationType
from util import *
from config import get_resource_path
from logistic import LogisticRegression

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

# --- Кодирование категориальных признаков ---
train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix='Embarked')
test_df = pd.get_dummies(test_df, columns=['Embarked'], prefix='Embarked')

train_df = pd.get_dummies(train_df, columns=['Sex'], prefix='Sex')
test_df = pd.get_dummies(test_df, columns=['Sex'], prefix='Sex')

train_df = pd.get_dummies(train_df, columns=['Pclass'], prefix='Class')
test_df = pd.get_dummies(test_df, columns=['Pclass'], prefix='Class')

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
              layout=(2, 3))
plt.suptitle('Гистограммы распределения признаков', fontsize=14)
plt.tight_layout()

# Box plot
train_df.plot(kind='box', subplots=True, layout=(2, 3), figsize=(10, 5),
              showfliers=True,  # Не отображать выбросы
              showmeans=True,  # Показывать среднее значение
              meanline=True,  # Отображать среднее значение линией
              meanprops={'linestyle': '--', 'linewidth': 2, 'color': 'red'})
plt.suptitle('Box-plots', fontsize=14)
plt.tight_layout()

# === Логическая регрессия ===
# --- Подготовка данных ---
# Нормировка значений
numeric_columns = ['Age', 'Fare', 'SibSp', 'Parch']
train_df, test_df = min_max_scale_with_params(train_df, test_df, numeric_columns)

# Сортируем колонки в одинаковом порядке
test_df = test_df[train_df.columns.drop('Survived')]

# Разделяем на X и y
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df
y_test = test_result_df['Survived']

# --- Исследование влияния гиперпараметров на производительность модели ---
# Градиентный спуск, шаг 0.01, итераций 100
logistic_regression = LogisticRegression(learning_rate=0.01, n_iterations=100)
logistic_regression.fit(X_train, y_train)
y_predicted = logistic_regression.predict(X_test)
tp, tn, fp, fn = test_values(y_predicted, y_test)
accuracy, precision, recall, f1 = get_model_evaluation(tp, tn, fp, fn)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

print_sep()

# Градиентный спуск, шаг 1, итераций 100
logistic_regression = LogisticRegression(learning_rate=1, n_iterations=100)
logistic_regression.fit(X_train, y_train)
y_predicted = logistic_regression.predict(X_test)
tp, tn, fp, fn = test_values(y_predicted, y_test)
accuracy, precision, recall, f1 = get_model_evaluation(tp, tn, fp, fn)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

print_sep()

# Градиентный спуск, шаг 0.1, итераций 100
logistic_regression = LogisticRegression(learning_rate=0.1, n_iterations=100)
logistic_regression.fit(X_train, y_train)
y_predicted = logistic_regression.predict(X_test)
tp, tn, fp, fn = test_values(y_predicted, y_test)
accuracy, precision, recall, f1 = get_model_evaluation(tp, tn, fp, fn)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

print_sep()

# Градиентный спуск, шаг 0.1, итераций 1000
logistic_regression = LogisticRegression(learning_rate=0.1, n_iterations=1000)
logistic_regression.fit(X_train, y_train)
y_predicted = logistic_regression.predict(X_test)
tp, tn, fp, fn = test_values(y_predicted, y_test)
accuracy, precision, recall, f1 = get_model_evaluation(tp, tn, fp, fn)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

print_sep()

# Градиентный спуск, шаг 0.1, итераций 10000
logistic_regression = LogisticRegression(learning_rate=0.1, n_iterations=10000)
logistic_regression.fit(X_train, y_train)
y_predicted = logistic_regression.predict(X_test)
tp, tn, fp, fn = test_values(y_predicted, y_test)
accuracy, precision, recall, f1 = get_model_evaluation(tp, tn, fp, fn)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

print_sep()

# Метод Ньютона, итераций 100
logistic_regression = LogisticRegression(opt_type=OptimizationType.NEWTON_METHOD, n_iterations=100)
logistic_regression.fit(X_train, y_train)
y_predicted = logistic_regression.predict(X_test)
tp, tn, fp, fn = test_values(y_predicted, y_test)
accuracy, precision, recall, f1 = get_model_evaluation(tp, tn, fp, fn)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

print_sep()

# Метод Ньютона, итераций 1000
logistic_regression = LogisticRegression(opt_type=OptimizationType.NEWTON_METHOD, n_iterations=1000)
logistic_regression.fit(X_train, y_train)
y_predicted = logistic_regression.predict(X_test)
tp, tn, fp, fn = test_values(y_predicted, y_test)
accuracy, precision, recall, f1 = get_model_evaluation(tp, tn, fp, fn)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

# Вывод графиков
plt.show()
