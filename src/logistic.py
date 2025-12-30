import numpy as np

from optimization_type import OptimizationType


def sigmoid(x):
    """Функция сигмоиды"""
    # Гарантируем, что это numpy array
    x = np.asarray(x, dtype=float)

    # Защита от переполнения
    x = np.clip(x, -500, 500)

    return 1 / (1 + np.exp(-x))


def log_loss(y_true, y_predicted_proba):
    """Функция потерь log loss"""
    n = len(y_true)
    result = 0
    for i in range(n):
        result += y_true[i] * np.log(y_predicted_proba[i]) + (1 - y_true[i]) * np.log(1 - y_predicted_proba[i])
    result *= -1 / n
    return result


class LogisticRegression:
    """Класс для логистической регрессии"""

    def __init__(self, learning_rate=0.01, n_iterations=1000, opt_type=OptimizationType.GRADIENT_DESCENT):
        """Конструктор"""
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.opt_type = opt_type

    def fit(self, x_train, y_train):
        """Обучение модели градиентным спуском"""
        x_train = np.array(x_train, dtype=float)
        y_train = np.array(y_train, dtype=float)

        n_samples, n_features = x_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        print(f"Начинаем обучение методом: {self.opt_type.name}")
        print(f"Скорость обучения: {self.learning_rate} ")
        print(f"Количество итераций: {self.n_iterations}")

        # Выбор метода оптимизации
        if self.opt_type == OptimizationType.GRADIENT_DESCENT:
            self._fit_gradient_descent(x_train, y_train, n_samples)
        elif self.opt_type == OptimizationType.NEWTON_METHOD:
            self._fit_newton_method(x_train, y_train, n_samples)
        else:
            raise ValueError(f"Неизвестный метод оптимизации: {self.opt_type}")

    def _fit_gradient_descent(self, x_train, y_train, n_samples):
        """Обучение градиентным спуском"""
        print_step = self.n_iterations / 5
        for i in range(1, self.n_iterations + 1):
            # Вычисляем линейную комбинацию
            linear_output = np.dot(x_train, self.weights) + self.bias

            # Применяем сигмоиду для получения вероятностей
            y_predicted_proba = sigmoid(linear_output)

            # Вычисление функции потерь
            loss = log_loss(y_train, y_predicted_proba)
            self.loss_history.append(loss)

            # Вычисление градиентов
            error = y_predicted_proba - y_train

            # Вычисляем градиенты (производные)
            dw = (1 / n_samples) * np.dot(x_train.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Обновление параметров
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % print_step == 0:
                print(f"  Итерация {i:4d}, Loss: {loss:.4f}")

        print(f"  Градиентный спуск завершен")

    def _fit_newton_method(self, x_train, y_train, n_samples):
        """Обучение методом Ньютона"""
        # Добавляем столбец единиц для bias
        x = np.hstack([x_train, np.ones((n_samples, 1))])

        # Инициализируем все параметры вместе
        theta = np.zeros(x.shape[1])  # [w1, w2, ..., wn, bias]

        print_step = self.n_iterations / 5
        for i in range(1, self.n_iterations + 1):
            # Вычисляем линейную комбинацию
            linear_output = np.dot(x, theta)

            # Применяем сигмоиду для получения вероятностей
            y_predicted_proba = sigmoid(linear_output)

            # Вычисление функции потерь
            loss = log_loss(y_train, y_predicted_proba)
            self.loss_history.append(loss)

            # Градиент (первая производная)
            # Формула: градиент = X^T * (p - y) / n
            gradient = np.dot(x.T, (y_predicted_proba - y_train)) / n_samples

            # Гессиан (вторая производная, матрица)
            # Формула: гессиан = X^T * D * X / n
            # где D = diag(p * (1 - p)) - диагональная матрица
            w = y_predicted_proba * (1 - y_predicted_proba)  # диагональные элементы

            # Быстрый расчет: X^T * diag(W) * X
            hessian = np.dot(x.T * w, x) / n_samples

            # Метод Ньютона: theta_new = theta_old - H^(-1) * градиент
            # Решаем: H * delta = градиент
            delta = np.linalg.solve(hessian, gradient)
            theta -= delta

            # Разделяем обратно на weights и bias
            self.weights = theta[:-1]  # все кроме последнего
            self.bias = theta[-1]  # последний - это bias

            if i % print_step == 0:
                print(f"  Итерация {i:4d}, Loss: {loss:.4f}")

            # Останавливаемся, если loss почти не меняется
            if i > 1 and abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-9:
                print(f"  Сошлось на итерации {i}")
                print(f"  Loss: {loss:.4f}")
                break

        print(f"  Метод Ньютона завершен")

    def predict_proba(self, x_test):
        """Предсказание вероятности принадлежности к классу 1"""

        # Линейная комбинация
        linear_output = np.dot(x_test, self.weights) + self.bias

        # Применяем сигмоиду
        return sigmoid(linear_output)

    def predict(self, x_test, threshold=0.5):
        """Предсказание классов (0 или 1)"""
        probabilities = self.predict_proba(x_test)

        return (probabilities >= threshold).astype(int)
