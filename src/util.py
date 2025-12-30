import numpy as np


def min_max_scale_with_params(train_df, test_df, columns):
    """Min-Max масштабирование с сохранением параметров"""
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    for col in columns:
        min_train = train_df[col].min()
        max_train = train_df[col].max()

        if max_train - min_train != 0:
            train_scaled[col] = (train_df[col] - min_train) / (max_train - min_train)
            test_scaled[col] = (test_df[col] - min_train) / (max_train - min_train)
        else:
            train_scaled[col] = 0
            test_scaled[col] = 0

    return train_scaled, test_scaled


def test_values(predictions, y_series):
    """Сопоставление предсказанных значений с реальными"""
    tp, tn, fp, fn = 0, 0, 0, 0
    n = len(predictions)
    y_values = np.array(y_series, dtype=float)
    for i in range(n):
        if y_values[i] == 1 and predictions[i] == 1:
            tp += 1
        elif y_values[i] == 0 and predictions[i] == 0:
            tn += 1
        elif y_values[i] == 1 and predictions[i] == 0:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


def get_model_evaluation(tp, tn, fp, fn):
    """Оценка модели"""
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1
