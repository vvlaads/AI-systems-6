from enum import Enum


class OptimizationType(Enum):
    """Методы оптимизации"""
    GRADIENT_DESCENT = 0,
    NEWTON_METHOD = 1
