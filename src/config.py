from pathlib import Path

# Получаем корневую директорию проекта
PROJECT_ROOT = Path(__file__).parent.parent

# Пути к ресурсам
RESOURCES_DIR = PROJECT_ROOT / "resources"


# Функция для получения пути к ресурсам
def get_resource_path(filename):
    """Получить путь к файлу ресурса"""
    return str(RESOURCES_DIR / filename)
