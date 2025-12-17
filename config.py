from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """
    Настройки модели и визуализации.
    Меняем числа здесь, логика приложения не трогается.
    """

    # сколько слов задаём в одном "шаге" бинарного поиска
    bs_batch_size: int = 5

    # максимальное количество вопросов пользователю
    max_questions: int = 50

    # размер текстового сегмента для кривой сложности (по токенам)
    segment_token_size: int = 300

    # окно сглаживания (по количеству сегментов)
    smoothing_window: int = 3

    # уровни комфорта по доле незнакомых слов
    # <= 2% (98%+ знакомых) — комфортное чтение
    comfort_green_max_unknown: float = 0.02
    # 2–5% (95–98%) — можно читать, но тяжело
    comfort_yellow_max_unknown: float = 0.05


CONFIG = AppConfig()
