from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from functools import lru_cache, wraps
from enum import Enum, auto

import pandas as pd
import spacy


# +++ Исключения +++
# 1. Использование классов. (1 балла)
# 2. Использование наследования от какого-нибудь класса. (0,5 балла)
class ConfigError(Exception):
    """Ошибка в конфигурации/параметрах."""
    pass


class TextProcessingError(Exception):
    """Ошибка при обработке текста (NLP‑этап)."""
    pass


# +++ Декоратор +++
# 7. Использование собственного декоратора. (0,25 балла)
def log_exceptions(func):
    """
    Ловим любые исключения и упаковываем
    их в TextProcessingError, чтобы UI обрабатывал всё единообразно.
    """
    @wraps(func) # Позваляет сохранить метаданные исходной функции
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TextProcessingError:
            raise
        except Exception as exc:
            raise TextProcessingError(f"{func.__name__} failed: {exc}") from exc
    return wrapper


# +++ Загрузка spaCy (кешируем) +++
# 6. Использование декораторов. (0,25 балла)
@lru_cache(maxsize=1) # Не дает ошибочно грузить тяжелую модель несколько раз и кэширует ее
def get_spacy_model():
    """Грузим модель spaCy"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise TextProcessingError(
            "spaCy model 'en_core_web_sm' is not installed. "
        ) from exc


# +++ Структуры данных для текста и частотника +++

@dataclass(slots=True)
class ProcessedText:
    """
    + Результат препроцессинга + 
    Слоты:
    - tokens: исходные токены
    - lemmas: леммы в нижнем регистре
    - is_proper: флаг собственных имен/названий
    """
    tokens: List[str]
    lemmas: List[str]
    is_proper: List[bool]
    # Постпроверка того, что размеры списков совпадают
    def __post_init__(self) -> None:
        if not (len(self.tokens) == len(self.lemmas) == len(self.is_proper)):
            raise ValueError("Inconsistent lengths in ProcessedText.")

# 8. Перегрузка операторов. (0,5 балла)
class FrequencyDict(dict):
    """
    Мой  частотный словарь с возможностью сложения.
    """

    def __add__(self, other: "FrequencyDict") -> "FrequencyDict":
        if not isinstance(other, FrequencyDict):
            return NotImplemented
        result = FrequencyDict(self)
        for key, value in other.items():
            result[key] = result.get(key, 0) + int(value)
        return result


# +++ Токенизатор / лемматизатор +++
# 4. Использование абстрактных классов в структуре классов. (0.5 балла)
class TextProcessor(ABC):
    """Абстрактный класс (интерфейс) для препроцессинга текста."""

    @abstractmethod
    def process(self, text: str) -> ProcessedText:
        raise NotImplementedError


# 3. Использование наследования от созданного своими руками класса. (0,5 балла)
class SpacyTextProcessor(TextProcessor):
    """
    Реализация TextProcessor на spaCy.
    """

    def __init__(self) -> None:
        self._nlp = get_spacy_model()

    @log_exceptions
    def process(self, text: str) -> ProcessedText:
        doc = self._nlp(text)
        tokens: List[str] = []
        lemmas: List[str] = []
        is_proper: List[bool] = []

        for token in doc:
            # берем только алфавитные токены
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower()
            if not lemma:
                continue

            # собственные имена / названия (не учитываем в частотнике)
            proper = (
                token.pos_ == "PROPN"
                or token.ent_type_ in (
                    "PERSON",
                    "ORG",
                    "GPE",
                    "LOC",
                    "NORP",
                    "FAC",
                    "WORK_OF_ART",
                    "EVENT",
                )
            )

            tokens.append(token.text)
            lemmas.append(lemma)
            is_proper.append(proper)

        if not tokens:
            raise TextProcessingError("Text contains no usable tokens.")
        return ProcessedText(tokens=tokens, lemmas=lemmas, is_proper=is_proper)


@dataclass(slots=True)
class VocabularyItem:
    """Одна лемма в частотном словаре."""
    lemma: str
    freq: int
    rank: int
    is_proper: bool = False


@dataclass
class VocabularyProfile:
    """
    Частотный профиль текста:
    - items: список (лемма, частота, ранг)
    - lemma_to_index: быстрый поиск ранга по лемме
    - total_tokens: количество токенов, учтённых в профиле
      (без собственных имён)
    """
    items: List[VocabularyItem]
    lemma_to_index: Dict[str, int]
    total_tokens: int

    @classmethod
    def from_processed_text(cls, processed: ProcessedText) -> "VocabularyProfile":
        freq = FrequencyDict()
        # total_tokens = 0

        # 1. Подсчитываем частоты всех НЕ собственных имён
        for lemma, proper in zip(processed.lemmas, processed.is_proper):
            if proper:
                continue
            freq[lemma] = freq.get(lemma, 0) + 1
            # total_tokens += 1

        # 2. Убираем гапаксы
        freq = FrequencyDict({lemma: c for lemma, c in freq.items() if c > 1})

        # 3. Теперь вычисляем total_tokens:
        total_tokens = sum(c for c in freq.values())

        # 4. Формируем частотный профиль
        sorted_items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        items: List[VocabularyItem] = []
        lemma_to_index: Dict[str, int] = {}

        for rank, (lemma, count) in enumerate(sorted_items):
            items.append(VocabularyItem(lemma=lemma, freq=count, rank=rank))
            lemma_to_index[lemma] = rank

        return cls(
            items=items,
            lemma_to_index=lemma_to_index,
            total_tokens=total_tokens,
        )


    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> VocabularyItem:
        return self.items[index]

    def rank_of(self, lemma: str) -> Optional[int]:
        return self.lemma_to_index.get(lemma)

    def known_token_share(self, threshold_index: int) -> float:
        """
        Доля токенов, приходящихся на слова с rank <= threshold_index.
        """
        if not self.items or threshold_index < 0:
            return 0.0
        max_index = min(threshold_index, len(self.items) - 1)
        known = sum(item.freq for item in self.items[: max_index + 1])
        return known / max(self.total_tokens, 1)

    def unknown_token_share(self, threshold_index: int) -> float:
        return 1.0 - self.known_token_share(threshold_index)


# +++ Алгоритм тестирования словаря +++

class Stage(Enum):
    # CALIBRATION = auto()
    BINARY_SEARCH = auto()
    FINISHED = auto()


@dataclass(slots=True)
class Question:
    """Один вопрос пользователю: знает/не знает такое слово."""
    vocab_index: int
    lemma: str


class VocabularyTester(ABC):
    """
    Абстрактный интерфейс стратегии тестирования.
    Это и есть Strategy‑паттерн: можно добавить другие стратегии,
    """

    @abstractmethod
    def next_question(self) -> Optional[Question]:
        ...

    @abstractmethod
    def record_answer(self, vocab_index: int, known: bool) -> None:
        ...

    @property
    @abstractmethod
    def is_finished(self) -> bool:
        ...

    @property
    @abstractmethod
    def estimated_threshold_index(self) -> Optional[int]:
        ...

    @property
    @abstractmethod
    def asked_answers(self) -> Dict[int, bool]:
        ...

    @property
    @abstractmethod
    def question_count(self) -> int:
        ...


@dataclass
class BinarySearchVocabularyTester(VocabularyTester):
    """
    Стратегия: только бинарный поиск.
    Без предварительного калибровочного прохода по всему словарю.
    """

    vocab: VocabularyProfile
    batch_size: int
    max_questions: int

    # сразу начинаем с бинарного поиска
    stage: Stage = field(default=Stage.BINARY_SEARCH, init=False)
    _asked_answers: Dict[int, bool] = field(default_factory=dict, init=False)
    _question_count: int = field(default=0, init=False)

    # поля калибровки больше не используются, но оставляем для совместимости
    # _calibration_pos: int = field(default=0, init=False)
    _last_known_index: Optional[int] = field(default=None, init=False)
    _first_unknown_index: Optional[int] = field(default=None, init=False)

    # границы бинарного поиска
    _binary_low: Optional[int] = field(default=None, init=False)
    _binary_high: Optional[int] = field(default=None, init=False)

    _current_batch: List[int] = field(default_factory=list, init=False)
    _current_batch_answers: Dict[int, bool] = field(default_factory=dict, init=False)

    _threshold_index: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be > 0")
        if self.max_questions <= 0:
            raise ConfigError("max_questions must be > 0")

        # начальный интервал для бинарного поиска: весь словарь
        n = len(self.vocab)
        if n == 0:
            # нечего спрашивать
            self._finish_with_threshold(-1)
        else:
            self._binary_low = -1
            self._binary_high = n - 1

    # --- публичные свойства / интерфейс ---

    @property # позволяет обращаться как к свойству класса через .
    def is_finished(self) -> bool:
        return self.stage is Stage.FINISHED

    @property
    def estimated_threshold_index(self) -> Optional[int]:
        return self._threshold_index

    @property
    def asked_answers(self) -> Dict[int, bool]:
        return self._asked_answers

    @property
    def question_count(self) -> int:
        return self._question_count

    @property
    def debug_state(self) -> dict:
        """Краткое состояние для dev-интерфейса."""
        return {
            "stage": self.stage.name,
            "last_known_index": self._last_known_index,
            "first_unknown_index": self._first_unknown_index,
            "binary_low": self._binary_low,
            "binary_high": self._binary_high,
            "current_batch": list(self._current_batch),
        }

    # --- основная логика ---

    def record_answer(self, vocab_index: int, known: bool) -> None:
        if self.is_finished:
            return
        if vocab_index in self._asked_answers:
            # защита от двойных кликов
            return

        self._asked_answers[vocab_index] = bool(known)
        self._question_count += 1

        # просто запоминаем ответы по текущей пачке
        if vocab_index in self._current_batch:
            self._current_batch_answers[vocab_index] = bool(known)

        if self._question_count >= self.max_questions and not self.is_finished:
            # вопросов больше задавать нельзя — завершаем по текущим границам
            self._finish_by_bounds()

    def next_question(self) -> Optional[Question]:
        if self.is_finished:
            return None
        return self._next_binary_search_question()

    # --------- бинарный поиск внутри интервала ---------

    def _next_binary_search_question(self) -> Optional[Question]:
        if self._binary_low is None or self._binary_high is None:
            self._finish_by_bounds()
            return None

        if self._question_count >= self.max_questions:
            self._finish_by_bounds()
            return None

        # 1. Если есть незавершённый пак, продолжаем его
        if self._current_batch:
            for idx in self._current_batch:
                if idx not in self._current_batch_answers:
                    return Question(vocab_index=idx, lemma=self.vocab[idx].lemma)
            # все вопросы в паке заданы — обновляем границы
            self._consume_batch_and_update_bounds()
            if self.is_finished:
                return None

        # 2. Проверяем, не слишком ли мал интервал
        if self._binary_high - self._binary_low <= 1:
            self._finish_by_bounds()
            return None

        # 3. Строим новый пак вокруг середины по 0.5 batch_size по обе стороны
        middle = (self._binary_low + self._binary_high) // 2
        half = max(1, self.batch_size // 2)
        start = max(self._binary_low + 1, middle - half)
        end = min(self._binary_high, middle + half + 1)

        batch = [idx for idx in range(start, end) if idx not in self._asked_answers]
        if not batch:
            self._finish_by_bounds()
            return None

        self._current_batch = batch
        self._current_batch_answers = {}
        first_idx = self._current_batch[0]
        return Question(vocab_index=first_idx, lemma=self.vocab[first_idx].lemma)

    def _consume_batch_and_update_bounds(self) -> None:
        if not self._current_batch:
            return

        yes = sum(1 for idx in self._current_batch if self._asked_answers.get(idx, False))
        no = sum(1 for idx in self._current_batch if self._asked_answers.get(idx, False) is False)

        if yes + no == 0:
            self._current_batch = []
            self._current_batch_answers = {}
            return

        if yes >= no:
            # пачка в основном "знаю" — поднимаем нижнюю границу
            self._binary_low = max(self._binary_low or 0, max(self._current_batch))
        else:
            # пачка в основном "не знаю" — опускаем верхнюю
            self._binary_high = min(self._binary_high or len(self.vocab), min(self._current_batch))

        self._current_batch = []
        self._current_batch_answers = {}

        if self._binary_low is None or self._binary_high is None:
            return

        if self._binary_high - self._binary_low <= 1 or self._question_count >= self.max_questions:
            self._finish_by_bounds()

    # --------- завершение ---------

    def _finish_by_bounds(self) -> None:
        if self.stage is Stage.FINISHED:
            return

        n = len(self.vocab)
        if self._binary_low is not None and self._binary_high is not None:
            est = (self._binary_low + self._binary_high) // 2
        elif self._last_known_index is not None and self._first_unknown_index is not None:
            est = (self._last_known_index + self._first_unknown_index) // 2
        elif self._last_known_index is not None:
            est = self._last_known_index
        elif self._first_unknown_index is not None:
            est = self._first_unknown_index - 1
        else:
            est = n // 2 if n else -1

        self._finish_with_threshold(est)

    def _finish_with_threshold(self, index: int) -> None:
        n = len(self.vocab)
        if n == 0:
            self._threshold_index = -1
        else:
            index = max(-1, min(index, n - 1))
            self._threshold_index = index
        self.stage = Stage.FINISHED
        self._current_batch.clear()
        self._current_batch_answers.clear()



# ====== Кривая сложности по ходу текста ======

def build_difficulty_curve(
    processed: ProcessedText,
    vocab: VocabularyProfile,
    threshold_index: int,
    segment_size: int,
    smoothing_window: int,
) -> pd.DataFrame:
    """
    Строим DataFrame с оценкой доли незнакомых слов по сегментам текста.
    Столбцы:
      segment, start, end, position_frac, unknown_ratio, unknown_ratio_smooth
    """
    if segment_size <= 0:
        raise ConfigError("segment_size must be > 0")

    tokens = processed.tokens
    lemmas = processed.lemmas
    is_proper = processed.is_proper
    total_tokens = len(tokens)

    rows: List[Dict[str, float]] = []

    for start in range(0, total_tokens, segment_size):
        end = min(start + segment_size, total_tokens)
        non_proper = 0
        unknown = 0

        for lemma, proper in zip(lemmas[start:end], is_proper[start:end]):
            if proper:
                continue

            rank = vocab.rank_of(lemma)

            # гапакс или любое слово, выброшенное из словаря: полностью игнорируем
            if rank is None:
                continue

            non_proper += 1

            # неизвестное слово — если его ранг выше порога
            if rank > threshold_index:
                unknown += 1


        unknown_ratio = (unknown / non_proper) if non_proper else 0.0
        center = start + (end - start) / 2
        position_frac = center / max(total_tokens, 1)

        rows.append(
            dict(
                segment=len(rows),
                start=start,
                end=end,
                position_frac=position_frac,
                unknown_ratio=unknown_ratio,
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["unknown_ratio_smooth"] = (
            df["unknown_ratio"]
            .rolling(window=smoothing_window, center=True, min_periods=1)
            .mean()
        )
    else:
        df["unknown_ratio_smooth"] = df["unknown_ratio"]

    return df


def comfort_level_from_unknown_share(
    unknown_share: float,
    comfort_green_max: float,
    comfort_yellow_max: float,
) -> str:
    """Возвращает 'green', 'yellow' или 'red' по доле незнакомых слов."""
    if unknown_share <= comfort_green_max:
        return "green"
    if unknown_share <= comfort_yellow_max:
        return "yellow"
    return "red"

# TODO: может, стоит вывводить не просто слова, а слово в предложении?
