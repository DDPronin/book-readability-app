import streamlit as st
import plotly.graph_objects as go

from config import CONFIG
from core import (
    SpacyTextProcessor,
    VocabularyProfile,
    BinarySearchVocabularyTester,
    build_difficulty_curve,
    comfort_level_from_unknown_share,
    TextProcessingError,
)


st.set_page_config(
    page_title="Сложность чтения книги",
    layout="wide",
)


# ---------- служебные функции ----------

def init_session_state():
    if "processor" not in st.session_state:
        st.session_state["processor"] = SpacyTextProcessor()
    if "processed_text" not in st.session_state:
        st.session_state["processed_text"] = None
    if "vocab" not in st.session_state:
        st.session_state["vocab"] = None
    if "tester" not in st.session_state:
        st.session_state["tester"] = None
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = None


def make_curve_figure(df):
    """Строим plotly‑график с цветными зонами комфорта."""
    if df.empty:
        return go.Figure()

    x = df["position_frac"] * 100
    y = df["unknown_ratio_smooth"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Незнакомые слова (сглажено)",
        )
    )

    green_max = CONFIG.comfort_green_max_unknown * 100
    yellow_max = CONFIG.comfort_yellow_max_unknown * 100
    y_max = max(y.max() if len(y) else 0, yellow_max * 1.5)

    # цветные зоны
    fig.add_shape(
        type="rect",
        x0=0,
        x1=100,
        y0=0,
        y1=green_max,
        fillcolor="green",
        opacity=0.15,
        line_width=0,
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=0,
        x1=100,
        y0=green_max,
        y1=yellow_max,
        fillcolor="yellow",
        opacity=0.15,
        line_width=0,
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=0,
        x1=100,
        y0=yellow_max,
        y1=y_max,
        fillcolor="red",
        opacity=0.15,
        line_width=0,
        layer="below",
    )

    fig.update_layout(
        xaxis_title="Позиция в книге, %",
        yaxis_title="Доля незнакомых слов, %",
        yaxis_range=[0, y_max],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------- UI ----------

init_session_state()

st.title("Оценка сложности чтения английской книги")
st.markdown(
    """
Это простое приложение оценивает, **какую долю слов** книги вы, скорее всего,
не знаете, и строит кривую сложности по ходу текста.

1. Загрузите `.txt` файл книги.
2. Пройдите короткий **yes/no** тест на знание слов.
3. Посмотрите итоговую статистику и график.
"""
)

uploaded = st.file_uploader("Загрузите текст книги (.txt)", type=["txt"])

if uploaded is not None and st.button("Обработать текст"):
    raw_bytes = uploaded.getvalue()
    text = raw_bytes.decode("utf-8", errors="ignore")
    processor = st.session_state["processor"]

    try:
        with st.spinner("Разбираю текст и строю частотный словарь..."):
            processed = processor.process(text)
            vocab = VocabularyProfile.from_processed_text(processed)
    except TextProcessingError as exc:
        st.error(f"Ошибка при обработке текста: {exc}")
    else:
        st.session_state["processed_text"] = processed
        st.session_state["vocab"] = vocab
        st.session_state["tester"] = None
        st.session_state["current_question"] = None

        st.success(
            f"Готово! В тексте {len(processed.tokens)} токенов, "
            f"{vocab.total_tokens} токенов учтено в частотном словаре, "
            f"{len(vocab)} различных лемм (без имён собственных)."
        )

processed = st.session_state["processed_text"]
vocab = st.session_state["vocab"]

if processed and vocab:
    st.subheader("Статистика текста")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Токенов (после очистки)", len(processed.tokens))
    with col2:
        st.metric("Лемм в частотнике", len(vocab))
    with col3:
        st.metric("Всего токенов для оценки", vocab.total_tokens)

    st.markdown("---")
    st.subheader("Тест на словарный запас")

    # запуск теста
    if st.session_state["tester"] is None:
        if st.button("Начать тест"):
            st.session_state["tester"] = BinarySearchVocabularyTester(
                vocab=vocab,
                batch_size=CONFIG.bs_batch_size,
                max_questions=CONFIG.max_questions,
            )
            st.session_state["current_question"] = None
            st.rerun()

tester: BinarySearchVocabularyTester | None = st.session_state.get("tester")

if tester and processed and vocab:
    if not tester.is_finished:
        st.info(
            "Отмечайте **да/нет**, исходя из того, знаете ли вы слово "
            "в типичном контексте чтения."
        )

        progress = tester.question_count / CONFIG.max_questions
        st.progress(
            progress,
            text=f"Вопросов задано: {tester.question_count} / {CONFIG.max_questions}",
        )

        if st.session_state["current_question"] is None:
            # Здесь next_question может сразу завершить тест и вернуть None
            q = tester.next_question()
            st.session_state["current_question"] = q

            # Если после вызова next_question тест уже завершён,
            # сразу перерисовываем приложение и уйдём в ветку "tester.is_finished"
            if q is None and tester.is_finished:
                st.rerun()
        else:
            q = st.session_state["current_question"]

        if q is None:
            st.info("Алгоритм получил достаточно данных, завершаем тест.")
        else:
            st.markdown(
                f"### Знаете ли вы это слово?\n\n"
                f"**`{q.lemma}`**"
            )
            c1, c2 = st.columns(2)
            if c1.button("Да, знаю", key=f"yes_{q.vocab_index}_{tester.question_count}"):
                tester.record_answer(q.vocab_index, True)
                st.session_state["current_question"] = None
                st.rerun()
            if c2.button("Нет, не знаю", key=f"no_{q.vocab_index}_{tester.question_count}"):
                tester.record_answer(q.vocab_index, False)
                st.session_state["current_question"] = None
                st.rerun()
    else:
        # <-- эта часть уже есть: здесь считаются known/unknown и рисуется график
        st.success("Тест завершён 🎉")

        threshold_index = tester.estimated_threshold_index
        if threshold_index is None:
            st.error("Не удалось оценить порог словарного запаса.")
        else:
            known_share = vocab.known_token_share(threshold_index)
            unknown_share = vocab.unknown_token_share(threshold_index)
            level = comfort_level_from_unknown_share(
                unknown_share,
                CONFIG.comfort_green_max_unknown,
                CONFIG.comfort_yellow_max_unknown,
            )

            st.markdown(
                f"""
**Оценка словаря для этой книги**

* Знаете примерно **{known_share * 100:.1f}%** словоупотреблений.
* Незнакомых — **{unknown_share * 100:.1f}%** (имена собственные не считаются).
"""
            )

            if level == "green":
                st.success("Зелёный уровень: книгу будет комфортно читать.")
            elif level == "yellow":
                st.warning(
                    "Жёлтый уровень: чтение возможно, но будет требовать усилий."
                )
            else:
                st.error(
                    "Красный уровень: книга будет тяжёлой для чтения, "
                    "много незнакомых слов."
                )

            st.subheader("Кривая сложности по ходу книги")
            df_curve = build_difficulty_curve(
                processed,
                vocab,
                threshold_index,
                CONFIG.segment_token_size,
                CONFIG.smoothing_window,
            )
            fig = make_curve_figure(df_curve)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Цветные зоны основаны на исследованиях по lexical coverage: "
                "зелёная зона ≈ 98–99% знакомых слов, жёлтая — 95–98%, "
                "красная — ниже 95% знакомых слов."
            )
