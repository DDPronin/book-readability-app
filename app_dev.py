import streamlit as st
import plotly.graph_objects as go
import pandas as pd

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
    page_title="Readability dev‑view",
    layout="wide",
)


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


init_session_state()

st.title("Dev‑версия: как работает оценка сложности")

st.markdown(
    """
Эта версия показывает **внутреннее состояние алгоритма**:

- как строится частотный словарь;
- как выбираются слова на этапах калибровки и бинарного поиска;
- как обновляются границы порога.

Используйте её для отладки и демонстрации.
"""
)

uploaded = st.file_uploader("Загрузите текст книги (.txt)", type=["txt"])

if uploaded is not None and st.button("Обработать текст (dev)"):
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
            f"Готово! {len(processed.tokens)} токенов, "
            f"{vocab.total_tokens} токенов в частотнике, "
            f"{len(vocab)} лемм."
        )

processed = st.session_state["processed_text"]
vocab = st.session_state["vocab"]

if processed and vocab:
    # -------- левая колонка: текст и частотник --------
    left, right = st.columns([2, 3])

    with left:
        st.subheader("Частотный словарь (top‑100)")
        vocab_df = pd.DataFrame(
            [
                dict(rank=item.rank, lemma=item.lemma, freq=item.freq)
                for item in vocab.items[:100]
            ]
        )
        st.dataframe(vocab_df, use_container_width=True)

    # -------- запуск теста --------
    with right:
        st.subheader("Алгоритм тестирования")

        if st.session_state["tester"] is None:
            if st.button("Запустить тест (dev)"):
                st.session_state["tester"] = BinarySearchVocabularyTester(
                    vocab=vocab,
                    batch_size=CONFIG.bs_batch_size,
                    max_questions=CONFIG.max_questions,
                )
                st.session_state["current_question"] = None
                st.rerun()

tester: BinarySearchVocabularyTester | None = st.session_state.get("tester")

if tester and processed and vocab:
    left, right = st.columns([2, 3])

    with left:
        if not tester.is_finished:
            st.markdown("### Тест (dev)")
            progress = tester.question_count / CONFIG.max_questions
            st.progress(
                progress,
                text=f"Вопросов: {tester.question_count} / {CONFIG.max_questions}",
            )

            if st.session_state["current_question"] is None:
                q = tester.next_question()
                st.session_state["current_question"] = q

                # Если в ходе next_question тест завершился и вопросов больше нет,
                # перерисовываем приложение и попадём в ветку "tester.is_finished"
                if q is None and tester.is_finished:
                    st.rerun()
            else:
                q = st.session_state["current_question"]

            if q is None:
                st.info("Алгоритм получил достаточно данных, завершаем тест.")
            else:
                st.markdown(
                    f"**Слово:** `{q.lemma}`  \n"
                    f"_index = {q.vocab_index}_"
                )
                c1, c2 = st.columns(2)
                if c1.button(
                    "Да, знаю (dev)",
                    key=f"dev_yes_{q.vocab_index}_{tester.question_count}",
                ):
                    tester.record_answer(q.vocab_index, True)
                    st.session_state["current_question"] = None
                    st.rerun()
                if c2.button(
                    "Нет, не знаю (dev)",
                    key=f"dev_no_{q.vocab_index}_{tester.question_count}",
                ):
                    tester.record_answer(q.vocab_index, False)
                    st.session_state["current_question"] = None
                    st.rerun()
        else:
            st.success("Тест завершён (dev)")

            threshold_index = tester.estimated_threshold_index
            if threshold_index is not None:
                known_share = vocab.known_token_share(threshold_index)
                unknown_share = vocab.unknown_token_share(threshold_index)
                level = comfort_level_from_unknown_share(
                    unknown_share,
                    CONFIG.comfort_green_max_unknown,
                    CONFIG.comfort_yellow_max_unknown,
                )
                st.write(f"Пороговый индекс: {threshold_index}")
                st.write(f"Доля знакомых токенов: {known_share * 100:.1f}%")
                st.write(f"Доля незнакомых токенов: {unknown_share * 100:.1f}%")
                st.write(f"Комфорт: {level}")

                df_curve = build_difficulty_curve(
                    processed,
                    vocab,
                    threshold_index,
                    CONFIG.segment_token_size,
                    CONFIG.smoothing_window,
                )
                fig = make_curve_figure(df_curve)
                st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Состояние алгоритма")
        st.json(tester.debug_state)

        if tester.asked_answers:
            st.markdown("### История заданных вопросов")
            history_df = pd.DataFrame(
                [
                    {
                        "vocab_index": idx,
                        "lemma": vocab[idx].lemma,
                        "freq": vocab[idx].freq,
                        "known": known,
                    }
                    for idx, known in sorted(tester.asked_answers.items())
                ]
            )
            st.dataframe(history_df, use_container_width=True)
