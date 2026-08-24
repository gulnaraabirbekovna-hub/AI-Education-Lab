import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# НАСТРОЙКА ПРОЕКТА
# =========================================================

st.set_page_config(
    page_title="AI-Education-Lab",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Education-Lab")
st.subheader("Практическая лаборатория машинного обучения для 10 класса")

st.write(
    "Практический проект: ученик работает с данными, "
    "создаёт регрессионную модель, получает прогноз, "
    "анализирует графики и делает собственный вывод."
)


# =========================================================
# 1. ПОСТАНОВКА ЗАДАЧИ
# =========================================================

st.header("🎯 1. Постановка задачи")

st.info(
    "Можно ли по часам подготовки, посещаемости и количеству "
    "выполненных заданий спрогнозировать итоговый результат ученика?"
)

st.markdown("""
**Цель:**  
Создать модель машинного обучения, которая прогнозирует числовой результат ученика.

**Признаки:**
- ⏱️ часы подготовки;
- 📅 посещаемость;
- 📝 выполненные задания.

**Целевая переменная:**
- 🎯 итоговый балл.
""")


# =========================================================
# БАЗА ДАННЫХ
# =========================================================

st.header("📂 2. База данных")

default_data = {
    "hours": [2, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 6, 3, 9],
    "attendance": [65, 68, 70, 75, 78, 80, 82, 85, 88, 90, 92, 93, 95, 75, 60, 88],
    "tasks": [8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 12, 7, 18],
    "score": [55, 58, 62, 66, 70, 72, 76, 81, 84, 87, 89, 91, 94, 73, 52, 88]
}

default_df = pd.DataFrame(default_data)

uploaded_file = st.file_uploader(
    "📥 Загрузите свою базу CSV или Excel",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Ваша база данных загружена!")

    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        df = default_df.copy()

else:
    df = default_df.copy()

required_columns = ["hours", "attendance", "tasks", "score"]

if all(column in df.columns for column in required_columns):

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna()

else:

    st.error(
        "❌ В базе должны присутствовать столбцы: "
        "hours, attendance, tasks, score"
    )

    st.stop()


st.write("### 📊 Данные")

st.dataframe(
    df,
    use_container_width=True
)


# =========================================================
# 3. ПОДГОТОВКА И АНАЛИЗ ДАННЫХ
# =========================================================

st.header("📊 3. Подготовка и анализ данных")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Количество учеников",
    len(df)
)

col2.metric(
    "Средний результат",
    f"{df['score'].mean():.1f}"
)

col3.metric(
    "Средняя посещаемость",
    f"{df['attendance'].mean():.1f}%"
)

col4.metric(
    "Средние часы подготовки",
    f"{df['hours'].mean():.1f}"
)


# =========================================================
# 4. ВИЗУАЛИЗАЦИЯ
# =========================================================

st.header("📈 4. Анализ и визуализация")

graph = st.selectbox(
    "Выберите график:",
    [
        "Часы подготовки → результат",
        "Посещаемость → результат",
        "Задания → результат",
        "Распределение результатов"
    ]
)

fig, ax = plt.subplots(figsize=(9, 5))

if graph == "Часы подготовки → результат":

    ax.scatter(
        df["hours"],
        df["score"],
        s=80
    )

    ax.set_xlabel("Часы подготовки")
    ax.set_ylabel("Итоговый балл")
    ax.set_title("Зависимость результата от подготовки")

elif graph == "Посещаемость → результат":

    ax.scatter(
        df["attendance"],
        df["score"],
        s=80
    )

    ax.set_xlabel("Посещаемость (%)")
    ax.set_ylabel("Итоговый балл")
    ax.set_title("Зависимость результата от посещаемости")

elif graph == "Задания → результат":

    ax.scatter(
        df["tasks"],
        df["score"],
        s=80
    )

    ax.set_xlabel("Количество заданий")
    ax.set_ylabel("Итоговый балл")
    ax.set_title("Зависимость результата от заданий")

else:

    ax.hist(
        df["score"],
        bins=6
    )

    ax.set_xlabel("Баллы")
    ax.set_ylabel("Количество")
    ax.set_title("Распределение результатов")

ax.grid(True, alpha=0.3)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 5. РЕГРЕССИЯ
# =========================================================

st.header("🤖 5. Построение регрессионной модели")

X = df[
    [
        "hours",
        "attendance",
        "tasks"
    ]
]

y = df["score"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


model = LinearRegression()

model.fit(
    X_train,
    y_train
)


predictions = model.predict(X_test)


# =========================================================
# 6. МЕТРИКИ
# =========================================================

st.header("📊 6. Метрики и результаты модели")

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{mae:.2f}"
)

col2.metric(
    "MSE",
    f"{mse:.2f}"
)

col3.metric(
    "R²",
    f"{r2:.2f}"
)

st.write("""
**MAE** — средняя абсолютная ошибка.

**MSE** — средняя квадратичная ошибка.

**R²** — показывает, насколько хорошо модель объясняет данные.
""")


# =========================================================
# 7. ВВОД ДАННЫХ И ПРОГНОЗ
# =========================================================

st.header("⌨️ 7. Практический ввод данных")

st.write(
    "Введите данные нового ученика и получите прогноз."
)

col1, col2, col3 = st.columns(3)

hours_input = col1.number_input(
    "⏱️ Часы подготовки",
    min_value=0.0,
    max_value=20.0,
    value=8.0
)

attendance_input = col2.number_input(
    "📅 Посещаемость (%)",
    min_value=0.0,
    max_value=100.0,
    value=90.0
)

tasks_input = col3.number_input(
    "📝 Выполнено заданий",
    min_value=0,
    max_value=30,
    value=18
)


if st.button(
    "🔮 ПОЛУЧИТЬ ПРОГНОЗ",
    type="primary"
):

    new_student = pd.DataFrame({

        "hours": [hours_input],

        "attendance": [attendance_input],

        "tasks": [tasks_input]

    })

    prediction = model.predict(
        new_student
    )[0]

    prediction = np.clip(
        prediction,
        0,
        100
    )

    st.session_state["prediction"] = prediction

    st.success(
        f"🎯 Прогнозируемый результат: **{prediction:.1f} балла**"
    )

    if prediction >= 90:

        st.balloons()

        st.success(
            "🏆 Высокий прогнозируемый результат!"
        )

    elif prediction >= 70:

        st.info(
            "👍 Хороший результат!"
        )

    else:

        st.warning(
            "📚 Рекомендуется увеличить подготовку."
        )


# =========================================================
# 8. ФАКТ И ПРОГНОЗ
# =========================================================

st.header("📈 Сравнение фактических и прогнозируемых значений")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    range(1, len(y_test) + 1),
    y_test.values,
    marker="o",
    label="Фактический результат"
)

ax.plot(
    range(1, len(predictions) + 1),
    predictions,
    marker="o",
    label="Прогноз модели"
)

ax.set_xlabel("Номер ученика")
ax.set_ylabel("Баллы")

ax.set_title(
    "Фактические и прогнозируемые результаты"
)

ax.legend()

ax.grid(True, alpha=0.3)

st.pyplot(fig)

plt.close(fig)


# =========================================================
# 9. ТЕСТ
# =========================================================

st.header("📝 9. Тест по машинному обучению")

questions = [

    (
        "1. Что такое регрессия?",
        [
            "Метод прогнозирования числового значения",
            "Компьютерная игра",
            "Графический редактор"
        ],
        0
    ),

    (
        "2. Что прогнозирует наша модель?",
        [
            "Посещаемость",
            "Итоговый балл",
            "Количество файлов"
        ],
        1
    ),

    (
        "3. Какая библиотека используется для ML?",
        [
            "Scikit-learn",
            "HTML",
            "PowerPoint"
        ],
        0
    ),

    (
        "4. Что показывает MAE?",
        [
            "Среднюю абсолютную ошибку",
            "Количество учеников",
            "Процент посещаемости"
        ],
        0
    ),

    (
        "5. Для чего нужны графики?",
        [
            "Для анализа закономерностей",
            "Только для украшения",
            "Для удаления данных"
        ],
        0
    )

]


answers = []

for i, (question, options, correct) in enumerate(questions):

    answer = st.radio(
        question,
        options,
        key=f"question_{i}"
    )

    answers.append(answer)


if st.button(
    "✅ ПРОВЕРИТЬ ТЕСТ"
):

    test_score = 0

    for i, (question, options, correct) in enumerate(questions):

        if answers[i] == options[correct]:

            test_score += 1

    test_percent = test_score * 20

    st.session_state["test_score"] = test_percent

    st.success(
        f"📝 Результат теста: {test_score}/5 — {test_percent}%"
    )

    if test_percent == 100:

        st.balloons()


# =========================================================
# 10. ПРАКТИЧЕСКОЕ ЗАДАНИЕ
# =========================================================

st.header("🎯 10. Практическое задание")

st.write(
    "Измените входные данные и исследуйте, "
    "как изменяется прогноз модели."
)

practice_hours = st.slider(
    "Часы подготовки",
    0,
    20,
    6
)

practice_attendance = st.slider(
    "Посещаемость",
    0,
    100,
    80
)

practice_tasks = st.slider(
    "Количество заданий",
    0,
    30,
    15
)


if st.button(
    "🚀 ВЫПОЛНИТЬ ПРАКТИЧЕСКОЕ ЗАДАНИЕ"
):

    practice_data = pd.DataFrame({

        "hours": [practice_hours],

        "attendance": [practice_attendance],

        "tasks": [practice_tasks]

    })

    practice_prediction = model.predict(
        practice_data
    )[0]

    practice_prediction = np.clip(
        practice_prediction,
        0,
        100
    )

    st.session_state[
        "practice_score"
    ] = practice_prediction

    st.success(
        f"🔮 Полученный прогноз: "
        f"**{practice_prediction:.1f} балла**"
    )


# =========================================================
# 11. СОБСТВЕННЫЙ ВЫВОД
# =========================================================

st.header("✍️ 11. Собственный вывод ученика")

st.write(
    "Напишите своими словами, что вы обнаружили "
    "в результате исследования."
)

student_conclusion = st.text_area(
    "Мой вывод:",
    placeholder=(
        "Например: я выяснил(а), что увеличение "
        "часов подготовки и посещаемости связано "
        "с повышением результата..."
    ),
    height=150
)


if st.button(
    "💾 Сохранить мой вывод"
):

    if student_conclusion.strip():

        st.session_state[
            "conclusion"
        ] = student_conclusion

        st.success(
            "✅ Ваш вывод сохранён!"
        )

    else:

        st.warning(
            "Сначала напишите вывод."
        )


# =========================================================
# 12. ЭТИКА
# =========================================================

st.header("⚖️ 12. Ответственное использование ИИ")

st.warning(
    "Результат модели является прогнозом, а не окончательной "
    "оценкой ученика. AI не должен заменять решение учителя."
)

st.write("""
При работе с данными необходимо:

- защищать персональные данные;
- не использовать реальные ФИО без разрешения;
- проверять качество данных;
- учитывать ошибки модели;
- использовать AI ответственно.
""")


# =========================================================
# 13. 8 КРИТЕРИЕВ ВНУТРИ ПРОЕКТА
# =========================================================

st.header("📋 13. 8 критериев проектной работы")

st.write(
    "Каждый критерий связан с практическим этапом "
    "выполнения проекта."
)

criteria = [

    "1. Постановка задачи",

    "2. Подготовка данных",

    "3. Анализ и визуализация",

    "4. Корректность модели",

    "5. Метрики и результаты",

    "6. Реализация прототипа",

    "7. Отчёт и собственный вывод",

    "8. Этические и устойчивые аспекты"

]


for criterion in criteria:

    st.success(
        f"✅ {criterion}"
    )


# =========================================================
# 14. ИТОГ
# =========================================================

st.header("🏆 Итоговая работа")

prediction_result = st.session_state.get(
    "prediction",
    0
)

test_result = st.session_state.get(
    "test_score",
    0
)

practice_result = st.session_state.get(
    "practice_score",
    0
)


result_table = pd.DataFrame({

    "Показатель": [

        "Прогноз модели",

        "Результат теста",

        "Практическое задание",

        "MAE",

        "MSE",

        "R²"

    ],

    "Результат": [

        f"{prediction_result:.1f} балла",

        f"{test_result}%",

        f"{practice_result:.1f} балла",

        f"{mae:.2f}",

        f"{mse:.2f}",

        f"{r2:.2f}"

    ]

})


st.dataframe(
    result_table,
    use_container_width=True
)


st.success(
    "🎓 Проект завершён! "
    "Ученик прошёл путь от постановки задачи "
    "до анализа данных, построения модели, "
    "получения прогноза и собственного вывода."
)


st.divider()

st.caption(
    "AI-Education-Lab | Python • Pandas • "
    "Scikit-learn • Matplotlib • Streamlit"
)
