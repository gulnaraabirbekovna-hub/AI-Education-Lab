import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='AI Education Lab', page_icon='🤖', layout='wide')

data = {
    'hours':[1,2,3,4,5,6,7,8,2,4,6,8,3,5,7,9],
    'attendance':[60,65,70,75,80,85,90,95,62,78,88,96,72,82,91,98],
    'result':['Низкий','Низкий','Низкий','Средний','Средний','Средний','Высокий','Высокий','Низкий','Средний','Высокий','Высокий','Низкий','Средний','Высокий','Высокий']
}
df = pd.DataFrame(data)
X, y = df[['hours','attendance']], df['result']
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

st.title('🤖 AI Education Lab')
st.subheader('Искусственный интеллект в образовании')
st.write('Интерактивный учебный проект для курса «ИИ в образовании».')

page = st.sidebar.radio('📚 Навигация', ['Главная','История ИИ','Направления ИИ','Python и Colab','ИИ в образовании','ML-лаборатория','Мини-тест','Выводы'])

if page == 'Главная':
    st.header('Добро пожаловать!')
    a,b,c = st.columns(3)
    a.metric('Направления','7'); b.metric('Язык','Python'); c.metric('ML-модель','Random Forest')
    st.info('Проект показывает путь от основных понятий ИИ до работающей ML-модели.')
    st.markdown('### Изучаем\n- 🤖 AI — искусственный интеллект\n- 📊 ML — машинное обучение\n- 🧠 DL — глубокое обучение\n- 💬 NLP — обработка языка\n- 👁️ CV — компьютерное зрение\n- 🎮 RL — обучение с подкреплением\n- ✨ GenAI — генеративный ИИ')

elif page == 'История ИИ':
    st.header('📜 История искусственного интеллекта')
    items = [('1950','Алан Тьюринг','Идея проверки интеллектуального поведения машины.'),('1956','Дартмутская конференция','Формирование искусственного интеллекта как научного направления.'),('1960–1980-е','Экспертные системы','Системы на основе правил и баз знаний.'),('1990–2000-е','Machine Learning','Рост методов обучения на данных.'),('2010-е','Deep Learning','Прорыв глубоких нейронных сетей.'),('2020-е','Generative AI','Широкое применение генеративных моделей.')]
    for year,title,desc in items:
        st.markdown(f'### {year} — {title}')
        st.write(desc)

elif page == 'Направления ИИ':
    st.header('🧠 Ключевые направления')
    topics = {'AI':'Искусственный интеллект — общее понятие интеллектуальных систем.','ML':'Machine Learning — обучение алгоритмов на данных.','DL':'Deep Learning — многослойные нейронные сети.','NLP':'Natural Language Processing — работа с человеческим языком.','CV':'Computer Vision — анализ изображений и видео.','RL':'Reinforcement Learning — обучение через взаимодействие и вознаграждение.','GenAI':'Generative AI — создание нового текста, изображений, кода и другого контента.'}
    for k,v in topics.items():
        with st.expander(k):
            st.write(v)
            st.write('**Пример в образовании:** цифровые помощники, адаптивное обучение, анализ материалов или создание учебного контента.')

elif page == 'Python и Colab':
    st.header('🐍 Python и Google Colab')
    st.write('Python удобен для AI благодаря библиотекам Pandas, Matplotlib и Scikit-learn.')
    st.code("""import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
prediction = model.predict([[5, 84]])""", language='python')
    st.markdown('- **Pandas** — данные\n- **Matplotlib** — графики\n- **Scikit-learn** — ML\n- **Streamlit** — веб-приложение')

elif page == 'ИИ в образовании':
    st.header('🎓 ИИ в образовании')
    for title,desc in [('Персонализация','Подбор заданий по уровню ученика.'),('Создание материалов','Генерация вопросов, упражнений и объяснений.'),('Анализ данных','Поиск закономерностей в учебных результатах.'),('Обратная связь','Помощь в подготовке комментариев и рекомендаций.'),('Проектная деятельность','Поддержка программирования и исследований.')]:
        st.markdown(f'### {title}')
        st.write(desc)
    st.warning('ИИ должен помогать учителю, а не заменять педагогическое решение. Важно учитывать качество данных, конфиденциальность и академическую честность.')

elif page == 'ML-лаборатория':
    st.header('🧪 ML-лаборатория')
    hours = st.slider('⏱️ Часы подготовки в неделю',0,12,5)
    attendance = st.slider('📅 Посещаемость (%)',40,100,84)
    new = pd.DataFrame({'hours':[hours],'attendance':[attendance]})
    pred = model.predict(new)[0]
    probs = model.predict_proba(new)[0]
    st.success(f'🤖 Прогноз модели: **{pred}**')
    st.dataframe(pd.DataFrame({'Категория':model.classes_,'Вероятность (%)':[round(x*100,1) for x in probs]}), use_container_width=True)
    fig,ax=plt.subplots(figsize=(8,5))
    for label in df.result.unique():
        s=df[df.result==label]
        ax.scatter(s.hours,s.attendance,label=label,s=80)
    ax.set_xlabel('Часы подготовки'); ax.set_ylabel('Посещаемость (%)'); ax.set_title('Данные для демонстрации ML'); ax.legend(); ax.grid(True)
    st.pyplot(fig)
    st.caption('Данные придуманы для учебной демонстрации и не предназначены для реальной оценки школьников.')

elif page == 'Мини-тест':
    st.header('📝 Мини-тест')
    qs=[('Что означает ML?',['Машинное обучение','Математическая логика','Мультимедийная лаборатория'],0),('Что создаёт GenAI?',['Только таблицы','Новый контент','Только сети'],1),('Что изучает CV?',['Изображения и видео','Только звук','Только таблицы'],0),('Какой язык используется?',['Python','HTML','SQL'],0)]
    ans=[]
    for i,(q,opts,_) in enumerate(qs): ans.append(st.radio(q,opts,key=f'q{i}'))
    if st.button('Проверить'):
        score=sum(ans[i]==opts[correct] for i,(q,opts,correct) in enumerate(qs))
        st.success(f'Результат: {score} из {len(qs)}')
        if score==len(qs): st.balloons()

else:
    st.header('🎯 Выводы')
    st.write('Проект показывает основные направления ИИ и практическое применение Python и ML.')
    st.success('ИИ при грамотном и ответственном использовании может расширять возможности учителя и делать обучение более интерактивным и персонализированным.')

st.divider()
st.caption('AI Education Lab | Учебный проект для курса «ИИ в образовании»')
