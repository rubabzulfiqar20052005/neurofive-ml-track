import streamlit as st
import joblib
import pandas as pd

pipeline = joblib.load('titanic_final_pipeline.pkl')

st.title('Titanic Survival Predictor')
st.write('Enter passenger details to predict survival chance')

pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 25)
fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=30.0)
sibsp = st.number_input('Siblings/Spouses Aboard', min_value=0, max_value=8, value=0)
parch = st.number_input('Parents/Children Aboard', min_value=0, max_value=6, value=0)
embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

if st.button('Predict'):
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare],
        'FamilySize': [family_size],
        'IsAlone': [is_alone],
        'Sex': [sex],
        'Embarked': [embarked]
    })

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f'This passenger would likely SURVIVE (probability: {probability:.2%})')
    else:
        st.error(f'This passenger would likely NOT survive (probability: {probability:.2%})')