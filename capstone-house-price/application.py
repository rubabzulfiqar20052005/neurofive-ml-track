import streamlit as st
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'house_price_model.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))

st.title('Pakistan House Price Predictor')
st.write('Estimate a fair market price for a property based on Zameen.com listing data')

property_type = st.selectbox('Property Type', ['House', 'Flat', 'Penthouse', 'Farm House', 'Lower Portion', 'Upper Portion', 'Room'])
city = st.selectbox('City', ['Islamabad', 'Lahore', 'Karachi', 'Rawalpindi', 'Faisalabad'])
location = st.text_input('Location (e.g. DHA Defence, Gulberg, Clifton)', 'DHA Defence')
baths = st.number_input('Bathrooms', min_value=0, max_value=20, value=3)
bedrooms = st.number_input('Bedrooms', min_value=0, max_value=15, value=3)
area = st.number_input('Total Area (sq ft)', min_value=100, max_value=100000, value=3000)

if st.button('Predict Price'):
    input_dict = {col: 0 for col in model_columns}
    input_dict['baths'] = baths
    input_dict['bedrooms'] = bedrooms
    input_dict['Total_Area'] = area

    ptype_col = f'property_type_{property_type}'
    if ptype_col in input_dict:
        input_dict[ptype_col] = 1

    city_col = f'city_{city}'
    if city_col in input_dict:
        input_dict[city_col] = 1

    loc_col = f'location_grouped_{location}'
    if loc_col in input_dict:
        input_dict[loc_col] = 1
    else:
        other_col = 'location_grouped_Other'
        if other_col in input_dict:
            input_dict[other_col] = 1

    input_df = pd.DataFrame([input_dict])[model_columns]
    prediction = model.predict(input_df)[0]
    st.success(f'Estimated Price: PKR {prediction:,.0f}')
