import streamlit as st



import pickle
#st.text_input("Enter Location", 'Rajajinagar')
import numpy as np
import pandas as pd
import joblib

model=joblib.load('best_random_forest_model2.joblib')
st.title('RealEstatePro')
loc_list=[]
with open('loc_dict .pkl', 'rb') as loc_enc_open:
    loc_enc_dict = pickle.load(loc_enc_open)
with open('area_type_means.pkl', 'rb') as area_enc_open:
    area_enc_dict = pickle.load(area_enc_open)
for key,value in loc_enc_dict.items():
        loc_list.append(key)
location_input = st.selectbox('Please select location',(loc_list))
size_input = st.slider("Enter size in BHK", 0,20)
total_sqft_input=st.slider("Enter total sqft",min_value=1.0, max_value=100000.00,step=1e-2, format="%.2f")
bath_input = st.slider("Enter number of bathrooms", 0,20)
balcony_input = st.slider("Enter number of balconies", 0,20)
area_type_input = st.selectbox(
   'Please enter the area type',
    ('Super built-up  Area', 'Plot  Area', 'Built-up  Area','Carpet  Area'))

st.write('You selected:', area_type_input)

st.write('Distinct locations:', len(loc_enc_dict))


def handle_submit():
    size = size_input
    total_sqft = total_sqft_input
    bath = bath_input
    balcony = balcony_input
    location = location_input
    area_type = area_type_input
    location_m = loc_enc_dict[location]
    area_m = area_enc_dict[area_type]
    data = {
    'size': [size],
    'total_sqft': [total_sqft],
    'bath': [bath],
    'balcony': [balcony],
    'location_encoded': [location_m],
    'area_type_encoded': [area_m]
}
    df = pd.DataFrame(data)
    price = model.predict(df)
    
    st.text_area(label="Predicted Price in Lakhs:", value=f"{price[0]:.{2}f}", height=10)
    
trigger = st.button('Predict', on_click=handle_submit)
