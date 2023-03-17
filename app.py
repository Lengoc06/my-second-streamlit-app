import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

filename = 'model.pickle'
model = pickle.load(open(filename, "rb"))

st.title('Revenue Prediction')
x_new = st.number_input('Input Temperature')
if st.button('Predict'):
    y_pred = model[x_new]
    st.success(y_pred)


