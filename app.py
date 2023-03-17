import streamlit as st
from sklearn.linear_model import LinearRegression
import pickle

filename = 'model.pickle'
model = pickle.load(open(filename, "rb"))

st.title('Revenue Prediction')
a = st.number_input('Input Temperature')
if st.button('Predict'):
    x_new = []
    x_new.append(a)
    x_new = np.array(x_new)
    x_new = x_new.reshape(-1, 1)
    y_pred = model.predict(x_new)
    st.success(*y_pred)
