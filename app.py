from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
df = pd.read_csv('/content/drive/MyDrive/IceCreamData.csv')
df.head()
x = df['Temperature'].values
y = df['Revenue'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=50)

from sklearn.linear_model import LinearRegression
x_train = x_train.reshape(-1,1)
model = LinearRegression()
model.fit(x_train, y_train)import streamlit as st
import plotly.graph_objects as go

st.title('Revenue Prediction')
x_new = st.number_input('Input Temperature')
if st.button('Predict'):
    y_pred = model.predict(x_new)
    st.success(str(y_pred))


