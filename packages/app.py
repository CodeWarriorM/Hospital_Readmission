import streamlit as st
import requests

st.title('Hospital Readmission')

#st.write('Select your features below:')

url = "http://127.0.0.1:8000/"
response = requests.get(url)
data = response.json()
st.write("Response from local API", data)
