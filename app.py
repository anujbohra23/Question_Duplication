import streamlit as st
import helper
import pickle


import pickle
from sklearn.exceptions import NotFittedError

try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except (ValueError, NotFittedError) as e:
    print(f"Error loading the model: {e}")
    # Handle the error as needed
st.header("Duplicate Question Pairs")

q1 = st.text_input("Enter question 1")
q2 = st.text_input("Enter question 2")

if st.button("Find"):
    query = helper.query_point_creator(q1, q2)
    result = model.predict(query)[0]

    if result:
        st.header("Duplicate")
    else:
        st.header("Not Duplicate")
