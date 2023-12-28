import streamlit as st
import helper
import numpy as np
from joblib import load

# Dummy data to demonstrate the conversion
# Replace this with your actual data loading logic
node_array = np.array([1, 2, 3], dtype=np.float64)

# Convert the data types to the expected data types
node_array = node_array.astype(
    {
        "left child": np.int64,
        "rights child": np.int64,
        "feature": np.int64,
        "threshold": np.float64,
        "impurity": np.float64,
        "n_node_samples": np.int64,
        "weighted_n_node_samples": np.float64,
    }
)

# Load the joblib model
model = load("model.joblib")

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
