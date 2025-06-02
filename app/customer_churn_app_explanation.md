
# Customer Churn Prediction App - Code Explanation

This Streamlit application predicts customer churn risk based on user inputs using a pre-trained machine learning model. Here's a detailed breakdown of the code:

---

## 1. **Imports and Setup**
```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
```
- Standard data handling and visualization libraries.
- `joblib` is used to load the machine learning model.

## 2. **Streamlit Page Configuration**
```python
st.set_page_config(...)
```
- Sets page title, icon, layout, and sidebar settings.

## 3. **Custom CSS Styling**
```python
st.markdown("""<style>...</style>""", unsafe_allow_html=True)
```
- Improves the visual aesthetics using inline CSS for headers, risk boxes, etc.

## 4. **Model Loader with Error Handling**
```python
@st.cache_resource
def load_model():
    ...
```
- Loads the model from a local path.
- Includes error messages for missing or faulty model files.

## 5. **App Header and Sidebar Navigation**
```python
st.markdown(...)
page = st.sidebar.selectbox(...)
```
- Displays a main title and navigation options (Prediction, Model Info, Help).

## 6. **Prediction Page UI**
### User Input
- Organizes inputs into two columns (demographics, services, account info, etc.).
- Uses `selectbox`, `slider`, and `number_input` widgets.

### Prediction Button and Output
```python
if st.button("🎯 Predict Churn Risk", ...):
    ...
```
- Gathers user input, transforms to NumPy array format.
- Makes a prediction using the loaded model.
- Displays:
  - Risk box (high/low)
  - Churn probability
  - Risk level (Low/Medium/High)
  - Plotly gauge chart
  - Recommendations based on risk

## 7. **Model Info Page**
```python
elif page == "📊 Model Info":
```
- Provides an overview of the model, features, and their importance.
- Displays a horizontal bar chart of top 10 important features.

## 8. **Help Page**
```python
else:  # Help page
```
- Offers instructions and tips for using the app.
- Describes churn risk levels and how to interpret the output.

## 9. **Footer**
```python
st.markdown(...)
```
- Adds a centered footer at the bottom of the app.

---

##  Summary
This app provides an intuitive interface for predicting customer churn based on user-entered data. It leverages a trained model and explains the output visually with charts, metrics, and actionable recommendations.

