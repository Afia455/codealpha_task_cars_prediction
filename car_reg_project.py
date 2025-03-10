import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import base64
import pickle

st.title("Car Shop 🚗")
st.subheader("Cars Selling Price Prediction App")
data_set = pd.read_csv("car_data.csv")

data_set.columns = ['Car_Name','Year','Selling_Price','Present_Price','Driven_kms','Fuel_Type','Selling_type','Transmission','Owner']

st.subheader("📊 Dataset Preview")
st.write(data_set)

st.subheader("EDA of Dataset: ")

st.write("Null values in dataSet",data_set.isnull().sum())
st.write("Columns present in dataSet", data_set.columns)
st.write("Numeric Description opf dataSet", data_set.describe())

st.subheader("Visualization of datatSet: ")

st.write("histogram to identifying selling price")
hist, ax = plt.subplots()
sns.histplot(data_set['Selling_Price'], kde=True)
st.pyplot(hist)

st.write("histogram to identifiyng present price")
hist, ax = plt.subplots()
sns.histplot(data_set['Present_Price'], kde=True)
st.pyplot(hist)

st.write("Hist plot to show relation between columns")

cor_colum = data_set[['Year', 'Selling_Price', 'Present_Price', 'Driven_kms']].corr()

fig, ax = plt.subplots()
sns.heatmap(cor_colum, annot=True, cmap="PiYG", ax=ax)
st.pyplot(fig)

scat, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x="Driven_kms", y="Present_Price", data=data_set, marker='o', hue="Present_Price", palette='deep', legend=False)
st.pyplot(scat)

def load_model():
    with open("car_model_reg.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def main():
    st.sidebar.header("Enter Car Details")
    
    # Load dataset
    data = pd.DataFrame({
        "Car_Name": ["ritz", "sx4", "ciaz", "wagon r", "swift", "vitara brezza", "alto 800", "ertiga", "dzire"],
        "Year": [2014, 2013, 2017, 2011, 2014, 2018, 2017, 2016, 2009],
        "Present_Price": [5.59, 9.54, 9.85, 4.15, 6.87, 9.83, 3.6, 10.79, 7.21],
        "Driven_kms": [27000, 43000, 6900, 5200, 42450, 2071, 2135, 43000, 77427],
        "Fuel_Type": ["Petrol", "Diesel", "Petrol", "Petrol", "Diesel", "Diesel", "Petrol", "Diesel", "Petrol"],
        "Selling_type": ["Dealer"] * 9,
        "Transmission": ["Manual"] * 9,
        "Owner": [0] * 9
    })
    
    # st.write("### Dataset Sample")
    # st.dataframe(data)
    
    # Sidebar Inputs
    year = st.sidebar.slider("Year of Purchase", 2000, 2024, 2015)
    present_price = st.sidebar.number_input("Present Price (in Lakh Rs.)", min_value=0.0, max_value=50.0, value=5.0)
    driven_kms = st.sidebar.number_input("Driven Kilometers", min_value=0, max_value=300000, value=30000)
    fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.sidebar.selectbox("Selling Type", ["Dealer", "Individual"])
    transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.sidebar.selectbox("Previous Owners", [0, 1, 3])
    
    # One-Hot Encoding for categorical variables
    fuel_diesel = 1 if fuel_type == "Diesel" else 0
    fuel_petrol = 1 if fuel_type == "Petrol" else 0
    selling_individual = 1 if selling_type == "Individual" else 0
    transmission_auto = 1 if transmission == "Automatic" else 0
    transmission_manual = 1 if transmission == "Manual" else 0
    owner_0 = 1 if owner == 0 else 0
    owner_1 = 1 if owner == 1 else 0
    owner_3 = 1 if owner == 3 else 0
    
    model = load_model()
    
    if st.sidebar.button("Predict Price"):
        features = [[year, present_price, driven_kms, fuel_diesel, fuel_petrol, selling_individual, transmission_auto, transmission_manual, owner_0, owner_1, owner_3]]
        prediction = model.predict(features)[0]
        
        # Find closest matching car name based on present price
        closest_match = data.iloc[(data["Present_Price"] - present_price).abs().argsort()[:1]]
        car_name = closest_match["Car_Name"].values[0]
        
        st.subheader("car price: ")
        st.success(f"Predicted Selling Price: Rs. {prediction:.2f} Lakh")
        st.subheader("Car Name: ")
        st.info(f"Similar Car: {car_name}")


if __name__ == "__main__":
    main()