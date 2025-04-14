import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1CqQ5SYtPKTOHBX6i-XDnbAMtR-5lWP3L"
    return pd.read_csv(url)

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Price Prediction"])

# Main content
st.title("House Price Prediction Dashboard")

if page == "Data Overview":
    st.header("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.write(df)
    
    st.subheader("Data Summary")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    # Price distribution
    
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')
    # Now compute correlation on numeric data
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    # Feature vs Price
    st.subheader("Feature vs Price")
    feature = st.selectbox("Select feature", df.columns.drop('price'))
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature], y=df['price'], ax=ax)
    st.pyplot(fig)

elif page == "Price Prediction":
    st.header("House Price Prediction")
    
    # Function to convert categorical features
    def preprocess_data(df):
        # Make a copy to avoid modifying original dataframe
        df_processed = df.copy()
        
        # Binary features (yes/no to 1/0)
        binary_features = ['mainroad', 'guestroom', 'basement', 
                          'hotwaterheating', 'airconditioning', 'prefarea']
        for feature in binary_features:
            df_processed[feature] = df_processed[feature].map({'yes': 1, 'no': 0})
        
        # Furnishing status (categorical to numerical)
        furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
        df_processed['furnishingstatus'] = df_processed['furnishingstatus'].map(furnishing_map)
        
        return df_processed
    
    # Load or train model
    try:
        model = joblib.load('linear_regression_model.joblib')
        # Load the preprocessing mapping
        preprocessing_info = joblib.load('preprocessing_info.joblib')
    except:
        # Preprocess the data
        df_processed = preprocess_data(df)
        
        # Features for modeling
        X = df_processed[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
                         'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                         'parking', 'prefarea', 'furnishingstatus']]
        y = df_processed['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save model and preprocessing info
        joblib.dump(model, 'linear_regression_model.joblib')
        joblib.dump({
            'binary_features': ['mainroad', 'guestroom', 'basement', 
                               'hotwaterheating', 'airconditioning', 'prefarea'],
            'furnishing_map': {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
        }, 'preprocessing_info.joblib')
    
    # User input
    st.subheader("Enter Property Details")
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Total area (sq ft)", min_value=500, max_value=10000, value=6000)
        bedrooms = st.slider("Number of bedrooms", 1, 10, 3)
        bathrooms = st.slider("Number of bathrooms", 1, 6, 2)
        stories = st.slider("Number of stories", 1, 4, 2)
        parking = st.slider("Parking spaces", 0, 4, 2)
        
    with col2:
        mainroad = st.selectbox("Main road access", ["yes", "no"])
        guestroom = st.selectbox("Guest room", ["yes", "no"])
        basement = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot water heating", ["yes", "no"])
        airconditioning = st.selectbox("Air conditioning", ["yes", "no"])
        prefarea = st.selectbox("Preferred area", ["yes", "no"])
        furnishingstatus = st.selectbox("Furnishing status", 
                                      ["unfurnished", "semi-furnished", "furnished"])
    
    # Prediction
    if st.button("Predict Price"):
        # Convert user input to model format
        input_dict = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'mainroad': 1 if mainroad == "yes" else 0,
            'guestroom': 1 if guestroom == "yes" else 0,
            'basement': 1 if basement == "yes" else 0,
            'hotwaterheating': 1 if hotwaterheating == "yes" else 0,
            'airconditioning': 1 if airconditioning == "yes" else 0,
            'parking': parking,
            'prefarea': 1 if prefarea == "yes" else 0,
            'furnishingstatus': {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}[furnishingstatus]
        }
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted House Price: ${prediction:,.2f}")

         # Model performance
        
        # Actual vs Predicted plot
        

# Footer
st.sidebar.markdown("---")
st.sidebar.info("House Price Prediction Dashboard | Made with Streamlit")