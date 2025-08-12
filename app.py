import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from sklearn.model_selection import train_test_split

# ======== PAGE CONFIGURATION ========
st.set_page_config(
    page_title="🚢 Titanic Survival Prediction",
    page_icon="🛳️",
    layout="wide"
)

# ======== CUSTOM STYLING ========
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 20px;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .metric-card {
            padding: 15px;
            border-radius: 12px;
            background-color: white;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            margin: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ======== TITLE ========
st.title("🚢 Titanic Survival Prediction")
st.markdown("""
Welcome to the Titanic Survival Prediction App!  
Predict survival chances, explore the dataset, and visualize patterns.
""")

# ======== SIDEBAR ========
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to", ["📊 Data Exploration", "📈 Visualizations", "🤖 Model Prediction", "📋 Model Performance"])

# ======== LOAD DATA ========
@st.cache_data
def load_data():
    titanic = pd.read_csv("data/Titanic-Dataset.csv")
    return titanic

titanic = load_data()

# ======== LOAD MODEL ========
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ======== PREPROCESS INPUT ========
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    sex_val = 1 if sex == "female" else 0
    embarked_val = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}[embarked]
    return np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])

# ======== DATA EXPLORATION ========
if section == "📊 Data Exploration":
    st.subheader("Dataset Overview")
    st.write(titanic.head())
    st.write(f"Rows: {titanic.shape[0]}, Columns: {titanic.shape[1]}")
    st.write(titanic.describe())

# ======== VISUALIZATIONS ========
elif section == "📈 Visualizations":
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(titanic, x='Pclass', color='Survived', barmode='group', title="Survival Count by Class")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(titanic, x='Age', color='Survived', nbins=30, title="Age Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(titanic, x='Age', y='Fare', color='Survived', title="Fare vs Age")
    st.plotly_chart(fig3, use_container_width=True)

# ======== MODEL PREDICTION ========
elif section == "🤖 Model Prediction":
    st.subheader("Enter Passenger Details")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class (1-3)", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 30)
    with col2:
        sibsp = st.number_input("Siblings/Spouses", 0, 8, 0)
        parch = st.number_input("Parents/Children", 0, 6, 0)
        fare = st.number_input("Fare", 0.0, 500.0, 30.0)
        embarked = st.selectbox("Embarked", ["Southampton", "Cherbourg", "Queenstown"])

    if st.button("Predict Survival"):
        input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.success(f"✅ Passenger likely SURVIVED! ({prob:.2%} probability)")
        else:
            st.error(f"❌ Passenger likely DID NOT survive. ({prob:.2%} probability)")

# ======== MODEL PERFORMANCE ========
elif section == "📋 Model Performance":
    titanic_processed = titanic.copy()
    titanic_processed['Age'] = titanic_processed['Age'].fillna(titanic_processed['Age'].median())
    titanic_processed['Embarked'] = titanic_processed['Embarked'].fillna(titanic_processed['Embarked'].mode()[0])
    titanic_processed['Fare'] = titanic_processed['Fare'].fillna(titanic_processed['Fare'].median())
    titanic_processed['Sex'] = titanic_processed['Sex'].map({'male': 0, 'female': 1})
    titanic_processed['Embarked'] = titanic_processed['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = titanic_processed[features]
    y = titanic_processed['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    st.subheader("Performance Metrics")
    metric_cols = st.columns(len(metrics))
    for i, (name, value) in enumerate(metrics.items()):
        with metric_cols[i]:
            st.markdown(f"<div class='metric-card'><h4>{name}</h4><h2>{value:.2f}</h2></div>", unsafe_allow_html=True)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
