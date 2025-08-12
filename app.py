import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from sklearn.model_selection import train_test_split
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import base64
from datetime import datetime

# Set page configuration with a custom icon
st.set_page_config(page_title="Titanic Time Machine", layout="wide", page_icon="🚢")

st.sidebar.write(f"Selected theme: {theme}")  # Debug output

# Custom CSS for themes with dynamic application
theme_css = """
<style>
    /* General App Styling */
    .vintage-theme { background-color: #f5e6cc; color: #2e2e2e; font-family: 'Garamond', serif; }
    .modern-theme { background-color: #f0f2f6; color: #1a1a1a; font-family: 'Arial', sans-serif; }
    .dark-theme { background-color: #1a1a1a; color: #ffffff; font-family: 'Arial', sans-serif; }
    .stButton>button { background-color: #4682b4; color: white; border-radius: 10px; }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: #2b4066; /* Deep navy blue for a Titanic feel */
        color: #e0e7ff; /* Light off-white for text */
        padding: 20px;
        border-right: 1px solid #a8b5c3; /* Subtle steel gray border */
    }
    .stSidebar .stRadio > label {
        color: #e0e7ff; /* Theme selector text */
    }
    .stSidebar .stRadio > div > div {
        background-color: #3b5582; /* Darker blue for dropdown background */
        border: 1px solid #a8b5c3; /* Steel gray border */
        border-radius: 5px;
    }
    .stSidebar .stRadio [type="radio"] + div span {
        color: #e0e7ff; /* Radio option text */
    }
    .stSidebar .stRadio [type="radio"]:checked + div span {
        color: #ffffff; /* Highlighted option text */
        background-color: #4a6ea9; /* Light blue highlight */
    }
    .stSidebar .css-1aumxhk { /* Target navigation items */
        color: #e0e7ff;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .stSidebar .css-1aumxhk:hover {
        background-color: #4a6ea9; /* Hover effect */
        color: #ffffff;
    }
    .stSidebar .css-1aumxhk[data-testid="stSidebarNavItem"]:before {
        content: "⦿ "; /* Custom bullet or icon */
        color: #a8b5c3; /* Steel gray bullet */
    }
    .stSidebar .css-1aumxhk[data-testid="stSidebarNavItem"][aria-selected="true"] {
        background-color: #3b5582; /* Selected item background */
        color: #ffffff;
        font-weight: bold;
    }
    
    /* Theme-Specific Adjustments for Sidebar */
    .vintage-theme .stSidebar { background-color: #f5e6cc; color: #2e2e2e; border-right: 1px solid #8b6f47; }
    .dark-theme .stSidebar { background-color: #1a1a1a; color: #ffffff; border-right: 1px solid #4a4a4a; }
    
    /* Dynamic Body Styling Based on Theme */
    body {
        background-color: {background_color} !important;
        color: {text_color} !important;
        font-family: {font_family} !important;
        transition: background-color 0.5s, color 0.5s;
    }
    
    /* Title and Content Animation */
    .title, .subheader, .stMarkdown {
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Custom Spinner Animation */
    div.stSpinner > div {
        border-top-color: #4a6ea9 !important;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Title Styling */
    .title { font-size: 2.5em; text-align: center; color: #2e2e2e; }
    .subheader { color: #4682b4; }
</style>
""".format(
    background_color={'Vintage': '#f5e6cc', 'Modern': '#f0f2f6', 'Dark': '#1a1a1a'}[theme],
    text_color={'Vintage': '#2e2e2e', 'Modern': '#1a1a1a', 'Dark': '#ffffff'}[theme],
    font_family={'Vintage': '"Garamond", serif', 'Modern': '"Arial", sans-serif', 'Dark': '"Arial", sans-serif'}[theme]
)

st.markdown(theme_css, unsafe_allow_html=True)

# Sidebar for theme selection and navigation
st.sidebar.title("Titanic Time Machine")
theme = st.sidebar.selectbox("Choose Theme", ["Vintage", "Modern", "Dark"])

section = st.sidebar.radio("Navigate", ["Welcome Aboard", "Explore the Ship", "Visualize the Voyage", "Predict Your Fate", "Model Insights", "Survival Challenge"])

# Load dataset with caching
@st.cache_data
def load_data():
    try:
        titanic = pd.read_csv("data/Titanic-Dataset.csv")
        titanic = titanic.astype({col: 'int64' for col in titanic.select_dtypes(include=['int64']).columns})
        return titanic
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None
titanic = load_data()

# Load pre-trained model with caching
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
model = load_model()

# Preprocess input data for prediction
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    sex_val = 1 if sex == "female" else 0
    embarked_val = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}[embarked]
    return np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])

# Generate PDF report
def generate_pdf_report(prediction, prob, passenger_details):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Titanic Survival Prediction Report")
    c.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(100, 700, "Passenger Details:")
    for i, (key, value) in enumerate(passenger_details.items()):
        c.drawString(120, 680 - i*20, f"{key}: {value}")
    c.drawString(100, 600, f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
    c.drawString(100, 580, f"Survival Probability: {prob:.2%}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Welcome Aboard Section
if section == "Welcome Aboard":
    st.title("Welcome Aboard the Titanic Time Machine 🚢")
    st.markdown("""
    Step into 1912 and board the RMS Titanic! This interactive app lets you explore the fateful voyage, analyze passenger data, 
    visualize survival patterns, and predict your own survival odds using a machine learning model. 
    Will you survive the journey? Navigate through the sections to find out!
    """)
    st.image("titanic.jpg", caption="RMS Titanic, 1912")

# Explore the Ship Section
elif section == "Explore the Ship":
    st.header("Explore the Ship's Passengers")
    if titanic is not None:
        df = titanic.copy()
        for col in df.select_dtypes(include=['int64', 'Int64']).columns:
            df[col] = df[col].astype('float')
        for col in df.select_dtypes(include=['float64', 'Float64']).columns:
            df[col] = df[col].astype('float')
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str)

        st.subheader("Passenger Database")
        st.write(f"Total Passengers: {df.shape[0]}")
        st.write("Columns:", df.columns.tolist())
        
        # Interactive Filtering with Search
        st.subheader("Filter Passengers")
        search_name = st.text_input("Search by Name")
        columns = st.multiselect("Select columns to display", df.columns, default=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age'])
        filtered_df = df[df['Name'].str.contains(search_name, case=False, na=False)] if search_name else df
        st.dataframe(filtered_df[columns], height=400)

        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button("Download Filtered Data", csv, "filtered_titanic.csv", "text/csv")

# Visualize the Voyage Section
elif section == "Visualize the Voyage":
    st.header("Visualize the Voyage")
    if titanic is not None:
        # 1. Animated 3D Scatter Plot with Time Simulation
        st.subheader("Animated 3D Passenger Analysis")
        titanic['PseudoTime'] = titanic['Pclass']  # Using Pclass as a pseudo-time variable
        fig_3d = px.scatter_3d(titanic, x='Age', y='Fare', z='Pclass', color='Survived', animation_frame='PseudoTime',
                              title="Animated 3D Plot: Age, Fare, and Class by Survival",
                              labels={'Pclass': 'Passenger Class', 'Survived': 'Survival (0 = No, 1 = Yes)'},
                              range_z=[0.5, 3.5])
        st.plotly_chart(fig_3d)

        # 2. Animated Survival Rate by Sex and Class
        st.subheader("Animated Survival Rate by Sex and Class")
        survival_rate = titanic.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index()
        fig_stacked = px.bar(survival_rate, x='Pclass', y='Survived', color='Sex', animation_frame='Pclass',
                            title="Survival Rate by Sex and Class", text=survival_rate['Survived'].apply(lambda x: f'{x:.2%}'),
                            labels={'Survived': 'Survival Rate', 'Pclass': 'Passenger Class'})
        fig_stacked.update_traces(textposition='auto')
        st.plotly_chart(fig_stacked)

        # 3. Animated Age Distribution with KDE
        st.subheader("Animated Age Distribution by Survival")
        age_bins = np.arange(0, 80, 5)  # Create age bins
        titanic['AgeBin'] = pd.cut(titanic['Age'], bins=age_bins, labels=age_bins[:-1])
        fig_kde = px.histogram(titanic, x='Age', color='Survived', animation_frame='AgeBin', nbins=30,
                              title="Animated Age Distribution by Survival", marginal="rug")
        st.plotly_chart(fig_kde)

        # 4. Virtual Deck Map (Simulated)
        st.subheader("Virtual Titanic Deck Map")
        deck_data = titanic.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
        fig_deck = px.scatter(deck_data, x='Pclass', y='Count', color='Survived', size='Count',
                             title="Passenger Distribution by Class (Simulated Deck Map)",
                             labels={'Pclass': 'Deck/Class'})
        st.plotly_chart(fig_deck)
        st.markdown("**Interact**: Zoom, pan, or hover to explore passenger distributions.")

# Predict Your Fate Section
elif section == "Predict Your Fate":
    st.header("Predict Your Fate")
    st.markdown("Step into a passenger's shoes. Enter your details to see if you'd survive the Titanic disaster.")
    
    if model is not None:
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First Class, 3 = Third Class")
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 0, 100, 30)
        with col2:
            sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
            parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
            fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=30.0)
            embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
        
        # Voice input toggle
        voice_input = st.checkbox("Enable Voice Narration (Beta)")
        if voice_input:
            st.info("Voice narration is enabled. Results will be narrated (simulated for now).")

        if st.button("Predict My Fate"):
            try:
                input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
                with st.spinner("Consulting the Time Machine..."):
                    prediction = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0][1]
                
                # Display results with flair
                if prediction == 1:
                    st.balloons()
                    st.success("You Survived! 🎉")
                    if voice_input:
                        st.write("(Voice: 'Congratulations, you would have survived the Titanic!')")
                else:
                    st.error("You Did Not Survive. 😔")
                    if voice_input:
                        st.write("(Voice: 'Sadly, you would not have survived the Titanic.')")
                st.write(f"Survival Probability: {prob:.2%}")

                # AI Explanation
                st.subheader("Why This Prediction?")
                explanation = f"Based on historical data, {'women' if sex == 'female' else 'men'} in {'first' if pclass == 1 else 'second' if pclass == 2 else 'third'} class with an age of {age} had a {'higher' if prob > 0.5 else 'lower'} chance of survival. Your fare of ${fare:.2f} and embarkation at {embarked} also influenced the outcome."
                st.write(explanation)

                # Download Prediction Report
                passenger_details = {"Class": pclass, "Sex": sex, "Age": age, "Siblings/Spouses": sibsp, 
                                    "Parents/Children": parch, "Fare": fare, "Embarked": embarked}
                pdf_buffer = generate_pdf_report(prediction, prob, passenger_details)
                st.download_button("Download Prediction Report", pdf_buffer, "titanic_prediction.pdf", "application/pdf")
            except Exception as e:
                st.error(f"Error in prediction: {e}")
    else:
        st.error("Model not loaded. Please ensure 'model.pkl' exists.")

# Model Insights Section
elif section == "Model Insights":
    st.header("Model Insights")
    if titanic is not None and model is not None:
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
        
        # Metrics
        y_pred = model.predict(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame([metrics])
        st.write(metrics_df)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix - Random Forest')
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        fig_imp = px.bar(importance, x='Importance', y='Feature', title="Feature Importance in Random Forest")
        st.plotly_chart(fig_imp)

# Survival Challenge Section
elif section == "Survival Challenge":
    st.header("Survival Challenge")
    st.markdown("Compete to create a passenger profile with the highest survival probability! Try different combinations and see your rank.")
    
    if 'leaderboard' not in st.session_state:
        st.session_state.leaderboard = []

    if model is not None:
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], key="challenge_pclass")
            sex = st.selectbox("Sex", ["male", "female"], key="challenge_sex")
            age = st.slider("Age", 0, 100, 30, key="challenge_age")
        with col2:
            sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=8, value=0, key="challenge_sibsp")
            parch = st.number_input("Parents/Children", min_value=0, max_value=6, value=0, key="challenge_parch")
            fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=30.0, key="challenge_fare")
            embarked = st.selectbox("Port", ["Southampton", "Cherbourg", "Queenstown"], key="challenge_embarked")
        
        if st.button("Submit to Leaderboard"):
            try:
                input_data = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
                prob = model.predict_proba(input_data)[0][1]
                st.session_state.leaderboard.append({
                    'Name': st.text_input("Enter your name", value="Anonymous", key="challenge_name"),
                    'Probability': prob,
                    'Details': f"Class: {pclass}, Sex: {sex}, Age: {age}, Fare: {fare}"
                })
                st.success(f"Your survival probability: {prob:.2%}")
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        
        # Display Leaderboard
        st.subheader("Leaderboard")
        if st.session_state.leaderboard:
            leaderboard_df = pd.DataFrame(st.session_state.leaderboard)
            leaderboard_df = leaderboard_df.sort_values(by='Probability', ascending=False).head(10)
            st.write(leaderboard_df[['Name', 'Probability', 'Details']])

# Footer with X Integration
st.markdown("---")
st.markdown("**Trending on X**: What's the world saying about the Titanic?")
# Simulated X posts (replace with real API call if available)
x_posts = [
    {"user": "HistoryBuff", "post": "Did you know the Titanic had only 20 lifeboats for over 2,200 passengers? #TitanicFacts"},
    {"user": "DataNerd", "post": "Analyzing Titanic survival data with ML is fascinating! Women and children first really mattered. #DataScience"}
]
for post in x_posts:
    st.write(f"@{post['user']}: {post['post']}")
st.markdown("Follow the conversation on [X](https://x.com) for more Titanic insights!")
