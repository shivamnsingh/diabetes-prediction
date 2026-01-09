import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Split features and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions and accuracy
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_y_pred)
test_acc = accuracy_score(y_test, test_y_pred)

# Streamlit app
def app():
    img = Image.open("img.jpg")
    img = img.resize((200, 200))
    st.image(img, caption="Diabetes Image", width=200)

    st.title('🩺 Diabetes Prediction App')

    # Sidebar inputs
    st.sidebar.title('Enter Patient Details')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Predict based on input
    input_data = np.array([[preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    # Output prediction
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.warning('⚠️ This person is likely to have diabetes.')
    else:
        st.success('✅ This person is not likely to have diabetes.')

    # Display dataset summary
    st.header('Dataset Summary')
    st.write(diabetes_dataset.describe())

    st.header('Sample Data')
    st.dataframe(diabetes_dataset.head())

    # Display model performance
    st.header('Model Performance')
    st.write(f"**Train Accuracy:** {train_acc*100:.2f}%")
    st.write(f"**Test Accuracy:** {test_acc*100:.2f}%")

if __name__ == '__main__':
    app()
