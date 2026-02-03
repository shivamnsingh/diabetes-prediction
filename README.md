🩺 Diabetes Prediction System
A streamlined Machine Learning pipeline to predict diabetes using medical data. This project demonstrates data preprocessing, feature scaling, and classification using SVM and Logistic Regression.

✨ Features
Data Scaling: Uses StandardScaler for optimized model performance.

Dual Models: Implements both Support Vector Machines (SVM) and Logistic Regression.

Fast Inference: Predicts outcomes for new patients in milliseconds.

🛠️ Tech Stack
Python

Pandas & NumPy (Data Handling)

Scikit-learn (Machine Learning)

🚀 Quick Start
Clone & Load: Load the diabetes.csv dataset.

Train: Run the script to scale features and fit the SVM model.

Predict:

Python
# Example input data
data = (5, 116, 74, 0, 0, 25.6, 0.201, 30)

# Standardize and Predict
scaled_data = scalr.transform(np.asarray(data).reshape(1,-1))
result = model.predict(scaled_data)

print("Diabetic" if result[0] == 1 else "Non-Diabetic")
📊 Dataset Stats
Entries: 768

Features: 8 (Glucose, BMI, Age, etc.)

Classes: 2 (Diabetic / Non-Diabetic)
