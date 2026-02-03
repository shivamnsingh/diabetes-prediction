🩺 Diabetes Prediction System

Predicts diabetes using medical data with SVM and Logistic Regression.

<details> <summary>✨ Features</summary>

Data Scaling with StandardScaler

Dual Models: SVM & Logistic Regression

Fast predictions for new patients

</details> <details> <summary>🛠️ Tech Stack</summary>

Python, Pandas, NumPy

Scikit-learn

</details> <details> <summary>📂 Dataset</summary>

768 entries, 8 features

Target: Diabetic / Non-Diabetic

</details> <details> <summary>🚀 Quick Start</summary>
git clone <repo-url>
pip install -r requirements.txt

# Train & Predict
scaled = scaler.transform(np.asarray(data).reshape(1,-1))
result = model.predict(scaled)
print("Diabetic" if result[0]==1 else "Non-Diabetic")

</details> <details> <summary>💡 Future Improvements</summary>

Add Random Forest / XGBoost

Web interface

Feature visualization

</details>
