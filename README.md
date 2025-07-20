 # 💼 Employee Salary Prediction (India)

This project is a machine learning web application built using **Streamlit** to predict employee salaries in India based on features like job title, experience, education level, location, and more.

It uses a synthetic yet realistic dataset and supports predictions via an interactive UI.

---

## 📂 Folder Structure

```
Employee-Salary-Prediction/
├── app.py                      # Streamlit app file
├── models/
│   ├── trained_model.pkl       # Trained regression model
│   ├── scaler.pkl              # StandardScaler used for inputs
│   └── feature_names.pkl       # Column order used during training
├── data/
│   └── employee_dataset_india.csv  # Cleaned + engineered dataset
├── notebooks/
│   └── model_train_and_export.ipynb  # EDA + training + export
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/employee-salary-predictor.git
cd employee-salary-predictor
```

### 2. Set up virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🎯 Features

- Predict salary based on:
  - Job title
  - Location
  - Education level
  - Industry & company size
  - Remote work, job level, certifications
- Multiple model comparison (Random Forest, Gradient Boosting, XGBoost)
- R² Score, RMSE, and Actual vs Predicted visualizations

---

## 📊 Model Info

The model was trained using:
- `RandomForestRegressor` / `GradientBoostingRegressor` / `XGBRegressor`
- 1000+ synthetic data points
- StandardScaler for numeric features
- One-hot encoding for categoricals

Best model based on R² was saved as `trained_model.pkl`

---

## 🔧 Tools Used

- Python
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Matplotlib / Seaborn
- Streamlit

---

## 📈 Sample Prediction

> 🎓 A Senior Data Scientist in Bangalore with a PhD and 10+ years experience from a Tier 1 university with certification earns ₹25–40 Lakhs annually.

---

## 📝 License

MIT License

---

## 🙋‍♂️ Author

Built by [Rahul Das]

Feel free to contribute or reach out!
