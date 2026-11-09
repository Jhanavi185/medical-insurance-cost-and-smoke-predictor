# 🏥 Medical Insurance Cost and Smoker Status Predictor

A Machine Learning-based web application that predicts medical insurance charges and classifies smoker status using demographic and health-related information.

## 📌 Project Overview

The **Medical Insurance Cost and Smoker Status Predictor** is a Machine Learning project designed to analyze insurance-related data and generate predictions using trained regression and classification models.

The application provides an interactive interface where users can enter personal details to estimate medical insurance costs and predict smoker status. It also includes data visualization and model performance comparison.

## 🎯 Objectives

- Predict estimated medical insurance charges using regression models.
- Classify smoker status using classification models.
- Compare multiple Machine Learning algorithms.
- Perform Exploratory Data Analysis (EDA).
- Visualize important patterns and relationships in the dataset.
- Evaluate model performance using appropriate evaluation metrics.

## 🚀 Features

### 1. Medical Insurance Cost Prediction
- Accepts age, gender, BMI, number of children, smoking status and region.
- Predicts estimated medical insurance charges.
- Displays the predicted insurance amount.

### 2. Smoker Status Classification
- Accepts demographic and health-related inputs.
- Uses a trained classification model.
- Predicts whether the individual is a smoker or non-smoker.

### 3. Data Analysis
- Dataset preview.
- Insurance charges distribution.
- Insurance charges by smoking status.
- BMI versus insurance charges.
- Correlation heatmap.

### 4. Model Performance
- Regression model comparison.
- Classification model comparison.
- R² score comparison.
- Classification accuracy comparison.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Web application interface |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |

## 📂 Project Structure

```text
medical-insurance-cost-and-smoke-predictor/
│
├── app.py
├── insurance_project_bundle.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Jhanavi185/medical-insurance-cost-and-smoke-predictor.git
```

### Step 2: Navigate to the Project Folder

```bash
cd medical-insurance-cost-and-smoke-predictor
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

### Step 5: Open the Application

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open this URL in your browser to use the application.

## 🤖 Machine Learning Approach

The project uses two Machine Learning approaches:

**Regression:**
- Estimates medical insurance charges.
- Compares multiple regression models.
- Uses R² score for model performance comparison.

**Classification:**
- Predicts smoker status.
- Compares multiple classification models.
- Uses accuracy for model performance comparison.

The trained models and supporting dataset are stored in the serialized model bundle.

## 📊 Application Modules

1. Home
2. Insurance Cost Prediction
3. Smoker Status Classification
4. Exploratory Data Analysis
5. Model Performance Evaluation

## 🔮 Future Enhancements

- Deploy the application online.
- Improve prediction accuracy through additional model tuning.
- Add more interactive visualizations.
- Expand the dataset for better model generalization.
- Enhance the user interface and reporting features.

## ⚠️ Disclaimer

This project is developed for educational and academic purposes. The predicted insurance charges are estimates and should not be considered actual insurance quotations. Smoker classification is an experimental Machine Learning prediction and not a medical test.

## 👩‍💻 Author

**Jhanavi K**

GitHub: [Jhanavi185](https://github.com/Jhanavi185)

---

⭐ If you find this project interesting, consider giving the repository a star!