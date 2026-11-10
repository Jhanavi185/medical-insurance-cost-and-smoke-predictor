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

## Model Bundle Structure

The notebook builds project_bundle and saves it as insurance_project_bundle.pkl with joblib. The application reads these 13 keys:

| Key | Purpose |
|---|---|
| best_reg_model | Selected estimator used by the app to predict insurance charges. |
| best_clf_model | Selected estimator used by the app to classify smoker status. |
| best_reg_name | Name of the selected regression model; the notebook output identifies Decision Tree. |
| best_clf_name | Name of the selected classification model; the notebook output identifies Decision Tree. |
| reg_models | Regression candidate dictionary. The notebook updates it with tuned estimators for model names that were tuned. |
| clf_models | Classification candidate dictionary, updated with tuned estimators for the tuned model names. |
| reg_results | Pre-tuning regression results table, with Model, MAE, MSE, RMSE, and R2 Score. |
| clf_results | Pre-tuning classification results table, with Model, Accuracy, Precision, Recall, and F1 Score. |
| tuned_reg_results | Tuned regression results table, with Model, MAE, RMSE, and R2 Score. |
| tuned_clf_results | Tuned classification results table, with Model, Accuracy, Precision, Recall, and F1 Score. |
| reg_columns | Ordered regression feature names taken from the training data columns. |
| clf_columns | Ordered classification feature names taken from the training data columns. |
| dataset | Cleaned DataFrame used by the app for record counts and EDA. The notebook removes one duplicate row, leaving 1,337 rows. |

### Preprocessing and model input features

The notebook loads insurance.csv, checks for missing values and duplicates, and drops duplicate rows. Its saved inspection output reports no missing values and one duplicate among the original 1,338 rows. It one-hot encodes sex, smoker, and region with drop_first=True and integer indicators. The omitted categories are female, no, and northeast respectively, represented by all-zero indicators. The classification target is smoker_yes: 0 means Non-Smoker and 1 means Smoker.

The exact feature columns and order in the notebook are:

- **Regression:** age, bmi, children, sex_male, region_northwest, region_southeast, region_southwest, smoker_yes.
- **Classification:** age, bmi, children, sex_male, region_northwest, region_southeast, region_southwest.

The app constructs these values and reindexes each input row to reg_columns or clf_columns before prediction. Logistic Regression and KNN classification models use StandardScaler in a pipeline; the notebook does not scale the regression inputs or tree-based classifiers.

### Train/test split and evaluation

Both tasks use an 80/20 train/test split with random_state=42. The classification split also uses stratify=y_clf. With 1,337 cleaned records, the notebook output shows 1,069 training rows and 268 test rows for each task.

The notebook evaluates regression models using MAE, MSE, RMSE, and R2 Score. Tuned regression results include MAE, RMSE, and R2 Score. Classification evaluation uses Accuracy, Precision, Recall, and F1 Score; precision, recall, and F1 use zero_division=0. The notebook also computes a confusion matrix for the Random Forest classifier.

### Grid search and best-model selection

GridSearchCV uses 5-fold cross-validation on the training split. Regression tuning covers Decision Tree and Random Forest with neg_root_mean_squared_error scoring. Classification tuning covers Logistic Regression, Decision Tree, Random Forest, and KNN with F1 scoring.

The notebook combines the original and tuned candidate dictionaries, then selects the regression model with the lowest RMSE on the test set and the classification model with the highest F1 score on the test set. The saved notebook output names Decision Tree for both tasks.

**Methodological limitation:** the test set is used to choose the best models after it has already been used for evaluation. This makes the test set part of model selection, so its reported performance is no longer an independent final estimate and may be optimistic. A separate validation set or nested cross-validation would avoid using the final test set for selection.

### What the application displays

The app displays the pre-tuning reg_results and clf_results tables and charts R2 Score and Accuracy when those columns are available. It does not currently display tuned_reg_results or tuned_clf_results.

The notebook imports pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and joblib, and uses google.colab.files for Colab upload/download. The notebook does not record exact dependency versions; requirements.txt also does not pin versions.

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