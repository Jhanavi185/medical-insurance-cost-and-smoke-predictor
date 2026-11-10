
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)
# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
:root {
    --navy: #102A43;
    --navy-light: #183B56;
    --teal: #0F9D91;
    --teal-dark: #087F78;
    --ink: #18324B;
    --muted: #627D98;
    --canvas: #F3F7FB;
    --line: #E2EAF2;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: var(--canvas);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: rgba(243,247,251,0.92);
}

.main .block-container {
    padding-top: 2.1rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Main text */
.main h1, .main h2, .main h3, .main h4,
.main [data-testid="stMarkdownContainer"] p,
.main [data-testid="stMarkdownContainer"] li,
.main label,
.main [data-testid="stCaptionContainer"] {
    color: var(--ink) !important;
}

.main h1 {
    font-weight: 750;
    letter-spacing: -0.7px;
}

.main h2, .main h3 {
    font-weight: 700;
}

.main [data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg, #102A43 0%, #153B55 100%
    );
}

[data-testid="stSidebar"] * {
    color: #F4F8FC !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C8D8E6 !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,.18);
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 5px 8px;
    border-radius: 8px;
}

/* Hero banner */
.hero-panel {
    background: linear-gradient(
        115deg, #102A43 0%, #18566B 100%
    );
    border-radius: 20px;
    padding: 30px 34px;
    margin: 8px 0 24px 0;
    box-shadow: 0 10px 26px rgba(16,42,67,.13);
}

.hero-kicker {
    color: #A8E6DE !important;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    font-size: .78rem;
    font-weight: 800;
    margin: 0 0 10px 0;
}

.hero-heading {
    color: #FFFFFF !important;
    font-size: 2.1rem;
    line-height: 1.18;
    font-weight: 800;
    margin: 0 0 10px 0;
}

.hero-copy {
    color: #D9E8F1 !important;
    font-size: 1rem;
    margin: 0;
    max-width: 760px;
}

/* KPI cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2EAF2;
    border-left: 4px solid #0F9D91;
    border-radius: 14px;
    padding: 17px 20px;
    box-shadow: 0 5px 18px rgba(16,42,67,.055);
}

[data-testid="stMetricLabel"] p {
    color: #627D98 !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #102A43 !important;
    font-weight: 800;
}

/* Content cards */
.section-card {
    background: #FFFFFF;
    border: 1px solid #E2EAF2;
    border-radius: 15px;
    padding: 20px 22px;
    min-height: 180px;
    box-shadow: 0 5px 18px rgba(16,42,67,.045);
}

.section-card h3 {
    color: #102A43 !important;
    font-size: 1.08rem;
    margin: 0 0 10px 0;
}

.section-card p,
.section-card li {
    color: #526B80 !important;
    font-size: .94rem;
    line-height: 1.65;
}

.section-card ul {
    padding-left: 20px;
    margin-bottom: 0;
}

.home-section-label {
    color: #627D98 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: .76rem;
    font-weight: 800;
    margin: 22px 0 10px 0;
}

/* Forms */
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E2EAF2;
    border-radius: 16px;
    padding: 22px 24px 18px;
    box-shadow: 0 5px 18px rgba(16,42,67,.045);
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #FFFFFF;
    color: #18324B;
    border-color: #D5E0EA;
    border-radius: 9px;
}

.stButton > button,
[data-testid="stFormSubmitButton"] button {
    background: #0F9D91;
    color: white !important;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    padding: .62rem 1.1rem;
    transition: .2s ease;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #087F78;
    border: 0;
    color: white !important;
}

/* Tables and charts */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #FFFFFF;
    border: 1px solid #E2EAF2;
    border-radius: 12px;
}

.main hr {
    border-color: #DCE6EF;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)
st.markdown(textwrap.dedent("""
<div class="hero-panel">
    <p class="hero-kicker">Machine learning • Healthcare analytics</p>
    <p class="hero-heading">Understand insurance costs.<br>Explore smarter predictions.</p>
    <p class="hero-copy">
        An interactive academic project for estimating medical insurance charges,
        exploring dataset patterns, and demonstrating smoker-status classification.
    </p>
</div>
"""), unsafe_allow_html=True)
@st.cache_resource
def load_bundle():
    bundle_path = Path(__file__).resolve().parent / "insurance_project_bundle.pkl"
    if not bundle_path.is_file():
        raise FileNotFoundError

    return joblib.load(bundle_path)


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("The model bundle file was not found. Place insurance_project_bundle.pkl beside app.py.")
    st.stop()
except Exception:
    st.error("The model bundle could not be loaded. Check that the file is valid and dependencies are compatible.")
    st.stop()


if not isinstance(bundle, dict):
    st.error("The model bundle has an invalid format. Expected a dictionary.")
    st.stop()

required_bundle_keys = [
    "best_reg_model",
    "best_clf_model",
    "reg_models",
    "clf_models",
    "reg_results",
    "clf_results",
    "reg_columns",
    "clf_columns",
    "dataset",
]
missing_bundle_keys = [
    key for key in required_bundle_keys if key not in bundle
]
if missing_bundle_keys:
    st.error(
        "The model bundle is incomplete. Missing required keys: "
        + ", ".join(missing_bundle_keys)
    )
    st.stop()

for model_key, model_label in (
    ("best_reg_model", "regression"),
    ("best_clf_model", "classification"),
):
    model = bundle[model_key]
    try:
        has_predict = callable(getattr(model, "predict", None))
    except Exception:
        has_predict = False
    if model is None or not has_predict:
        st.error(
            f"The {model_label} model in the bundle is invalid. "
            "Restore a valid project bundle."
        )
        st.stop()

for columns_key, task_label in (
    ("reg_columns", "regression"),
    ("clf_columns", "classification"),
):
    columns = bundle[columns_key]
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(name, str) or not name.strip() for name in columns)
        or len(set(columns)) != len(columns)
    ):
        st.error(
            f"The {task_label} feature columns in the bundle are invalid. "
            "Restore a valid project bundle."
        )
        st.stop()


expected_feature_columns = {
    "reg_columns": [
        "age",
        "bmi",
        "children",
        "sex_male",
        "region_northwest",
        "region_southeast",
        "region_southwest",
        "smoker_yes",
    ],
    "clf_columns": [
        "age",
        "bmi",
        "children",
        "sex_male",
        "region_northwest",
        "region_southeast",
        "region_southwest",
    ],
}
for columns_key, task_label in (
    ("reg_columns", "regression"),
    ("clf_columns", "classification"),
):
    if bundle[columns_key] != expected_feature_columns[columns_key]:
        st.error(
            f"The {task_label} feature schema is incompatible with this "
            "application. Restore a compatible project bundle."
        )
        st.stop()

for model_key, columns_key, task_label in (
    ("best_reg_model", "reg_columns", "regression"),
    ("best_clf_model", "clf_columns", "classification"),
):
    model = bundle[model_key]
    try:
        model_feature_count = getattr(model, "n_features_in_", None)
        model_feature_names = getattr(model, "feature_names_in_", None)
    except Exception:
        st.error(
            f"The {task_label} model feature schema could not be verified. "
            "Restore a compatible project bundle."
        )
        st.stop()

    if model_feature_count is not None and (
        not isinstance(model_feature_count, (int, np.integer))
        or model_feature_count != len(bundle[columns_key])
    ):
        st.error(
            f"The {task_label} model expects a different number of features. "
            "Restore a compatible project bundle."
        )
        st.stop()

    if model_feature_names is not None:
        try:
            names_match = list(model_feature_names) == bundle[columns_key]
        except Exception:
            names_match = False
        if not names_match:
            st.error(
                f"The {task_label} model feature names or order do not match "
                "the application schema. Restore a compatible project bundle."
            )
            st.stop()

dataset_data = bundle["dataset"]
if not isinstance(dataset_data, pd.DataFrame):
    st.error("The dataset in the model bundle is invalid. Expected a table of records.")
    st.stop()
if dataset_data.empty:
    st.error("The dataset in the model bundle is empty.")
    st.stop()

required_dataset_columns = ["charges", "smoker", "bmi"]
missing_dataset_columns = [
    column for column in required_dataset_columns
    if column not in dataset_data.columns
]
if missing_dataset_columns:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_dataset_columns)
    )
    st.stop()
if not pd.api.types.is_numeric_dtype(dataset_data["charges"]):
    st.error("The charges column in the dataset must contain numeric values.")
    st.stop()
if not pd.api.types.is_numeric_dtype(dataset_data["bmi"]):
    st.error("The BMI column in the dataset must contain numeric values.")
    st.stop()

reg_results_data = bundle["reg_results"]
if not isinstance(reg_results_data, pd.DataFrame):
    st.error("The regression evaluation results in the model bundle are invalid.")
    st.stop()
if reg_results_data.empty:
    st.error("The regression evaluation results are empty.")
    st.stop()

required_reg_result_columns = ["Model", "R2 Score"]
missing_reg_result_columns = [
    column for column in required_reg_result_columns
    if column not in reg_results_data.columns
]
if missing_reg_result_columns:
    st.error(
        "The regression evaluation results are missing required columns: "
        + ", ".join(missing_reg_result_columns)
    )
    st.stop()
if not pd.api.types.is_numeric_dtype(reg_results_data["R2 Score"]):
    st.error("The R2 Score column in regression results must be numeric.")
    st.stop()

clf_results_data = bundle["clf_results"]
if not isinstance(clf_results_data, pd.DataFrame):
    st.error("The classification evaluation results in the model bundle are invalid.")
    st.stop()
if clf_results_data.empty:
    st.error("The classification evaluation results are empty.")
    st.stop()

required_clf_result_columns = ["Model", "Accuracy"]
missing_clf_result_columns = [
    column for column in required_clf_result_columns
    if column not in clf_results_data.columns
]
if missing_clf_result_columns:
    st.error(
        "The classification evaluation results are missing required columns: "
        + ", ".join(missing_clf_result_columns)
    )
    st.stop()
if not pd.api.types.is_numeric_dtype(clf_results_data["Accuracy"]):
    st.error("The Accuracy column in classification results must be numeric.")
    st.stop()

# ---------------- EXTRACT MODELS AND DATA ----------------
best_reg_model = bundle["best_reg_model"]
best_clf_model = bundle["best_clf_model"]

reg_models = bundle["reg_models"]
clf_models = bundle["clf_models"]

df = bundle["dataset"]

reg_columns = bundle["reg_columns"]
clf_columns = bundle["clf_columns"]


# ---------------- SIDEBAR ----------------
st.sidebar.title("🏥 MediPredict")
st.sidebar.write("Medical Insurance Analytics")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Insurance Prediction",
        "Smoker Classification",
        "Data Analysis",
        "Model Performance"
    ]
)


# ---------------- HOME ----------------
if page == "Home":


    # KPI Cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Dataset Records", f"{len(df):,}")

    with c2:
        st.metric("Input Features", "6")

    with c3:
        st.metric(
            "Models Evaluated",
            len(reg_models) + len(clf_models)
        )

    # Project Overview
    st.markdown(
        '<p class="home-section-label">About this project</p>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.25, 1])

    with left:
        st.html("""
        <div class="section-card">
            <h3>Project Overview</h3>

            <p>
                This application brings together two machine
                learning tasks: estimating medical insurance
                charges with a regression model and predicting
                smoker status with a classification model.
            </p>

            <p>
                It also provides exploratory data analysis and
                model comparison views to help understand the
                dataset and evaluate the trained models.
            </p>
        </div>
        """)

    with right:
        st.html("""
        <div class="section-card">
            <h3>Project Objectives</h3>

            <ul>
                <li>Estimate medical insurance charges.</li>
                <li>Demonstrate smoker-status classification.</li>
                <li>Compare machine learning algorithms.</li>
                <li>Explore patterns through data visualizations.</li>
            </ul>
        </div>
        """)

    # Application Features
    st.markdown(
        '<p class="home-section-label">Explore the application</p>',
        unsafe_allow_html=True
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.html("""
        <div class="section-card">
            <h3>💰 Insurance Prediction</h3>

            <p>
                Enter demographic and lifestyle details
                to estimate medical insurance charges.
            </p>
        </div>
        """)

    with q2:
        st.html("""
        <div class="section-card">
            <h3>🚭 Smoker Classification</h3>

            <p>
                Explore the trained model's smoker-status
                prediction from the provided inputs.
            </p>
        </div>
        """)

    with q3:
        st.html("""
        <div class="section-card">
            <h3>📊 Analysis & Performance</h3>

            <p>
                Review dataset visualizations and compare
                regression and classification models.
            </p>
        </div>
        """)

    st.caption(
        "Educational project only. Predictions are estimates "
        "and are not medical advice or actual insurance quotations."
    )


# ---------------- INSURANCE PREDICTION ----------------
elif page == "Insurance Prediction":

    st.title("💰 Insurance Cost Prediction")

    st.write(
        "Enter the details below to estimate insurance charges."
    )

    with st.form("insurance_form"):

        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=25
            )

            sex = st.selectbox(
                "Gender",
                ["female", "male"]
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=27.5
            )

        with c2:
            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=0
            )

            smoker = st.selectbox(
                "Smoking Status",
                ["no", "yes"]
            )

            region = st.selectbox(
                "Region",
                [
                    "southwest",
                    "southeast",
                    "northwest",
                    "northeast"
                ]
            )

        submitted = st.form_submit_button(
            "Predict Insurance Charges",
            use_container_width=True
        )

    if submitted:

        user_data = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex_male": int(sex == "male"),
            "smoker_yes": int(smoker == "yes"),
            "region_northwest": int(region == "northwest"),
            "region_southeast": int(region == "southeast"),
            "region_southwest": int(region == "southwest")
        }])

        reg_input = user_data.reindex(
            columns=reg_columns,
            fill_value=0
        )

        try:
            prediction_values = np.asarray(
                best_reg_model.predict(reg_input)
            )
        except Exception:
            st.error("Insurance cost prediction failed. Please try again.")
        else:
            if prediction_values.size != 1:
                st.error(
                    "The model returned an invalid number of insurance "
                    "cost predictions. Please try again."
                )
            elif (
                not np.issubdtype(prediction_values.dtype, np.number)
                or np.issubdtype(prediction_values.dtype, np.complexfloating)
            ):
                st.error(
                    "The model returned an invalid insurance cost. "
                    "Please try again."
                )
            else:
                prediction = prediction_values.item()
                if not np.isfinite(prediction):
                    st.error(
                        "The model returned a non-finite insurance cost. "
                        "Please try again."
                    )
                elif prediction < 0:
                    st.error(
                        "The model returned a negative insurance cost. "
                        "Please try again."
                    )
                else:
                    st.success("Prediction completed!")

                    st.metric(
                        "Estimated Insurance Charges",
                        f"${prediction:,.2f}"
                    )

                    st.caption(
                        "This is an educational estimate, "
                        "not an actual insurance quotation."
                    )


# ---------------- SMOKER CLASSIFICATION ----------------
elif page == "Smoker Classification":

    st.title("🚭 Smoker Status Classification")

    st.write(
        "Demonstration of the trained classification model."
    )

    with st.form("smoker_form"):

        age = st.number_input(
            "Age",
            18,
            100,
            25,
            key="smoker_age"
        )

        sex = st.selectbox(
            "Gender",
            ["female", "male"],
            key="smoker_sex"
        )

        bmi = st.number_input(
            "BMI",
            10.0,
            60.0,
            27.5,
            key="smoker_bmi"
        )

        children = st.number_input(
            "Number of Children",
            0,
            10,
            0,
            key="smoker_children"
        )

        region = st.selectbox(
            "Region",
            [
                "southwest",
                "southeast",
                "northwest",
                "northeast"
            ],
            key="smoker_region"
        )

        submitted = st.form_submit_button(
            "Classify",
            use_container_width=True
        )

    if submitted:

        user_data = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex_male": int(sex == "male"),
            "region_northwest": int(region == "northwest"),
            "region_southeast": int(region == "southeast"),
            "region_southwest": int(region == "southwest")
        }])

        clf_input = user_data.reindex(
            columns=clf_columns,
            fill_value=0
        )

        try:
            classification_values = np.asarray(
                best_clf_model.predict(clf_input)
            )
        except Exception:
            st.error("Smoker classification failed. Please try again.")
        else:
            if classification_values.size != 1:
                st.error(
                    "The model returned an invalid number of smoker "
                    "classification results. Please try again."
                )
            else:
                result = classification_values.item()
                valid_label = (
                    isinstance(
                        result,
                        (int, float, np.integer, np.floating)
                    )
                    and not isinstance(result, (bool, np.bool_))
                    and np.isfinite(result)
                    and result in (0, 1)
                )
                if not valid_label:
                    st.error(
                        "The model returned an unexpected smoker label. "
                        "Expected 0 or 1."
                    )
                elif result == 1:
                    st.warning("Model prediction: Smoker")
                else:
                    st.success("Model prediction: Non-Smoker")

                if valid_label:
                    st.caption(
                        "This is an experimental ML prediction, "
                        "not a medical test."
                    )


# ---------------- DATA ANALYSIS ----------------
elif page == "Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Insurance Charges Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.histplot(
        df["charges"],
        kde=True,
        ax=ax,
        color="#159D91"
    )

    ax.set_title("Distribution of Insurance Charges")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Insurance Charges by Smoker Status")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.boxplot(
        data=df,
        x="smoker",
        y="charges",
        ax=ax
    )

    ax.set_title("Charges Comparison by Smoking Status")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("BMI vs Insurance Charges")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.scatterplot(
        data=df,
        x="bmi",
        y="charges",
        hue="smoker",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)


# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.subheader("Regression Models")

    reg_results = bundle["reg_results"].copy()

    if "Linear Regression" not in reg_results["Model"].values:
        st.warning("Regression results are incomplete.")

    st.dataframe(
        reg_results.round(3),
        use_container_width=True
    )

    st.subheader("Classification Models")

    clf_results = bundle["clf_results"].copy()

    st.dataframe(
        clf_results.round(4),
        use_container_width=True
    )

    st.subheader("Regression R2 Score Comparison")

    if "R2 Score" in reg_results.columns:

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.barplot(
            data=reg_results,
            x="Model",
            y="R2 Score",
            ax=ax
        )

        ax.tick_params(
            axis="x",
            rotation=15
        )

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Classification Accuracy Comparison")

    if "Accuracy" in clf_results.columns:

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.barplot(
            data=clf_results,
            x="Model",
            y="Accuracy",
            ax=ax
        )

        ax.tick_params(
            axis="x",
            rotation=15
        )

        st.pyplot(fig)
        plt.close(fig)


# ---------------- FOOTER ----------------
st.sidebar.divider()

st.sidebar.caption(
    "Medical Insurance ML Mini Project"
)

st.sidebar.caption(
    "Educational use only"
)
