
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

    # Live statistics calculated from the dataset
    total_records = len(df)
    average_charges = float(df["charges"].mean())
    highest_charges = float(df["charges"].max())
    smoker_count = int(
        df["smoker"].astype(str).str.strip().str.lower().eq("yes").sum()
    )
    smoker_percentage = smoker_count / total_records * 100 if total_records else 0
    average_bmi = float(df["bmi"].mean())

    st.markdown(
        '<p class="home-section-label">Dataset at a glance</p>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Dataset Records", f"{total_records:,}")
    with k2:
        st.metric("Average Insurance Charges", f"${average_charges:,.0f}")
    with k3:
        st.metric("Highest Insurance Charges", f"${highest_charges:,.0f}")
    with k4:
        st.metric("Smoker Percentage", f"{smoker_percentage:.1f}%")

    # Charts based on the actual dataset
    st.markdown(
        '<p class="home-section-label">Insights from the actual dataset</p>',
        unsafe_allow_html=True
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Insurance Charges by Smoker Status")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df,
            x="smoker",
            y="charges",
            hue="smoker",
            palette={"yes": "#0F9D91", "no": "#8AA6B8"},
            legend=False,
            ax=ax
        )
        ax.set_xlabel("Smoker")
        ax.set_ylabel("Insurance charges ($)")
        ax.set_title("Smokers and non-smokers")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "This boxplot compares the distribution of insurance charges "
            "for smokers and non-smokers."
        )

    with chart_right:
        st.subheader("Age vs Insurance Charges")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(
            data=df,
            x="age",
            y="charges",
            hue="smoker",
            palette={"yes": "#0F9D91", "no": "#8AA6B8"},
            alpha=0.7,
            ax=ax
        )
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Insurance charges ($)")
        ax.set_title("How charges vary with age")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Each point represents one dataset record; colour indicates "
            "smoker status."
        )

    st.markdown(
        '<p class="home-section-label">More dataset facts</p>',
        unsafe_allow_html=True
    )
    fact1, fact2, fact3 = st.columns(3)
    with fact1:
        st.metric("Average BMI", f"{average_bmi:.1f}")
    with fact2:
        st.metric("Smokers in Dataset", f"{smoker_count:,}")
    with fact3:
        st.metric("Non-smokers in Dataset", f"{total_records - smoker_count:,}")

    # Best regression model is selected by highest R2.
    # Best classification model is selected by highest F1 when available,
    # because accuracy alone can be misleading for imbalanced classes.
    st.markdown(
        '<p class="home-section-label">Best model snapshot</p>',
        unsafe_allow_html=True
    )

    reg_home = bundle["reg_results"].copy()
    clf_home = bundle["clf_results"].copy()

    reg_home["R2 Score"] = pd.to_numeric(reg_home["R2 Score"], errors="coerce")
    best_reg_row = reg_home.loc[reg_home["R2 Score"].idxmax()]
    reg_name = str(best_reg_row["Model"])
    reg_score = float(best_reg_row["R2 Score"])

    if "F1 Score" in clf_home.columns:
        clf_home["F1 Score"] = pd.to_numeric(
            clf_home["F1 Score"], errors="coerce"
        )
    if (
        "F1 Score" in clf_home.columns
        and clf_home["F1 Score"].notna().any()
    ):
        best_clf_row = clf_home.loc[clf_home["F1 Score"].idxmax()]
        clf_name = str(best_clf_row["Model"])
        clf_score = float(best_clf_row["F1 Score"])
        clf_metric_label = "F1 Score"
    else:
        clf_home["Accuracy"] = pd.to_numeric(
            clf_home["Accuracy"], errors="coerce"
        )
        best_clf_row = clf_home.loc[clf_home["Accuracy"].idxmax()]
        clf_name = str(best_clf_row["Model"])
        clf_score = float(best_clf_row["Accuracy"])
        clf_metric_label = "Accuracy"

    model_left, model_right = st.columns(2)
    with model_left:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>🏆 Best Regression Model</h3>
                <p><strong>{reg_name}</strong></p>
                <p>R² score: <strong>{reg_score:.3f}</strong></p>
                <p>A higher R² score means the model explains more variation
                in insurance charges.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with model_right:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>🎯 Best Classification Model</h3>
                <p><strong>{clf_name}</strong></p>
                <p>{clf_metric_label}: <strong>{clf_score:.3f}</strong></p>
                <p>F1 score is prioritised when available because the smoker
                and non-smoker classes are imbalanced.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<p class="home-section-label">Explore the application</p>',
        unsafe_allow_html=True
    )

    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(
            """
            <div class="section-card">
                <h3>💰 Insurance Prediction</h3>
                <p>Estimate medical insurance charges using age, BMI,
                family details, region and smoking status.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with q2:
        st.markdown(
            """
            <div class="section-card">
                <h3>🚭 Smoker Classification</h3>
                <p>Explore the classification model's prediction using
                demographic and lifestyle inputs.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with q3:
        st.markdown(
            """
            <div class="section-card">
                <h3>📊 Data & Model Analysis</h3>
                <p>View additional dataset visualizations and compare
                the evaluated machine learning algorithms.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "Dataset statistics and charts are calculated from the project dataset. "
        "Model scores are from the saved evaluation results. Educational use only; "
        "predictions are not medical advice or actual insurance quotations."
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
                        "The model returned an invalid insurance cost. "
                        "Please try again."
                    )
                else:
                    st.success("Your prediction is ready!")

                    st.markdown("### 💰 Your Estimated Insurance Cost")

                    st.metric(
                        label="Estimated Annual Insurance Charges",
                        value=f"${prediction:,.2f}"
                    )

                    st.markdown("### 📋 Your Input Summary")

                    result_col1, result_col2, result_col3 = st.columns(3)

                    with result_col1:
                        st.write(f"**Age:** {age}")
                        st.write(f"**Gender:** {sex.title()}")

                    with result_col2:
                        st.write(f"**BMI:** {bmi}")
                        st.write(f"**Children:** {children}")

                    with result_col3:
                        st.write(f"**Smoker:** {smoker.title()}")
                        st.write(f"**Region:** {region.title()}")

                    st.caption(
                        "This is an educational ML prediction, "
                        "not an actual insurance quotation."
                    )
                
                    
                 




# ---------------- SMOKER CLASSIFICATION ----------------

elif page == "Smoker Classification":

    st.title("🚭 Smoker Status Classification")
    st.write(
        "Enter personal details to predict the smoking status "
        "using the trained machine learning model."
    )

    st.info(
        "This prediction is experimental and is intended "
        "for educational purposes only."
    )

    with st.form("smoker_form"):

        st.subheader("Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=25,
                key="smoker_age"
            )

            sex = st.selectbox(
                "Gender",
                ["female", "male"],
                key="smoker_sex"
            )

        with col2:
            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=27.5,
                key="smoker_bmi"
            )

            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=0,
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
            "Classify Smoking Status",
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

            if classification_values.size != 1:
                st.error("Invalid prediction returned by the model.")

            else:
                result = classification_values.item()

                if (
                    isinstance(result, (int, float, np.integer, np.floating))
                    and not isinstance(result, (bool, np.bool_))
                    and np.isfinite(result)
                    and result in (0, 1)
                ):

                    st.divider()
                    st.subheader("📋 Prediction Result")

                    if int(result) == 1:
                        st.warning("🚬 Predicted Status: Smoker")
                    else:
                        st.success("✅ Predicted Status: Non-Smoker")

                    # Show probability only if supported by the model
                    if hasattr(best_clf_model, "predict_proba"):

                        probabilities = best_clf_model.predict_proba(
                            clf_input
                        )[0]

                        classes = list(best_clf_model.classes_)

                        if 1 in classes:
                            smoker_probability = probabilities[
                                classes.index(1)
                            ]

                            st.metric(
                                "Estimated Smoker Probability",
                                f"{smoker_probability * 100:.2f}%"
                            )

                            st.progress(
                                float(smoker_probability)
                            )

                    st.markdown("### Your Input Summary")

                    summary_col1, summary_col2, summary_col3 = st.columns(3)

                    with summary_col1:
                        st.write(f"**Age:** {age}")
                        st.write(f"**Gender:** {sex.title()}")

                    with summary_col2:
                        st.write(f"**BMI:** {bmi}")
                        st.write(f"**Children:** {children}")

                    with summary_col3:
                        st.write(f"**Region:** {region.title()}")

                    st.caption(
                        "This is an experimental machine learning prediction, "
                        "not a medical test."
                    )

                else:
                    st.error(
                        "The model returned an unexpected result. "
                        "Expected 0 or 1."
                    )

        except Exception as e:
            st.error(f"Smoker classification failed: {e}")





    # ---------------- DATA ANALYSIS ----------------

elif page == "Data Analysis":

    st.title("📊 Exploratory Data Analysis")
    st.write(
            "Explore the medical insurance dataset using filters "
            "and interactive summaries."
        )

        # Keep the original dataset unchanged
    analysis_df = df.copy()

    st.markdown(
            '<p class="home-section-label">Filter the dataset</p>',
            unsafe_allow_html=True
        )

    filter_col1, filter_col2 = st.columns(2)

        # Age filter
    age_min = int(analysis_df["age"].min())
    age_max = int(analysis_df["age"].max())

    with filter_col1:
            age_range = st.slider(
                "Age range",
                min_value=age_min,
                max_value=age_max,
                value=(age_min, age_max)
            )

    analysis_df = analysis_df[
            analysis_df["age"].between(age_range[0], age_range[1])
        ]

        # Gender filter
    gender_options = sorted(analysis_df["sex"].dropna().unique())

    with filter_col2:
            selected_genders = st.multiselect(
                "Gender",
                gender_options,
                default=gender_options
            )

    analysis_df = analysis_df[
            analysis_df["sex"].isin(selected_genders)
        ]

    filter_col3, filter_col4 = st.columns(2)

        # Smoking filter
    smoker_options = sorted(analysis_df["smoker"].dropna().unique())

    with filter_col3:
            selected_smokers = st.multiselect(
                "Smoking Status",
                smoker_options,
                default=smoker_options
            )

    analysis_df = analysis_df[
            analysis_df["smoker"].isin(selected_smokers)
        ]

        # Region filter
    region_options = sorted(analysis_df["region"].dropna().unique())

    with filter_col4:
            selected_regions = st.multiselect(
                "Region",
                region_options,
                default=region_options
            )

    analysis_df = analysis_df[
            analysis_df["region"].isin(selected_regions)
        ]

    st.caption(
            f"Showing {len(analysis_df):,} of {len(df):,} records"
        )

    if analysis_df.empty:
            st.warning("No records match your selected filters.")

    else:

            # SUMMARY METRICS
            st.markdown(
                '<p class="home-section-label">Filtered Dataset Summary</p>',
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Total Records", f"{len(analysis_df):,}")

            m2.metric(
                "Average Charges",
                f"${analysis_df['charges'].mean():,.0f}"
            )

            m3.metric(
                "Average BMI",
                f"{analysis_df['bmi'].mean():.1f}"
            )

            smoker_percentage = (
                analysis_df["smoker"].eq("yes").mean() * 100
            )

            m4.metric(
                "Smoker Percentage",
                f"{smoker_percentage:.1f}%"
            )

            # CHART 1 AND 2
            st.markdown(
                '<p class="home-section-label">Dataset Insights</p>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Insurance Charges Distribution")

                fig, ax = plt.subplots(figsize=(7, 4))

                sns.histplot(
                    data=analysis_df,
                    x="charges",
                    kde=True,
                    color="#159D91",
                    ax=ax
                )

                ax.set_xlabel("Insurance Charges ($)")
                ax.set_ylabel("Number of Records")

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            with col2:
                st.subheader("Average Charges by Smoker Status")

                smoker_charges = (
                    analysis_df.groupby("smoker")["charges"]
                    .mean()
                    .reset_index()
                )

                fig, ax = plt.subplots(figsize=(7, 4))

                sns.barplot(
                    data=smoker_charges,
                    x="smoker",
                    y="charges",
                    hue="smoker",
                    palette={
                        "yes": "#0F9D91",
                        "no": "#8AA6B8"
                    },
                    legend=False,
                    ax=ax
                )

                ax.set_xlabel("Smoker Status")
                ax.set_ylabel("Average Charges ($)")

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            # CHART 3 AND 4
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("BMI vs Insurance Charges")

                fig, ax = plt.subplots(figsize=(7, 4))

                sns.scatterplot(
                    data=analysis_df,
                    x="bmi",
                    y="charges",
                    hue="smoker",
                    palette={
                        "yes": "#0F9D91",
                        "no": "#8AA6B8"
                    },
                    alpha=0.7,
                    ax=ax
                )

                ax.set_xlabel("BMI")
                ax.set_ylabel("Insurance Charges ($)")

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            with col2:
                st.subheader("Average Charges by Region")

                region_charges = (
                    analysis_df.groupby("region")["charges"]
                    .mean()
                    .reset_index()
                )

                fig, ax = plt.subplots(figsize=(7, 4))

                sns.barplot(
                    data=region_charges,
                    x="region",
                    y="charges",
                    color="#159D91",
                    ax=ax
                )

                ax.set_xlabel("Region")
                ax.set_ylabel("Average Charges ($)")
                ax.tick_params(axis="x", rotation=15)

                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            # FILTERED DATA
            st.markdown(
                '<p class="home-section-label">Filtered Records</p>',
                unsafe_allow_html=True
            )

            st.markdown(
                analysis_df.head(15).to_html(
                    index=False,
                    border=0
                ),
                unsafe_allow_html=True
            )

            # DOWNLOAD
            csv_data = analysis_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Filtered Dataset",
                data=csv_data,
                file_name="filtered_insurance_data.csv",
                mime="text/csv"
            )

            st.caption(
                "All charts and metrics update according to the selected filters. "
                "For educational analysis only."
            )



   


# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.title("📈 Machine Learning Model Performance")
    st.write(
        "Compare the performance of trained regression and "
        "classification models using evaluation metrics."
    )

    # Load saved model results
    reg_results = bundle["reg_results"].copy()
    clf_results = bundle["clf_results"].copy()

    # ---------------- BEST MODEL HIGHLIGHTS ----------------

    st.subheader("🏆 Best Model Highlights")

    reg_best = None
    clf_best = None

    if not reg_results.empty and "R2 Score" in reg_results.columns:
        reg_best = reg_results.loc[
            reg_results["R2 Score"].idxmax()
        ]

    score_column = None

    if not clf_results.empty:
        for col in ["F1 Score", "F1", "Accuracy"]:
            if col in clf_results.columns:
                score_column = col
                break

        if score_column:
            clf_best = clf_results.loc[
                clf_results[score_column].idxmax()
            ]

    card1, card2 = st.columns(2)

    with card1:
        if reg_best is not None:
            st.metric(
                "Best Regression Model",
                str(reg_best["Model"]),
                f"R² Score: {reg_best['R2 Score']:.3f}"
            )
        else:
            st.info("Regression results are unavailable.")

    with card2:
        if clf_best is not None:
            st.metric(
                "Best Classification Model",
                str(clf_best["Model"]),
                f"{score_column}: {clf_best[score_column]:.3f}"
            )
        else:
            st.info("Classification results are unavailable.")

    st.divider()

    # ---------------- REGRESSION RESULTS ----------------

    st.subheader("📊 Regression Model Evaluation")

    st.write(
        "Regression models predict the medical insurance charges "
        "of a person."
    )

    if not reg_results.empty:

        st.markdown("**Regression Results Table**")

        st.markdown(
            reg_results.round(4).to_html(index=False),
            unsafe_allow_html=True
        )

        if "R2 Score" in reg_results.columns:

            st.markdown("**R² Score Comparison**")

            fig, ax = plt.subplots(figsize=(9, 4))

            ax.bar(
                reg_results["Model"].astype(str),
                reg_results["R2 Score"]
            )

            ax.set_xlabel("Regression Models")
            ax.set_ylabel("R² Score")
            ax.set_title("Regression Model Comparison")
            ax.tick_params(axis="x", rotation=20)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if "RMSE" in reg_results.columns:

            st.markdown("**RMSE Comparison**")

            fig, ax = plt.subplots(figsize=(9, 4))

            ax.bar(
                reg_results["Model"].astype(str),
                reg_results["RMSE"]
            )

            ax.set_xlabel("Regression Models")
            ax.set_ylabel("RMSE")
            ax.set_title("Root Mean Squared Error Comparison")
            ax.tick_params(axis="x", rotation=20)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("No regression results are available.")

    st.divider()

    # ---------------- CLASSIFICATION RESULTS ----------------

    st.subheader("🎯 Classification Model Evaluation")

    st.write(
        "Classification models predict whether a person is a smoker "
        "or a non-smoker."
    )

    if not clf_results.empty:

        st.markdown("**Classification Results Table**")

        st.markdown(
            clf_results.round(4).to_html(index=False),
            unsafe_allow_html=True
        )

        available_metrics = [
            col for col in
            ["Accuracy", "Precision", "Recall", "F1 Score"]
            if col in clf_results.columns
        ]

        if available_metrics:

            st.markdown("**Classification Metrics Comparison**")

            fig, ax = plt.subplots(figsize=(10, 5))

            x = np.arange(len(clf_results))
            width = 0.8 / len(available_metrics)

            for i, metric in enumerate(available_metrics):
                ax.bar(
                    x + i * width,
                    clf_results[metric],
                    width,
                    label=metric
                )

            ax.set_xlabel("Classification Models")
            ax.set_ylabel("Score")
            ax.set_title("Classification Model Comparison")
            ax.set_xticks(
                x + width * (len(available_metrics) - 1) / 2
            )
            ax.set_xticklabels(
                clf_results["Model"].astype(str),
                rotation=20
            )
            ax.legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if (
            "Precision" in clf_results.columns
            and "Recall" in clf_results.columns
        ):

            st.markdown("**Precision vs Recall**")

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.scatter(
                clf_results["Recall"],
                clf_results["Precision"],
                s=100
            )

            for _, row in clf_results.iterrows():
                ax.annotate(
                    str(row["Model"]),
                    (row["Recall"], row["Precision"]),
                    xytext=(5, 5),
                    textcoords="offset points"
                )

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision and Recall Comparison")
            ax.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("No classification results are available.")

    st.divider()

    # ---------------- METRIC EXPLANATIONS ----------------

    st.subheader("📚 Understanding Evaluation Metrics")

    with st.expander("View metric explanations"):

        st.markdown("""
        **Regression Metrics**

        - **MAE:** Average absolute difference between actual and predicted charges.
        - **MSE:** Average squared prediction error.
        - **RMSE:** Square root of the mean squared error.
        - **R² Score:** Shows how well the model explains variations in charges.

        **Classification Metrics**

        - **Accuracy:** Percentage of correctly classified predictions.
        - **Precision:** How many predicted positive cases were actually positive.
        - **Recall:** How many actual positive cases were correctly identified.
        - **F1 Score:** Balance between precision and recall.
        """)

    st.caption(
        "Model results are based on the evaluation metrics stored "
        "in the trained project bundle. Educational use only."
    )

    # ---------------- FOOTER ----------------

    st.sidebar.divider()
    st.sidebar.caption("Medical Insurance ML Mini Project")
    st.sidebar.caption("Educational use only")
