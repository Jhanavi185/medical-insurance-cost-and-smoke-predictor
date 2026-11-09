
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}
[data-testid="stSidebar"] {
    background-color: #102A43;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    color: #123B5D;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    color: #627D98;
    font-size: 17px;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #159D91;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL BUNDLE ----------------
@st.cache_resource
def load_bundle():
    return joblib.load("insurance_project_bundle.pkl")

try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Could not load model bundle: {e}")
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

    st.markdown(
        '<p class="main-title">Medical Insurance Analytics</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Machine Learning Based Prediction System</p>',
        unsafe_allow_html=True
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Records", len(df))

    with c2:
        st.metric("Features", 6)

    with c3:
        st.metric("ML Models", len(reg_models) + len(clf_models))

    st.subheader("Project Overview")

    st.write("""
    This project uses Machine Learning algorithms to estimate
    medical insurance charges and demonstrate smoker-status
    classification using demographic information.

    The system uses data preprocessing, exploratory data analysis,
    model training, model evaluation and hyperparameter tuning.
    """)

    st.subheader("Project Objectives")

    st.markdown("""
    - Predict estimated medical insurance charges.
    - Demonstrate smoker-status classification.
    - Compare different Machine Learning algorithms.
    - Visualize insurance dataset patterns.
    """)

# ---------------- INSURANCE PREDICTION ----------------
elif page == "Insurance Prediction":

    st.title("💰 Insurance Cost Prediction")

    st.write("Enter the details below to estimate insurance charges.")

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

        prediction = best_reg_model.predict(reg_input)[0]

        st.success("Prediction completed!")

        st.metric(
            "Estimated Insurance Charges",
            f"${prediction:,.2f}"
        )

        st.caption(
            "This is an educational estimate, not an actual insurance quotation."
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
            ["southwest", "southeast", "northwest", "northeast"],
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

        result = best_clf_model.predict(clf_input)[0]

        if result == 1:
            st.warning("Model prediction: Smoker")
        else:
            st.success("Model prediction: Non-Smoker")

        st.caption(
            "This is an experimental ML prediction, not a medical test."
        )

# ---------------- DATA ANALYSIS ----------------
elif page == "Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

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

        ax.tick_params(axis="x", rotation=15)

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

        ax.tick_params(axis="x", rotation=15)

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
