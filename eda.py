import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file for local development
load_dotenv()

# Try to get key from Streamlit Secrets first, then fallback to local .env
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# Graceful handling if no key found
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in Streamlit Secrets or .env file.")
    st.stop()

# Configure Gemini model
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-pro")

# Streamlit & EDA Imports
import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings
import streamlit.components.v1 as components

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import optuna, shap
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from utils.pdf_report import generate_project_report_pdf




# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AnalystIQ", layout="wide")

st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: #2D033B;
            color: #FFFFFF;
            padding: 1rem;
        }
        section[data-testid="stSidebar"] h1, h2, h3, h4, label, p, span {
            color: #FFFFFF !important;
        }
        [data-baseweb="radio"] label {
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 220px;
        }
        .analystiq-header {
            border: 3px solid #810CA8;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            background-color: #2D033B;
            margin-bottom: 10px;
        }
        .analystiq-header h1 {
            color: #FFFFFF;
        }
        .stDownloadButton > button {
            background-color: #C147E9;
            color: white;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)



st.markdown("""
    <div class='analystiq-header'>
        <h1 style='color: white;'> AnalystIQ: Automated EDA & ML Assistant</h1>
    </div>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🔍 AnalystIQ Menu")

# Sidebar navigation options
pages = {
    "📁 Upload Data": "Upload",
    "🧹 Auto Clean": "Auto Clean",
    "📊 EDA Report": "EDA Report",
    "🧠 GPT Summary": "GPT Summary",
    "💻 AutoML": "AutoML",
    "📈 Model Evaluation": "Model Evaluation"
}
selection = st.sidebar.radio("Go to Section:", list(pages.keys()))
selection = pages[selection]

# ---------------- SIDEBAR DOWNLOAD REPORT BUTTON ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Full Project Report")

# ✅ Only enable download if all steps done
data_ready = "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None
gpt_ready = "gpt_summary" in st.session_state and st.session_state.gpt_summary not in [None, ""]
model_ready = "model_trained" in st.session_state and st.session_state.model_trained

if data_ready and gpt_ready and model_ready:
    with st.sidebar:
        st.markdown(
            """
            <style>
                div[data-testid="stDownloadButton"] button {
                    width: 90% !important;
                    font-size: 13px !important;
                    padding: 4px 0px;
                    border-radius: 6px;
                    background-color: #198754;
                    color: white;
                    font-weight: bold;
                }
                div[data-testid="stDownloadButton"] button:hover {
                    background-color: #146c43;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # ✅ Click to generate report first
        if st.button("📝 Generate Report"):
            pdf_path = "project_report.pdf"
            generate_project_report_pdf(
                st.session_state.cleaned_df,
                st.session_state.gpt_summary,
                st.session_state.best_model,
                pdf_path
            )
            st.session_state.report_ready = True  # mark report as ready

        # ✅ Show download button if report is generated
        if "report_ready" in st.session_state and st.session_state.report_ready:
            with open("project_report.pdf", "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file,
                    file_name="AnalystIQ_Report.pdf",
                    mime="application/pdf"
                )
else:
    with st.sidebar:
        st.info("ℹ️ Complete all steps (Clean Data → GPT Summary → Train Model) to download report.")


# ---------------- CLEANING FUNCTION ----------------
def clean_and_impute_df(df):
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
        elif df[col].isnull().mean() > 0.9:
            cols_to_drop.append(col)
        elif df[col].nunique() > 1000:
            cols_to_drop.append(col)
    df_cleaned = df.drop(columns=cols_to_drop)

    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            if df_cleaned[col].dtype in ['float64', 'int64']:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            else:
                df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0], inplace=True)
    return df_cleaned

# ---------------- GPT FUNCTIONS ----------------

# ✅ Define Gemini Pro model globally (only once, renamed to gpt_model)
try:
    gpt_model = genai.GenerativeModel("models/gemini-2.5-pro")
except Exception as e:
    st.warning(f"⚠️ Gemini Model Initialization Failed: {e}")
    gpt_model = None

def generate_summary_from_df(df):
    """
    Generate a professional EDA summary using Gemini Pro based on the dataset.
    """
    if gpt_model is None:
        return "⚠️ GPT Summary Failed: Model not initialized"

    rows, cols = df.shape
    missing = df.isnull().mean().sort_values(ascending=False).head(5).to_dict()
    types = df.dtypes.value_counts().to_dict()

    prompt = f"""
You are an experienced data analyst preparing a final report section based on Exploratory Data Analysis (EDA). 
Write the summary in **clean, professional markdown bullet points** for executives and stakeholders.

### Dataset Metadata
- Rows: {rows}
- Columns: {cols}
- Data Types Distribution: {types}
- Top Missing Columns: {missing}

### Instructions:
- Start with a **short, clear overview** of the dataset structure (what kind of data it might represent).
- Highlight the **distribution of data types** (numeric, categorical, etc.) and why that matters.
- Comment on **missing values**: how severe, any notable patterns, and possible reasons why some missing values may remain after cleaning.
- Suggest **sensible imputation strategies** (mean, mode, advanced techniques) where needed.
- Mention any **interesting findings** from EDA (e.g., correlations, outliers, skewed distributions, duplicates, imbalances).
- End with a forward-looking statement: what type of **ML task** (classification/regression) this dataset seems best suited for and what columns might serve well as **target vs. features**.
- Keep the tone **insightful, concise, and executive-friendly**.
"""

    try:
        response = gpt_model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}]
        )
        if response.candidates and response.candidates[0].content.parts:
            summary_text = response.candidates[0].content.parts[0].text.strip()
            return "## 📊 EDA Insights\n\n" + summary_text
        else:
            return "⚠️ GPT Summary Failed: No response returned by Gemini."
    except Exception as e:
        return f"⚠️ GPT Summary Failed: {str(e)}"

def generate_model_explainer(model_name, target, features, metrics):
    """
    Generate a model explanation using Gemini Pro focused on dataset-specific insights.
    """
    if gpt_model is None:
        return "⚠️ GPT Model Explanation Failed: Model not initialized"

    prompt = f"""
You are a machine learning analyst. A model has been trained as part of an analytical project.

### Model Info
- Model Used: {model_name}
- Target Column: {target}
- Features: {', '.join(features)}
- Evaluation Metrics: {metrics}

### Write a clear, markdown-formatted explanation covering:
- **Why this model was appropriate** for the detected problem (classification or regression).
- **What the key features suggest** about the data — any surprising or important relationships.
- **Interpretation of the evaluation metrics**: what do they say about the model’s performance?
- **Potential limitations** or caveats (e.g., small dataset, missing data impact, feature bias).
- **Practical recommendations**: how could the model be improved or deployed for decision-making?

Keep it **insightful and contextual** (don’t explain “what is logistic regression” — assume the reader knows). 
Instead, focus on **what this model tells us about THIS dataset**.
"""

    try:
        response = gpt_model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}]
        )
        if response.candidates and response.candidates[0].content.parts:
            summary_text = response.candidates[0].content.parts[0].text.strip()
            return "## 🤖 Model Insights\n\n" + summary_text
        else:
            return "⚠️ GPT Model Explanation Failed: No response returned by Gemini."
    except Exception as e:
        return f"⚠️ GPT Model Explanation Failed: {str(e)}"


# ---------------- OPTUNA FUNCTION ----------------
def optuna_objective(trial, X, y, problem_type, model_choice):
    if problem_type == "Classification":
        if model_choice == "Random Forest":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        elif model_choice == "SVM (Support Vector Machine)":
            C = trial.suggest_loguniform("C", 1e-3, 1e3)
            gamma = trial.suggest_loguniform("gamma", 1e-4, 1e1)
            model = SVC(C=C, gamma=gamma, probability=True)
        else:
            return 0.0  # Skip unsupported models for Optuna

        score = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        return score.mean()

    else:
        if model_choice == "Random Forest":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        elif model_choice == "SVR (Support Vector Regressor)":
            C = trial.suggest_loguniform("C", 1e-3, 1e3)
            gamma = trial.suggest_loguniform("gamma", 1e-4, 1e1)
            model = SVR(C=C, gamma=gamma)
        else:
            return 0.0  # Skip unsupported models for Optuna

        score = cross_val_score(model, X, y, cv=3, scoring="r2")
        return score.mean()


# ---------------- MAIN ----------------
custom_config = Settings()
custom_config.interactions.continuous = False
custom_config.html.minify_html = True

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.cleaned_df = None
    st.session_state.gpt_summary = None


# ----------- UPLOAD ----------
if selection == "Upload":
    st.header("📁 Upload CSV")
    uploaded_file = st.file_uploader("Choose your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.reset_index(drop=True, inplace=True)
        st.session_state.df = df
        st.write(df.head())

# ----------- AUTO CLEAN ----------
if selection == "Auto Clean":
    st.header("🧹 Auto Clean Columns")
    if st.session_state.df is not None:
        df_cleaned = clean_and_impute_df(st.session_state.df)
        st.session_state.cleaned_df = df_cleaned
        st.success("✅ Cleaned columns removed and missing filled.")
        st.dataframe(df_cleaned.head())
        csv_cleaned = df_cleaned.to_csv(index=False)
        st.download_button("📥 Download Cleaned CSV", data=csv_cleaned, file_name="cleaned_data.csv", mime="text/csv")
    else:
        st.warning("Upload data first.")

# ----------- EDA REPORT ----------
if selection == "EDA Report":
    st.header("📊 EDA Report")

    # ✅ Initialize EDA plots store if not exists
    if "eda_plots" not in st.session_state:
        st.session_state.eda_plots = []

    if st.session_state.cleaned_df is not None:
        with st.spinner("Generating report..."):
            # Full HTML report (like before)
            profile = ProfileReport(st.session_state.cleaned_df, explorative=True, config=custom_config)
            report_html = profile.to_html()

        # Display in Streamlit
        components.html(report_html, height=1000, scrolling=True)

        # ✅ Correlation heatmap for PDF
        st.subheader("📈 Correlation Heatmap")
        corr = st.session_state.cleaned_df.corr(numeric_only=True)

        if not corr.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            # ✅ Save fig object to session (NOT file path)
            st.session_state.eda_plots.append(fig)

        # ✅ Allow HTML report download
        st.download_button("📥 Download EDA (HTML)", data=report_html,
                           file_name="EDA_Report.html", mime="text/html")
    else:
        st.warning("⚠️ Please clean the data first.")

# ----------- GPT SUMMARY ----------
if selection == "GPT Summary":
    st.header("🧠 GPT Data Summary")
    if st.session_state.cleaned_df is not None:
        if st.button("Generate GPT Summary"):
            with st.spinner("Contacting GPT..."):
                summary = generate_summary_from_df(st.session_state.cleaned_df)
                st.session_state.gpt_summary = summary
        if st.session_state.gpt_summary:
            st.markdown(st.session_state.gpt_summary)
            st.download_button("📥 Download GPT Summary (.md)", st.session_state.gpt_summary, file_name="GPT_Summary.md")
            st.download_button("📥 Download GPT Summary (.txt)", st.session_state.gpt_summary, file_name="GPT_Summary.txt")
    else:
        st.warning("Clean data first.")

# ----------- AUTOML ----------
if selection == "AutoML":
    st.header("💻 AutoML: Train Models")

    # ✅ Initialize ML plots store if not exists
    if "ml_plots" not in st.session_state:
        st.session_state.ml_plots = []

    if st.session_state.cleaned_df is not None:
        df_original = st.session_state.cleaned_df.copy()

        # Auto-detect target (last column if binary/categorical)
        inferred_target = df_original.select_dtypes(include=['object', 'bool', 'category']).columns[-1] \
            if df_original.select_dtypes(include=['object', 'bool', 'category']).shape[1] > 0 \
            else df_original.columns[-1]

        default_target = st.session_state.get("selected_target", inferred_target)

        # Drop high-null columns from features
        null_threshold = 0.5
        cleaned_features = df_original.drop(columns=[default_target])
        cleaned_features = cleaned_features.loc[:, cleaned_features.isnull().mean() < null_threshold]
        default_features = cleaned_features.columns.tolist()

        st.markdown(f"**Detected Target:** `{default_target}`")
        st.markdown(f"**Selected Features:** `{', '.join(default_features)}`")

        # [🔁] Optional Manual Override
        with st.expander("🔁 Change Target/Features Manually"):
            manual_target = st.selectbox("🎯 Select Target Column", df_original.columns, index=df_original.columns.get_loc(default_target))
            manual_features = st.multiselect("📊 Select Feature Columns", df_original.columns.drop(manual_target), default=default_features)

            if manual_target and manual_features:
                 target_col = manual_target
                 feature_cols = manual_features
            else:
                 target_col = default_target
                 feature_cols = default_features
           
        # Encode data
        df_encoded = pd.get_dummies(df_original[[*feature_cols, target_col]], drop_first=True)
        all_cols = df_encoded.columns.tolist()
        encoded_target_col = [col for col in df_encoded.columns if col.startswith(target_col)][0] \
            if target_col not in df_encoded.columns else target_col

        problem_type = "Classification" if len(df_encoded[encoded_target_col].unique()) <= 10 else "Regression"
        st.markdown(f"**Detected Problem Type:** `{problem_type}`")

        # Model selection
        model_options = {
            "Classification": ["Random Forest", "Logistic Regression", "Decision Tree", "SVM (Support Vector Machine)"],
            "Regression": ["Random Forest", "Linear Regression", "Decision Tree", "SVR (Support Vector Regressor)"]
        }
        model_choice = st.selectbox("Select Model:", model_options[problem_type])

        if st.button("🚀 Train Model"):
            with st.spinner("Training Model..."):
                try:
                    # Prepare data
                    X = df_encoded.drop(columns=[encoded_target_col])
                    y = df_encoded[encoded_target_col]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # ✅ Choose model
                    if problem_type == "Classification":
                        if model_choice == "Random Forest":
                            model = RandomForestClassifier()
                        elif model_choice == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000)
                        elif model_choice == "Decision Tree":
                            model = DecisionTreeClassifier()
                        elif model_choice == "SVM (Support Vector Machine)":
                            model = SVC(probability=True)
                    else:
                        if model_choice == "Random Forest":
                            model = RandomForestRegressor()
                        elif model_choice == "Linear Regression":
                            model = LinearRegression()
                        elif model_choice == "Decision Tree":
                            model = DecisionTreeRegressor()
                        elif model_choice == "SVR (Support Vector Regressor)":
                            model = SVR()

                    # ✅ Fit and Predict
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # ✅ Mark model as trained so PDF unlocks later
                    st.session_state.model_trained = True
                    st.session_state.best_model = model

                    st.success("✅ Model Trained!")
                    

                    # 📊 Evaluation Metrics
                    st.subheader("📊 Evaluation Metrics")
                    if problem_type == "Classification":
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        metrics = f"**Accuracy**: {acc:.2f} | **Precision**: {prec:.2f} | **Recall**: {recall:.2f} | **F1 Score**: {f1:.2f}"
                        st.markdown(metrics)
                        st.session_state.ml_metrics = metrics
                        evaluation_summary = f"Accuracy: {acc:.2f}\nPrecision: {prec:.2f}\nRecall: {recall:.2f}"

                        # ✅ ROC and PR Curve
                        if hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            precision, recall, _ = precision_recall_curve(y_test, y_proba)

                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            ax[0].plot(fpr, tpr, label="ROC Curve")
                            ax[0].plot([0, 1], [0, 1], 'k--')
                            ax[0].set_title("ROC Curve")
                            ax[0].set_xlabel("FPR")
                            ax[0].set_ylabel("TPR")

                            ax[1].plot(recall, precision, label="PR Curve", color='purple')
                            ax[1].set_title("Precision-Recall Curve")
                            ax[1].set_xlabel("Recall")
                            ax[1].set_ylabel("Precision")

                            st.pyplot(fig)

                            # ✅ Save fig object to session for PDF
                            st.session_state.ml_plots.append(fig)

                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        metrics = f"**RMSE**: {rmse:.2f} | **R² Score**: {r2:.2f}"
                        st.markdown(metrics)
                        st.session_state.ml_metrics = metrics

                        # ✅ Residual Plot
                        fig, ax = plt.subplots()
                        residuals = y_test - y_pred
                        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
                        ax.axhline(0, linestyle='--', color='red')
                        ax.set_title("Residual Plot")
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Residuals")
                        st.pyplot(fig)

                        # ✅ Save fig object to session for PDF
                        st.session_state.ml_plots.append(fig)

                    # 🧠 SHAP Explainability
                    st.subheader("🧠 SHAP Explainability")
                    try:
                        explainer = shap.Explainer(model, X_train)
                        shap_values = explainer(X_test[:50])

                        fig, ax = plt.subplots(figsize=(8, 6))
                        shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(bbox_inches="tight")

                        # ✅ Save SHAP fig
                        st.session_state.ml_plots.append(fig)

                    except Exception:
                        st.info("SHAP not available for this model.")

                    # 📡 GPT Model Explanation
                    with st.spinner("📡 GPT Explanation..."):
                        gpt_explainer = generate_model_explainer(model_choice, target_col, feature_cols, metrics)
                        st.markdown("### 📘 GPT Model Summary")
                        st.markdown(gpt_explainer)

                    # 📦 Save Trained Model for download
                    st.subheader("📦 Download Trained Model")
                    with open("trained_model.pkl", "wb") as f:
                        pickle.dump(model, f)
                    with open("trained_model.pkl", "rb") as f:
                        st.download_button("📥 Download Model (.pkl)", f, file_name="trained_model.pkl")

                except Exception as e:
                    st.error(f"⚠️ AutoML Failed: {e}")
    else:
        st.warning("⚠️ Please clean the data first.")


# ----------- MODEL EVALUATION ----------
if selection == "Model Evaluation":
    st.header("📊 Model Evaluation Visuals")
    if st.session_state.cleaned_df is not None:
        df_encoded = pd.get_dummies(st.session_state.cleaned_df, drop_first=True)
        all_cols = df_encoded.columns.tolist()
        target_col = all_cols[-1]
        feature_cols = [col for col in all_cols if col != target_col]
        X = df_encoded[feature_cols]
        y = df_encoded[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Auto Detect Problem
        problem_type = "Classification" if len(df_encoded[target_col].unique()) <= 10 else "Regression"

        model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == "Classification":
            if len(y_test.unique()) == 2 and hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                st.markdown("### 🟢 ROC Curve")
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend()
                st.pyplot(fig)

            # Precision-Recall Curve
            from sklearn.metrics import precision_recall_curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            st.markdown("### 🟣 Precision-Recall Curve")
            fig, ax = plt.subplots()
            ax.plot(recall, precision, color='purple')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            st.pyplot(fig)

        else:
            st.markdown("### 🔵 Residual Plot")
            fig, ax = plt.subplots()
            sns.residplot(x=y_test, y=y_pred, lowess=True, ax=ax, color="purple")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Residuals")
            st.pyplot(fig)

       


# ✅ Show download button ONLY after AutoML is trained
if "model_trained" in st.session_state and st.session_state.model_trained:
    # Generate PDF first
    df_final = st.session_state.cleaned_df
    gpt_summary = st.session_state.get("gpt_summary", "No GPT Summary Generated.")
    best_model = st.session_state.get("best_model", "Random Forest (Default)")
    evaluation_summary = st.session_state.get("evaluation_summary", "No evaluation summary available.")

    pdf_path = "project_report.pdf"
    generate_project_report_pdf(df_final, evaluation_summary, gpt_summary, best_model, pdf_path)

    with open(pdf_path, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Full Project Report",
            data=f,
            file_name="AnalystIQ_Report.pdf",
            mime="application/pdf",
            key="sidebar_download_btn"
        )
else:
    st.sidebar.info("⚠️ Complete all steps (Upload → Clean → EDA → GPT → AutoML) to unlock the report.")





