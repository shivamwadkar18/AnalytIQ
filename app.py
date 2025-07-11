from dotenv import load_dotenv
import os

# ✅ Load API Key from custom .env file
load_dotenv(dotenv_path="gpt.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

# 🔐 Debug: Print first few chars of the key
print("🔐 Loaded API Key (start):", openai_api_key[:5] if openai_api_key else "❌ NOT LOADED")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Check your gpt.env file.")

# ✅ Initialize OpenAI client for OpenRouter
from openai import OpenAI
client = OpenAI(
    api_key=openai_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# 🖥️ Streamlit & EDA Imports
import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings
import streamlit.components.v1 as components

# 🤖 ML Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(page_title="Auto EDA Report Generator", layout="wide")
st.title("📊 Auto EDA Report Generator")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV file", type=["csv"])
apply_cleaning = st.sidebar.checkbox("🧹 Auto-clean columns", value=True)
st.sidebar.markdown(
    "**Auto-clean removes:**\n"
    "- Constant columns\n"
    "- Columns with >90% missing values\n"
    "- High-cardinality columns (>1000 unique)"
)

# ---------- CLEANING FUNCTION ----------
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

    # Impute missing values
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            if df_cleaned[col].dtype in ['float64', 'int64']:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            else:
                df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0], inplace=True)

    return df_cleaned

# ---------- GPT SUMMARY FUNCTION ----------
def generate_summary_from_df(df):
    rows, cols = df.shape
    missing = df.isnull().mean().sort_values(ascending=False).head(5).to_dict()
    types = df.dtypes.value_counts().to_dict()

    prompt = f"""
You are a professional data analyst.

Please analyze this dataset and write an executive-style summary. Structure your output into clear bullet points. Use markdown-style formatting for boldness and clarity.

Dataset Metadata:
- Total Rows: {rows}
- Total Columns: {cols}
- Column Types: {types}
- Top 5 Columns with Missing Data: {missing}

Instructions(answer all in short summary format):
- Start in short intro about the dataset structure.
- Explain the column types and their distribution.
- Discuss the top missing-value columns and likely reasons for missing data (e.g., user input errors, legacy systems, optional fields).
- Justify why some missing values might still exist after cleaning (hint: cleaning only removes columns with >90% missing).
- Suggest suitable imputation or handling strategies.
- Highlight anything interesting: e.g., imbalance in types, presence of high cardinality, or duplicate entries (if inferable).
- Make the tone professional and useful for data storytelling.
- Also suggest the what ml model should be used ,what to select target column & feature column , what should be target to find using ml(based on available columns dont suggest the columns or target which is not available).
"""

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT Summary Failed: {str(e)}"

# ---------- YDATA PROFILE CONFIG ----------
custom_config = Settings()
custom_config.interactions.continuous = False
custom_config.html.minify_html = True

# ---------- MAIN APP ----------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.reset_index(drop=True, inplace=True)

        st.subheader("🙀 Data Preview")
        st.dataframe(df.head())

        st.subheader("🎯 Column Selection")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select columns for EDA", all_columns, default=all_columns)

        if selected_columns:
            df_selected = df[selected_columns]

            df_final = clean_and_impute_df(df_selected) if apply_cleaning else df_selected.copy()

            if apply_cleaning:
                st.success("🧽 Auto-cleaning applied: constant, mostly-null, and high-cardinality columns removed.")
                csv_cleaned = df_final.to_csv(index=False)
                st.download_button("📥 Download Cleaned CSV", data=csv_cleaned, file_name="cleaned_data.csv", mime="text/csv")

            with st.spinner("🔎 Generating EDA report..."):
                profile = ProfileReport(df_final, explorative=True, config=custom_config)
                report_html = profile.to_html()

            st.subheader("📊 Exploratory Data Analysis Report")
            components.html(report_html, height=1000, scrolling=True)
            st.download_button("📥 Download EDA Report (HTML)", data=report_html, file_name="EDA_Report.html", mime="text/html")

            st.subheader("🧠 GPT Data Summary")
            if st.button("Generate GPT Summary"):
                with st.spinner("Contacting GPT..."):
                    summary = generate_summary_from_df(df_final)
                    st.markdown("### 🧠 GPT-Generated Data Summary")
                    st.markdown(summary)

            # ---------- 🤖 AutoML Section ----------
            st.subheader("🤖 AutoML: Train a Simple Model")

            original_cols = df_final.columns.tolist()
            if len(original_cols) < 2:
                st.warning("Need at least 2 usable columns for AutoML.")
            else:
                target_col = st.selectbox("🎯 Select Target Column (Label)", original_cols)
                feature_cols = st.multiselect(
                    "📊 Select Feature Columns (Inputs)",
                    [col for col in original_cols if col != target_col],
                    default=[col for col in original_cols if col != target_col]
                )

                model_choice = st.radio("🧠 Choose ML Model", ["Logistic Regression", "Decision Tree"])

                if st.button("🚀 Train Model"):
                    try:
                        df_ml = df_final[feature_cols + [target_col]].copy()
                        df_encoded = pd.get_dummies(df_ml, drop_first=True)

                        X = df_encoded.drop(columns=[col for col in df_encoded.columns if col.startswith(target_col + "_")])
                        y = df_encoded[[col for col in df_encoded.columns if col.startswith(target_col + "_")]]
                        if y.shape[1] == 1:
                            y = y.iloc[:, 0]
                        else:
                            st.warning("⚠️ Multiclass target detected. Some models may not support this well.")

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = LogisticRegression() if model_choice == "Logistic Regression" else DecisionTreeClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        st.success("✅ Model trained successfully!")
                        st.markdown(f"**🔢 Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
                        st.markdown("**📋 Classification Report:**")
                        st.text(classification_report(y_test, y_pred))

                        st.markdown("📊 **Confusion Matrix:**")
                        fig, ax = plt.subplots()
                        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"⚠️ Model training failed: {e}")

        else:
            st.warning("Please select at least one column to generate the report.")

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
else:
    st.info("⬅️ Use the sidebar to upload a CSV file and configure settings.")
