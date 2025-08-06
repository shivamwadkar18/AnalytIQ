# 📊 AnalystIQ: Automated EDA, GPT Insights & ML Modeling Platform

**AnalystIQ** is an advanced all-in-one Streamlit application for Data Analysts and Data Scientists. It allows you to upload a dataset, automatically clean it, explore it using profiling reports, summarize it using GPT models, train ML models with AutoML (including hyperparameter tuning via Optuna), and generate a professional PDF report – all with a polished UI.

---

## 🚀 Features

- **📁 Upload CSV Data**
- **🧹 Auto Data Cleaning**
  - Removes constant, high-cardinality, and high-null columns
- **📊 Interactive EDA Reports**
  - Powered by `ydata-profiling`
- **🧠 GPT-Based Dataset Summary**
  - Uses OpenRouter's Mistral-7B for professional markdown summaries
- **🤖 AutoML Module**
  - Automatically detects target column
  - Auto-selects appropriate ML model (classification/regression)
  - Supports:
    - Logistic Regression
    - Decision Tree
    - Random Forest (with Optuna)
    - SVM (with Optuna)
- **📈 Model Evaluation Metrics**
  - Classification: Accuracy, Precision, Recall, F1
  - Regression: R², MAE, MSE, RMSE
- **📄 Downloadable PDF Report**
  - Includes GPT summary, model details, evaluation metrics

---

## 🖼️ UI Highlights

- Professional dark theme with dark green + black palette
- Slim sidebar with section navigation
- Responsive layout with styled components

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/AnalystIQ.git
cd AnalystIQ
pip install -r requirements.txt
