import streamlit as st
import pdfkit
import os
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def fig_to_base64(fig):
    """Convert a Matplotlib figure to base64 for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_project_report_pdf(df, gpt_summary, best_model, output_path):
    """
    Generates a full AnalystIQ PDF report with:
    ✅ Dataset overview
    ✅ GPT summary
    ✅ ML summary + metrics
    ✅ EDA & ML visuals (base64 embedded)
    """

    plot_base64_list = []  # store all encoded images

    # ----------------------------------------------------
    # 1️⃣ Add existing EDA plots from Streamlit session
    # ----------------------------------------------------
    if "eda_plots" in st.session_state:
        for fig in st.session_state.eda_plots:
            if isinstance(fig, str):  # if already saved path
                with open(fig, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    plot_base64_list.append(img_b64)
            elif hasattr(fig, "savefig"):  # Matplotlib fig
                img_b64 = fig_to_base64(fig)
                plot_base64_list.append(img_b64)

    # ----------------------------------------------------
    # 2️⃣ Add existing ML plots from Streamlit session
    # ----------------------------------------------------
    if "ml_plots" in st.session_state:
        for fig in st.session_state.ml_plots:
            if isinstance(fig, str):
                with open(fig, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    plot_base64_list.append(img_b64)
            elif hasattr(fig, "savefig"):
                img_b64 = fig_to_base64(fig)
                plot_base64_list.append(img_b64)

    # ----------------------------------------------------
    # 3️⃣ Generate additional EDA visuals on the fly
    # ----------------------------------------------------

    # 🔹 Histograms for numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        n_cols = 2
        n_rows = (len(numeric_df.columns) + 1) // n_cols
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 4 * n_rows))

        # make sure axes is always a flat numpy array
        axes = np.array(axes).reshape(-1)

        numeric_df.hist(ax=axes, bins=20, color='skyblue', edgecolor='black')
        plt.suptitle("Feature Distributions", fontsize=16)

        # remove extra empty plots if any
        for ax in axes[len(numeric_df.columns):]:
            ax.remove()

        img_b64 = fig_to_base64(fig)
        plot_base64_list.append(img_b64)

    # 🔹 Missing Values Heatmap
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="YlOrRd", ax=ax)
        ax.set_title("Missing Values Heatmap")
        img_b64 = fig_to_base64(fig)
        plot_base64_list.append(img_b64)

    # 🔹 Correlation Heatmap
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        img_b64 = fig_to_base64(fig)
        plot_base64_list.append(img_b64)

    # ----------------------------------------------------
    # 4️⃣ Build HTML report
    # ----------------------------------------------------
    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AnalystIQ Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; }}
            h1 {{ color: #198754; }}
            h2 {{ color: #222; border-bottom: 2px solid #198754; padding-bottom: 5px; }}
            h3 {{ color: #444; }}
            p {{ line-height: 1.6; font-size: 14px; }}
            .section {{ margin-bottom: 30px; }}
            img {{ max-width: 650px; margin: 12px 0; border: 1px solid #ddd; border-radius: 6px; }}
            .metrics {{ background: #f8f9fa; padding: 10px; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <h1>📊 AnalystIQ - Comprehensive Report</h1>

        <div class="section">
            <h2>📄 Dataset Overview</h2>
            <p><strong>Rows:</strong> {df.shape[0]} | <strong>Columns:</strong> {df.shape[1]}</p>
        </div>

        <div class="section">
            <h2>🧠 GPT Summary</h2>
            <p>{gpt_summary.replace("\n", "<br>")}</p>
        </div>

        <div class="section">
            <h2>🤖 Machine Learning Summary</h2>
            <h3>Best Model:</h3>
            <p><code>{str(best_model)}</code></p>
    """

    # ✅ If ML metrics are available, embed them nicely
    if "ml_metrics" in st.session_state:
        html_content += f"<div class='metrics'><h3>📊 Model Performance:</h3><p>{st.session_state.ml_metrics}</p></div>"

    html_content += """
        </div>

        <div class="section">
            <h2>📊 Visual Analysis</h2>
            <p>Below are important visuals from EDA and ML phases:</p>
    """

    # ✅ Embed all plots inline using base64
    for img_b64 in plot_base64_list:
        html_content += f'<img src="data:image/png;base64,{img_b64}">'

    html_content += """
        </div>
    </body>
    </html>
    """

    # ----------------------------------------------------
    # 5️⃣ Write temp HTML & convert to PDF
    # ----------------------------------------------------
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    config = pdfkit.configuration(wkhtmltopdf=r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")
    pdfkit.from_file("report.html", output_path, configuration=config)
