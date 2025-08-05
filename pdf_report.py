import streamlit as st
import pdfkit
import os
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Convert Matplotlib figure to base64 string
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ✅ Main PDF generation function
def generate_project_report_pdf(df, evaluation_summary="No evaluation summary available.",
                                model_summary="No GPT summary available.",
                                best_model="N/A", output_path="project_report.pdf"):
    plot_base64_list = []
    

    # Capture EDA plots from session
    if "eda_plots" in st.session_state:
        for fig in st.session_state.eda_plots:
            if hasattr(fig, "savefig"):
                plot_base64_list.append(fig_to_base64(fig))

    # Capture ML plots from session
    if "ml_plots" in st.session_state:
        for fig in st.session_state.ml_plots:
            if hasattr(fig, "savefig"):
                plot_base64_list.append(fig_to_base64(fig))

    # Generate histogram for numeric columns
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        fig = numeric_df.hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black')
        plt.tight_layout()
        img_b64 = fig_to_base64(plt.gcf())
        plot_base64_list.append(img_b64)
        plt.close()

    # Missing value heatmap
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="YlOrRd", ax=ax)
        ax.set_title("Missing Values Heatmap")
        img_b64 = fig_to_base64(fig)
        plot_base64_list.append(img_b64)
        plt.close()

    # Correlation heatmap
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        img_b64 = fig_to_base64(fig)
        plot_base64_list.append(img_b64)
        plt.close()

    # HTML template for the report
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
            img {{ max-width: 650px; margin: 12px 0; border: 1px solid #ddd; border-radius: 6px; }}
            .metrics {{ background: #f8f9fa; padding: 10px; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <h1>📊 AnalystIQ - Project Report</h1>

        <h2>📄 Dataset Overview</h2>
        <p><strong>Rows:</strong> {df.shape[0]} | <strong>Columns:</strong> {df.shape[1]}</p>

        <h2>🧑‍🤖 GPT EDA Summary</h2>
       if gpt_summary is None:
    gpt_summary = "No GPT summary generated."


        <h2>🤖 ML Model Summary</h2>
        <p><strong>Best Model:</strong> <code>{str(best_model)}</code></p>
        <p>{model_summary.replace("\n", "<br>")}</p>
    """

    # Include metrics if available
    if "ml_metrics" in st.session_state:
        html_content += f"""
        <div class='metrics'>
            <h3>📊 Model Performance:</h3>
            <p>{st.session_state.ml_metrics.replace('\n', '<br>')}</p>
        </div>
        """

    # Add images
    html_content += "<h2>📊 Visual Analysis</h2>"
    for img_b64 in plot_base64_list:
        html_content += f'<img src="data:image/png;base64,{img_b64}"><br>'

    html_content += "</body></html>"

    # Save HTML to file
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    # Convert to PDF
    try:
        config = pdfkit.configuration(wkhtmltopdf=r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")
        pdfkit.from_file("report.html", output_path, configuration=config)
        st.success("✅ PDF Report generated successfully!")
    except Exception as e:
        st.error(f"❌ PDF generation failed: {e}")
