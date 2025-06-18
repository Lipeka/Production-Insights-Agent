# ============================ 📦 IMPORTS ============================
import os, io, re, base64, tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from PIL import Image
import gradio as gr
import mlflow
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from openai import OpenAI
from db_config import get_engine

# ============================ 🔐 API CONFIG ============================
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
MODEL = "mistralai/mistral-7b-instruct:free"

# ============================ 🔗 DB CONFIG ============================
engine = get_engine()

# ============================ 📜 SYSTEM PROMPT ============================
SYSTEM_PROMPT = """
You are a production data analyst for a manufacturing company.
You can answer any query about:
- Total/average/max production
- Forecasting using past trends
- Comparing planned vs actual
- Zero production entries
- Time-based or line-specific analysis

Data columns: Line, Job, Time_slot, Shift_Date, Shift, Planned, Actual, Losstime, Month, MonthNumber, Date.

Always use a concise, analytical tone. Add insights if data has clear trends or anomalies.

User Query: {{USER_QUERY}}
"""

# ============================ 📊 FETCH DATA ============================
def fetch_data(user_query=None):
    df = pd.read_sql("SELECT * FROM production_data", engine)
    df["Shift_Date"] = pd.to_datetime(df["Shift_Date"])
    
    if user_query:
        uq = user_query.lower()

        # Line filter: A-1, A-10, etc.
        line_matches = re.findall(r'a-?\d{1,2}', uq)
        if line_matches:
            normalized_lines = [line.upper() if "-" in line else f"A-{line[-1]}" for line in line_matches]
            df = df[df["Line"].isin(normalized_lines)]

        # Job filter
        job_matches = re.findall(r'job\s?\d+', uq)
        if job_matches:
            job_ids = [re.search(r'\d+', job).group() for job in job_matches]
            df = df[df["Job"].astype(str).isin(job_ids)]

        # Shift filter
        shift_matches = re.findall(r'\bshift\s?[A-Z]?\d+', uq)
        if shift_matches:
            shifts = [re.sub(r'\s+', '', s).split("shift")[-1] for s in shift_matches]
            df = df[df["Shift"].astype(str).isin(shifts)]

        # Time_slot filter
        time_slot_matches = re.findall(r'\btime\s?slot\s?\d+\b', uq)
        if time_slot_matches:
            time_slots = [re.search(r'\d+', slot).group() for slot in time_slot_matches]
            df = df[df["Time_slot"].astype(str).isin(time_slots)]

        # Date-specific filter
        date_matches = re.findall(r'\d{4}-\d{2}-\d{2}', uq)
        if date_matches:
            dates = pd.to_datetime(date_matches)
            df = df[df["Shift_Date"].isin(dates)]

        # Month filtering (e.g., "June" or "month 6")
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        mentioned_months = [num for name, num in month_map.items() if name in uq]
        month_numbers = re.findall(r'month\s?(\d+)', uq)
        mentioned_months += list(map(int, month_numbers))
        if mentioned_months:
            df = df[df["Shift_Date"].dt.month.isin(mentioned_months)]

        # Year filter
        year_match = re.search(r'(20\d{2})', uq)
        if year_match:
            df = df[df["Shift_Date"].dt.year == int(year_match.group(1))]

        # Zero production filter
        if "zero" in uq and "production" in uq:
            df = df[df["Actual"] == 0]

    return df.sort_values("Shift_Date")


# ============================ 🔮 FORECASTING ============================
def extract_forecast_period(query):
    query = query.lower()
    days = 30
    if match := re.search(r"(\d+)\s*day", query):
        days = int(match.group(1))
    elif match := re.search(r"(\d+)\s*week", query):
        days = int(match.group(1)) * 7
    elif match := re.search(r"(\d+)\s*month", query):
        days = int(match.group(1)) * 30
    return days

def generate_forecast(df, periods=30):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Forecasting_Production")

    with mlflow.start_run():
        daily = df.groupby("Shift_Date")["Actual"].sum().reset_index()
        daily.columns = ["ds", "y"]
        model = Prophet()
        model.fit(daily)

        mlflow.log_param("periods", periods)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        mape = ((abs(forecast["yhat"][:len(daily)] - daily["y"]) / daily["y"]).mean()) * 100
        mlflow.log_metric("MAPE", mape)

        fig = model.plot(forecast)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

    return forecast, buf

# ============================ 📈 VISUALIZATION ============================
def plot_planned_vs_actual(df):
    if df.empty:
        return None

    daily = df.groupby("Shift_Date")[["Planned", "Actual"]].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4
    ax.bar(daily["Shift_Date"] - pd.Timedelta(days=0.2), daily["Planned"], width=width, label="Planned", color="skyblue")
    ax.bar(daily["Shift_Date"] + pd.Timedelta(days=0.2), daily["Actual"], width=width, label="Actual", color="orange")

    ax.set_title("📊 Planned vs Actual Production")
    ax.set_ylabel("Units")
    ax.set_xlabel("Date")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)

# ============================ 🧠 SHAP EXPLAINABILITY ============================
def compare_shap_lines(lines_list):
    df = pd.read_sql("SELECT * FROM production_data", engine)
    df["Shift_Date"] = pd.to_datetime(df["Shift_Date"])

    if "MonthNumber" not in df.columns or df["MonthNumber"].isnull().all():
        df["MonthNumber"] = df["Shift_Date"].dt.month

    features = ["Planned", "Losstime", "MonthNumber"]
    n_lines = len(lines_list)

    fig, axes = plt.subplots(1, n_lines, figsize=(6 * n_lines, 5))
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for i, line in enumerate(lines_list):
        line_df = df[df["Line"] == line].dropna(subset=features + ["Actual"])
        if len(line_df) < 10:
            axes[i].text(0.5, 0.5, f"❌ Not enough data for {line}", ha='center', va='center')
            axes[i].axis("off")
            continue

        X = line_df[features]
        y = line_df["Actual"]

        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(X, y)

        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        shap.plots.bar(shap_values, max_display=5, show=False, ax=axes[i])
        axes[i].set_title(f"SHAP: {line}")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return f"📊 SHAP Feature Importance Comparison: {', '.join(lines_list)}", Image.open(buf)

# ============================ 🧠 OPENROUTER RAG ============================
def ask_agent(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT.replace("{{USER_QUERY}}", prompt)}]
    )
    return response.choices[0].message.content

# ============================ 🚀 QUERY HANDLER ============================
def handle_query_gradio(query):
    df = fetch_data(query)
    if df.empty:
        return "🚫 No data found for your query.", None

    lower_q = query.lower()

    try:
        if "average" in lower_q or "avg" in lower_q:
            avg_val = df["Actual"].mean()
            return f"📊 Average actual production: **{avg_val:.2f}**", None

        elif "total" in lower_q or "sum" in lower_q:
            total_val = df["Actual"].sum()
            return f"🧾 Total actual production: **{total_val:.0f}**", None

        elif "max" in lower_q or "highest" in lower_q:
            max_val = df["Actual"].max()
            max_day = df[df["Actual"] == max_val]["Shift_Date"].iloc[0]
            return f"🏆 Max actual: **{max_val}** on {max_day.date()}", None

        elif "planned vs actual" in lower_q or "compare planned" in lower_q:
            planned = df["Planned"].sum()
            actual = df["Actual"].sum()
            diff = actual - planned
            img = plot_planned_vs_actual(df)
            insight = f"""📈 Planned: {planned:.0f}  
✅ Actual: {actual:.0f}  
🔍 Difference: {diff:+.0f}"""
            return insight.strip(), img

        elif "shap" in lower_q and "compare" in lower_q:
            found_lines = re.findall(r"a-\d+", lower_q)
            lines = list(set(line.upper() for line in found_lines))
            return compare_shap_lines(lines)

        elif "zero production" in lower_q or "0 production" in lower_q:
            zero_df = df[df["Actual"] == 0]
            return f"🚨 {len(zero_df)} time slots had zero production.", None

        elif "forecast" in lower_q:
            forecast_days = extract_forecast_period(query)
            forecast_df, img_buf = generate_forecast(df, periods=forecast_days)
            img = Image.open(img_buf)
            return f"🔮 Forecast for next {forecast_days} days.", img

        else:
            return ask_agent(query), None

    except Exception as e:
        return f"❌ Error during data processing: {str(e)}", None

# ============================ 🎨 GRADIO UI ============================
def ui_pipeline(query):
    try:
        insight, image = handle_query_gradio(query)
        return insight, image
    except Exception as e:
        return f"❌ Error: {str(e)}", None

demo = gr.Interface(
    fn=ui_pipeline,
    inputs=gr.Textbox(label="🧠 Ask your production query"),
    outputs=[gr.Textbox(label="📋 AI Insight"), gr.Image(type="pil", label="📊 Chart Output")],
    title="🏭 Production Insights AI Agent",
    description="Ask about production trends, forecasts, feature importance, and more. Data-powered insights on demand!",
    theme="default"
)

if __name__ == "__main__":
    demo.launch(share=True)
