import os, io, re, json, calendar, difflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from PIL import Image
import gradio as gr
from sqlalchemy import create_engine
from openai import OpenAI
import mlflow
import mlflow.pyfunc
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import shap

# ✅ OpenAI API Setup
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
MODEL = "mistralai/mistral-7b-instruct:free"
SYSTEM_PROMPT = """
You are an AI agent for analyzing production CSV data from PostgreSQL.
Your response MUST be a JSON object with these keys:
{
  "task": "forecast" | "compare" | "zero_count",
  "table": string,
  "date_col": string,
  "target_col": string,
  "compare_cols": optional [string, string],
  "filters": optional { column: value },
  "period_days": optional int (only for forecast)
}
If the user mentions a specific month (e.g., January), add a filter like:
"filters": {"month": "January"}
Return only valid JSON.
"""

# ✅ MLflow Experiment Init
mlflow.set_experiment("Production_Insights_AI_Agent")

def get_engine():
    return create_engine("postgresql://postgres:root@localhost:5432/prod_insights")

engine = get_engine()

def smart_match_column(user_key, df_columns):
    user_key_norm = user_key.strip().lower()
    df_columns_norm = [col.strip().lower() for col in df_columns]
    close_matches = difflib.get_close_matches(user_key_norm, df_columns_norm, n=1, cutoff=0.75)
    if close_matches:
        return df_columns[df_columns_norm.index(close_matches[0])]
    time_keywords = ['time', 'slot', 'range']
    if any(kw in user_key_norm for kw in time_keywords):
        for col in df_columns:
            if any(kw in col.lower() for kw in time_keywords):
                return col
    return user_key

def try_filtered_df(filters, df):
    temp = df.copy()
    for user_key, val in filters.items():
        matched_col = smart_match_column(user_key, temp.columns)
        if matched_col in temp.columns:
            temp[matched_col] = temp[matched_col].astype(str).str.strip().str.lower()
            val_norm = str(val).strip().lower()
            col_values = temp[matched_col].unique()
            col_values_norm = [v.strip().lower() for v in col_values]
            matched_val = next((v for v in col_values if v.strip().lower() == val_norm), None)
            if not matched_val:
                fuzzy = difflib.get_close_matches(val_norm, col_values_norm, n=1, cutoff=0.7)
                if fuzzy:
                    matched_val = col_values[col_values_norm.index(fuzzy[0])]
            if matched_val:
                temp = temp[temp[matched_col] == matched_val]
            else:
                print(f"❌ No match for filter {user_key} = {val}, skipping this filter.")
    return temp

def upload_and_insert(files):
    for file in files:
        df = pd.read_csv(file.name)
        table_name = os.path.splitext(os.path.basename(file.name))[0].lower().replace("-", "_").replace(" ", "_")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
    return "✅ CSVs uploaded & inserted into PostgreSQL."

def get_schema():
    with engine.connect() as conn:
        tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'", conn)
        schema = {}
        for table in tables.table_name:
            cols = pd.read_sql(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'", conn)
            schema[table] = cols['column_name'].tolist()
        return schema

def ask_agent_schema_and_task(query, schema):
    prompt = SYSTEM_PROMPT + f"\n\nSchema:\n{json.dumps(schema)}\n\nUser Query: {query}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}]
    )
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        content_fixed = re.sub(r"```json|```", "", content)
        return json.loads(content_fixed)

def auto_detect_date_column(df):
    for col in df.columns:
        sample = df[col].astype(str).head(100)
        parsed_ddmmyyyy = pd.to_datetime(sample, format='%d-%m-%Y', errors='coerce')
        parsed_yyyymmdd = pd.to_datetime(sample, format='%Y-%m-%d', errors='coerce')
        if parsed_ddmmyyyy.notna().sum() >= len(sample) * 0.5:
            df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
            return col
        elif parsed_yyyymmdd.notna().sum() >= len(sample) * 0.5:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
            return col
    return None

def extract_month_year_from_query(query):
    query_lower = query.lower()
    month_map = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    detected_month = next((month for month in month_map if month in query_lower), None)
    match = re.search(r"\b(20[0-9]{2})\b", query_lower)
    detected_year = int(match.group(1)) if match else None
    return detected_month, detected_year

def apply_time_filters(df, date_col, filters):
    month_map = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    if 'month' in filters:
        month_val = str(filters['month']).strip().lower()
        try:
            month_num = int(month_val)
        except ValueError:
            month_num = month_map.get(month_val)
        if month_num:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df[df[date_col].dt.month == month_num]
    if 'year' in filters:
        try:
            year_val = int(filters['year'])
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df[df[date_col].dt.year == year_val]
        except ValueError:
            pass
    return df

def dynamic_analysis(query):
    try:
        with mlflow.start_run(run_name="Dynamic_Prod_Analysis"):
            schema = get_schema()
            plan = ask_agent_schema_and_task(query, schema)
            table_name = plan["table"].lower().replace("-", "_")
            with engine.connect() as conn:
                df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)

            df.columns = df.columns.str.strip().str.lower()
            plan['target_col'] = plan['target_col'].strip().lower()
            if 'compare_cols' in plan:
                plan['compare_cols'] = [col.strip().lower() for col in plan['compare_cols']]
            if 'filters' in plan:
                plan['filters'] = {k.strip().lower(): str(v).strip() for k, v in plan['filters'].items()}
            else:
                plan['filters'] = {}

            month, year = extract_month_year_from_query(query)
            if month and 'month' not in plan['filters']:
                plan['filters']['month'] = month
            if year and 'year' not in plan['filters']:
                plan['filters']['year'] = str(year)

            detected_date_col = auto_detect_date_column(df)
            if not detected_date_col:
                return "❌ No valid date format detected in the table.", None
            plan['date_col'] = detected_date_col

            df = df.dropna(subset=[plan['date_col']])
            df = apply_time_filters(df, plan['date_col'], plan['filters'])

            fig = None
            buf = io.BytesIO()

            # ✅ Forecast Task
            if plan['task'] == 'forecast':
                df = df.dropna(subset=[plan['target_col']])
                used_filters = plan['filters'].copy()
                filtered_df = try_filtered_df(used_filters, df)
                while filtered_df.shape[0] < 2 and used_filters:
                    removed = used_filters.popitem()
                    filtered_df = try_filtered_df(used_filters, df)
                if filtered_df.shape[0] < 2:
                    return "❌ Not enough data for forecasting.", None

                filtered_df = filtered_df.groupby(filtered_df[plan['date_col']].dt.date)[plan['target_col']].sum().reset_index()
                filtered_df.columns = ['ds', 'y']

                model = Prophet()
                model.fit(filtered_df)
                future = model.make_future_dataframe(periods=plan.get("period_days", 30))
                forecast = model.predict(future)

                if 'yhat' in forecast.columns:
                    y_true = filtered_df['y'].values
                    y_pred = forecast['yhat'].iloc[:len(y_true)].values
                    mlflow.log_metric("MAE", mean_absolute_error(y_true, y_pred))
                    mlflow.log_metric("MAPE", mean_absolute_percentage_error(y_true, y_pred))

                fig = model.plot(forecast)
                mlflow.prophet.log_model(model, "forecast_model")

            # ✅ Compare Task
            elif plan['task'] == 'compare':
                if 'compare_cols' not in plan or len(plan['compare_cols']) < 2:
                    detected = [c for c in df.columns if "plan" in c.lower() or "actual" in c.lower()]
                    if len(detected) >= 2:
                        plan['compare_cols'] = detected[:2]
                    else:
                        return "❌ Couldn't detect comparison columns.", None
                df = df.groupby(plan['date_col'])[plan['compare_cols']].sum().reset_index()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(df[plan['date_col']], df[plan['compare_cols'][0]], width=0.4, label=plan['compare_cols'][0], align='edge')
                ax.bar(df[plan['date_col']] - pd.Timedelta(days=0.4), df[plan['compare_cols'][1]], width=0.4, label=plan['compare_cols'][1], align='edge')
                ax.set_title("Planned vs Actual")
                ax.legend()

            # ✅ Zero Count Task
            elif plan['task'] == 'zero_count':
                count = df[df[plan['target_col']] == 0].shape[0]
                mlflow.log_metric("Zero_Count", count)
                return f"🚨 Found {count} zero-production entries.", None

            else:
                return "❌ Unsupported task.", None

            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            mlflow.log_image(buf, "result_plot.png")
            plt.close(fig)
            buf.seek(0)
            return "✅ Task completed and logged to MLflow 🎉", Image.open(buf)

    except Exception as e:
        mlflow.log_param("error", str(e))
        return f"❌ Error: {str(e)}", None

def query_ui(query):
    result, image = dynamic_analysis(query)
    return result, image

def ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 AI Production Agent (Now MLflow-Powered 🚀)")
        with gr.Tab("📄 Upload CSVs"):
            csv_upload = gr.File(file_types=['.csv'], file_count="multiple")
            upload_btn = gr.Button("📥 Upload to PostgreSQL")
            upload_output = gr.Textbox()
            upload_btn.click(fn=upload_and_insert, inputs=[csv_upload], outputs=[upload_output])
        with gr.Tab("🔍 Ask AI"):
            query_box = gr.Textbox(label="Enter your production-related query")
            query_btn = gr.Button("🚀 Analyze")
            output_text = gr.Textbox(label="📋 Insight")
            output_img = gr.Image(label="📈 Chart")
            query_btn.click(fn=query_ui, inputs=[query_box], outputs=[output_text, output_img])
    return demo

if __name__ == "__main__":
    ui().launch(share=True)
