import os, io, re, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from PIL import Image
import gradio as gr
from sqlalchemy import create_engine
from openai import OpenAI
import difflib
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
Return only valid JSON.
"""
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
    return user_key  # fallback
def try_filtered_df(filters, df):
    temp = df.copy()
    for user_key, val in filters.items():
        matched_col = smart_match_column(user_key, temp.columns)
        if matched_col in temp.columns:
            temp[matched_col] = temp[matched_col].astype(str).str.strip().str.lower()
            val_norm = str(val).strip().lower()
            time_pattern = r'\b\d{1,2}(am|pm)?\s*[-to]+\s*\d{1,2}(am|pm)\b'
            is_time_value = bool(re.search(time_pattern, val_norm))
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
                print(f"❌ No match found for filter {user_key} = {val}, skipping this filter.")
    return temp
def get_engine():
    return create_engine("postgresql://postgres:root@localhost:5432/prod_insights")
engine = get_engine()
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
            print(f"📆 Detected date format: DD-MM-YYYY in column '{col}'")
            return col
        elif parsed_yyyymmdd.notna().sum() >= len(sample) * 0.5:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
            print(f"📆 Detected date format: YYYY-MM-DD in column '{col}'")
            return col
    return None
def dynamic_analysis(query):
    try:
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
        detected_date_col = auto_detect_date_column(df)
        if not detected_date_col:
            return "❌ No valid date format detected in the table.", None
        plan['date_col'] = detected_date_col
        df = df.dropna(subset=[plan['date_col']])
        if plan['task'] == 'forecast':
            df = df.dropna(subset=[plan['target_col']])
            used_filters = plan.get('filters', {}).copy()
            filtered_df = try_filtered_df(used_filters, df)
            while filtered_df.shape[0] < 2 and used_filters:
                removed = used_filters.popitem()
                print(f"⚠️ Not enough data. Removed filter: {removed[0]} = {removed[1]}")
                filtered_df = try_filtered_df(used_filters, df)
            print(f"🔎 Filtered rows: {filtered_df.shape[0]} for filters: {used_filters}")
            if filtered_df.shape[0] < 2:
                return "❌ Not enough data to forecast, even after relaxing filters.", None
            filtered_df = filtered_df.groupby(filtered_df[plan['date_col']].dt.date)[plan['target_col']].sum().reset_index()
            filtered_df.columns = ['ds', 'y']
            model = Prophet()
            model.fit(filtered_df)
            future = model.make_future_dataframe(periods=plan.get("period_days", 30))
            forecast = model.predict(future)
            fig = model.plot(forecast)
        elif plan['task'] == 'compare':
            if 'compare_cols' not in plan or len(plan['compare_cols']) < 2:
                all_cols = df.columns.tolist()
                detected = [c for c in all_cols if "plan" in c.lower() or "actual" in c.lower()]
                if len(detected) >= 2:
                    plan['compare_cols'] = detected[:2]
                else:
                    return "❌ Couldn't detect both 'planned' and 'actual' columns automatically.", None
            df = df.groupby(plan['date_col'])[plan['compare_cols']].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df[plan['date_col']], df[plan['compare_cols'][0]], width=0.4, label=plan['compare_cols'][0], align='edge')
            ax.bar(df[plan['date_col']] - pd.Timedelta(days=0.4), df[plan['compare_cols'][1]], width=0.4, label=plan['compare_cols'][1], align='edge')
            ax.set_title("Planned vs Actual")
            ax.legend()
        elif plan['task'] == 'zero_count':
            count = df[df[plan['target_col']] == 0].shape[0]
            return f"🚨 Found {count} zero-production entries.", None
        else:
            return "❌ Unsupported task.", None
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return "✅ Task executed successfully", Image.open(buf)
    except Exception as e:
        return f"❌ Error: {str(e)}", None
def query_ui(query):
    result, image = dynamic_analysis(query)
    return result, image
def ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 AI Production Agent")
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
