import os
import io
import re
import json
import base64
import contextlib
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sqlalchemy import create_engine, inspect
from langgraph.graph import StateGraph, END
from openai import OpenAI
from prophet import Prophet
def load_csv_to_postgres(df, table_name, engine):
    try:
        df.to_sql(table_name, engine, index=False, if_exists='replace')
        st.success(f"✅ Uploaded '{table_name}' to PostgreSQL")
    except Exception as e:
        st.error(f"❌ Failed to upload '{table_name}': {e}")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)
MODEL = "qwen/qwen-2.5-72b-instruct:free"
def load_all_tables():
    engine = create_engine("postgresql://postgres:root@localhost:5432/ai_agent_db")
    schema_info = {}
    with engine.connect() as conn:
        inspector = inspect(engine)
        for table in inspector.get_table_names():
            col_query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}';
            """
            col_df = pd.read_sql(col_query, conn)
            try:
                sample_df = pd.read_sql(f'SELECT * FROM "{table}" LIMIT 3;', conn)
                samples = sample_df.to_dict(orient='records')
            except:
                samples = []
            schema_info[table] = {
                "columns": [(row["column_name"], row["data_type"]) for _, row in col_df.iterrows()],
                "sample_rows": samples
            }
    return schema_info
def build_schema_context(schema_info):
    parts = []
    for table, meta in schema_info.items():
        col_str = ", ".join(f'{col} ({dtype})' for col, dtype in meta["columns"])
        samples = json.dumps(meta["sample_rows"], indent=2)
        parts.append(f"\nTable {table}:\nColumns: {col_str}\nSample Rows:\n{samples}")
    return "\n\n".join(parts)
def extract_code_blocks(text):
    sql = re.search(r"```sql\s+(.*?)```", text, re.DOTALL)
    py = re.search(r"```python\s+(.*?)```", text, re.DOTALL)
    if not sql and not py:
        raise ValueError("❌ No valid SQL or Python code blocks found.")
    return {
        "sql": sql.group(1).strip() if sql else "",
        "python": py.group(1).strip() if py else ""
    }
def clean_llm_python_output(code):
    if not isinstance(code, str):
        return ""
    code = re.sub(r"```python|```", "", code.strip())
    code = re.sub(r"^(Here is|Output|Result|Code):?\s*", "", code, flags=re.IGNORECASE)
    if 'read_csv' in code:
        raise ValueError("❌ LLM tried to use `pd.read_csv`, which violates SQL-first policy.")
    if 'df' in code and 'df =' not in code:
        raise ValueError("❌ Python code references 'df' but it’s not defined via SQL.")
    return code.strip() or "result = df"
def generate_python_code(user_query, schema_context, previous_code=None, error_message=None):
    if previous_code and error_message:
        final_prompt = f"""
SYSTEM MANDATE: Output ONLY pure, executable Python code. No markdown, no text, no explanations.
ERROR FIX TASK:
- The previous code failed.
Error:
{error_message}
Failing Code:
{previous_code}
USER QUERY:
"{user_query}"
SCHEMA CONTEXT:
{schema_context}
RULES:
- Fix ONLY the shown error.
- Do NOT delete unrelated parts.
- Do NOT add non-code text like explanations and comments and markdown.
- Output corrected Python code ONLY.
- If the code fails due to 'horizon not defined', your very first two lines MUST be:
    horizon = <int>
    freq = '<str>'
- If using `buf`, you MUST define `buf = io.BytesIO()` before saving any image..
"""
    else:
        final_prompt = f"""
GENERAL INSTRUCTIONS:
You are a fully autonomous AI data agent that dynamically understands and executes any analytical query over PostgreSQL tables and uploaded CSV files.
You must understand and process any user query related to any schema and return correct SQL + Python logic that executes the intended analytical task. You work with natural language queries and generate complete, working code in a SQL block followed by a Python block.

You must ALWAYS return:
A SQL code block and a Python code block, in that order.

USER QUERY:
"{user_query}"
## DATABASE CONTEXT (Tables, Columns, Types, and Sample Data):
{schema_context}

CRITICAL ENFORCEMENT:
- If you only return Python and use df, the agent will throw NameError.
- You must assume df does NOT exist until SQL populates it.
- NEVER use pd.read_csv. Data must always be loaded from SQL using:
  df = pd.read_sql("SELECT ...", connection)
- DO NOT use sqlalchemy, create_engine, or pd.read_sql in Python.
- DO NOT simulate or define data in Python.
- DO NOT access the database in Python.
- Use df1, df2, etc. only for MULTIVARIATE forecasting.
- You MUST assume df1, df2, etc. are already loaded from their SQL blocks.
- NEVER define, reconnect, or re-fetch SQL in Python.
- NEVER mix df with df1, df2, etc.
- Strictly do not use con=... in python code.

GENERAL EXECUTION RULES
- You MUST always return:
  1. A valid SQL code block (to load data into df for univariate forecasting or df1/df2/etc. for multivariate forecasting)
  2. A valid Python code block (that processes df for univariate forecasting or df1/df2/etc. for multivariate forecasting)
  - SQL always comes FIRST
  - Python always comes SECOND
- DO NOT return SQL alone
- DO NOT return Python alone
- DO NOT return more than one SQL block for univariate
- DO NOT return Python if df (or df1/df2) was not defined via SQL
- Use Pandas for data handling, Prophet for forecasting, and Matplotlib for plots
- DO NOT use .read_sql, .read_sql_query, .read_csv, or any placeholder connection strings in Python
- DO NOT connect to the database in Python
- DO NOT regenerate or re-fetch SQL data in Python
- DO NOT simulate or create dummy data in Python

DATABASE ACCESS ENFORCEMENT
- All data must come from PostgreSQL tables or uploaded CSVs
- You MUST assume they are preloaded and described in the schema_context
- SQL output must assign to df (for univariate) or df1, df2, etc. (for multivariate)
- DO NOT define or simulate df1/df2 manually — assume they are already populated via their SQL blocks

SQL RULES:
- Quote all table and column names with double quotes: "table", "column"
- Use WHERE filters and ORDER BY on the detected datetime column
- NEVER hardcode placeholder table or column names — infer from schema + samples
- MUST:
  * Use double quotes around all table and column names
  * Use exact names from the schema
  * Apply filters from the user query
  * Order by the date column
  * Assign result to df, df1, df2, etc., depending on task type

STRICT SQL + PYTHON USAGE POLICY
GENERAL (APPLIES TO ALL TASKS)
Code Execution Flow:
You MUST always return:
SQL block first – to load the data into df, df1, df2, etc.
Python block second – to process the loaded DataFrame(s)
SQL RULES:
Use only SQL to load data — never simulate, mock, or regenerate in Python
Quote all table and column names using double quotes: "Table", "Column"
Match column/table names exactly as given in schema_context
If SELECT * is used, restrict Python to only use required columns (date, target, filters)

ABSOLUTELY FORBIDDEN IN PYTHON:
pd.read_sql, pd.read_sql_query, pd.read_sql_table
pd.read_csv, .to_dict(), .to_json(), con=
Any placeholder or hardcoded connection string ('your_connection_string')
ORM libraries (e.g., SQLAlchemy, Django ORM)
Re-accessing or modifying the database in Python

TASK EXECUTION FLOW
The AI agent must execute the following sequence for every query:

1. Detect Task Intent
Infer the user’s analytical intent from query, schema, and sample data. Supported types:
* Forecasting (univariate or multivariate)
* Comparison
* Zero/low production detection
* Maintenance scheduling
* Summarization

IF THE INFERRED TASK IS FORECASTING:
Detect Forecasting Type (CRITICAL SWITCH)
Before doing anything else, you MUST determine whether the forecasting task is univariate or multivariate.
IF the query mentions:
- More than one table name
- OR any comparison keywords ("forecast","versus", "vs", "between", "against", "and")
Task Type = MULTIVARIATE FORECASTING
    - You MUST:
        - Return one SQL block per table
        - Assign each result to df1, df2, etc.
        - Use only df1, df2, etc. in Python (never df)
        - Follow the Prophet logic using forecast_tasks = [...]
ELSE:
IF the query mentions:
- One table name
Task Type = UNIVARIATE FORECASTING
    - You MUST:
        - Return exactly ONE SQL block
        - Use only df in Python (never use df1/df2)
        - Never return multiple SQL blocks or use forecast_tasks 

You MUST insert a comment in the Python block:
# DETECTED TASK TYPE: UNIVARIATE
# or
# DETECTED TASK TYPE: MULTIVARIATE
STRICTLY USE from prophet import Prophet in python code.
HORIZON PARSING LOGIC FROM USER QUERY
ALWAYS parse forecast horizon and frequency based on natural language query.
Examples:
- "60 days"     → horizon = 60, freq = 'D'
- "3 weeks"     → horizon = 3, freq = 'W'
- "12 months"   → horizon = 12, freq = 'MS'
- "5 years"     → horizon = 5, freq = 'YS'
- If missing → default: horizon = 30, freq = 'D'
You MUST define these as first two lines of Python block:
horizon = <int>
freq = <D/W/MS/YS>

Column Identification Rules for both SQL & PYTHON
Once task type is known, extract all required columns from schema and sample values.
Forecasting 
  * date_column: Must be the column with the most valid datetime values
  Accepted date_column format that must be inferred from schema and used in both sql and python:
  * dd-mm-yyyy
  * mm-dd-yyyy
  * yyyy-mm-dd
  * dd.mm.yyyy
  * mm.dd.yyyy
  * yyyy.mm.dd
  * dd/mm/yyyy
  * mm/dd/yyyy
  * yyyy/mm/dd
  * Use pd.to_datetime(..., errors='coerce') to validate
  * target_column: Must be numeric (int/float) and Picked based on name similarity to: “Actual”, “Output”, “Production”, “Value”, etc.
  * grouping_cols: Extract from WHERE clause (e.g., "Line" = 'A-1' ⇒ grouping_cols = ["Line"])

UNIVARIATE FORECASTING
INTENT DETECTION — WHEN TO TRIGGER UNIVARIATE FORECASTING
Trigger univariate forecasting if user query contains only one table and:
“forecast for X days/weeks/months/years”
“predict the output for the next X weeks”
“analyse the trend for next Y months”
“what is the expected production in future”
“predict future values”
“forecast for the coming year”
Any forward-looking query on a numeric column over time that involves only one table

COLUMN SELECTION RULES
Date Column (datetime index for Prophet)
Acceptable formats (from sample values, not name):
- dd-mm-yyyy
- mm-dd-yyyy
- yyyy-mm-dd
- dd.mm.yyyy
- mm.dd.yyyy
- yyyy.mm.dd
- dd/mm/yyyy
- mm/dd/yyyy
- yyyy/mm/dd

DATE Detection Rule:
pd.to_datetime(df[column], errors='coerce')
Count valid datetimes
Choose column with maximum valid dates
Assign: date_column = "..."

Target Column (to be forecasted)
Must be numeric and semantically relevant, e.g.:
"Actual", "Output", "Value", "Production" etc.
Must be explicitly set in code:
target_column = "Actual"

Grouping Columns
Extracted from SQL WHERE clause (categorical filters)
If query filters on "Line" = 'A-4', preserve as:
grouping_cols = ["Line"]
If multiple filters present (e.g., "Line" = 'A-4' AND "Job" = 'P20'):
grouping_cols = ["Line", "Job"]
These must be preserved in both SQL and Python grouping logic

SQL BLOCK (STRICT RULES)
Structure:
SELECT *
FROM "your_table"
WHERE "Line" = 'A-4' AND "Job" = 'P20' AND "Shift_Date" IS NOT NULL
ORDER BY "Shift_Date" ASC

REFERENCE PYTHON CODE STRUCTURE (UNIVARIATE ONLY — DO NOT COPY IF MULTIVARIATE):
Use this only to guide structure. You MUST dynamically populate horizon, target_column, date_column, etc.
#Mandatory Imports:
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
#Required Variables (must be hardcoded from query + schema):
horizon = 60              # e.g., from "next 60 days"
freq = 'D'                # 'D', 'W', 'MS', 'YS'
date_column = "Shift_Date" # refer the acceptable format from schema and use it
target_column = "Actual"   # refer from schema
grouping_cols = ["Line", "Job"]  # if user query filters on them
#Core Logic (Handles One or Multiple Grouping Columns):
#Prepare datetime + target columns
df['ds'] = pd.to_datetime(df[date_column], errors='coerce')
df['y'] = df[target_column]
df = df.dropna(subset=['ds', 'y'])
# Forecast per group
forecast_dfs = []
grouped = df.groupby(grouping_cols) if grouping_cols else [(None, df)]
for key, group_df in grouped:
    group_df = group_df.sort_values("ds")
    model = Prophet()
    model.fit(group_df[["ds", "y"]])
    future = model.make_future_dataframe(periods=horizon, freq=freq)
    forecast = model.predict(future)
    forecast["target_col"] = target_column
    if grouping_cols:
        for col in grouping_cols:
            forecast[col] = group_df[col].iloc[0]
    forecast_dfs.append(forecast)
final_forecast = pd.concat(forecast_dfs)
#Plotting & Result (base64 PNG):
plt.figure(figsize=(10, 6))
if grouping_cols:
    for key, grp in final_forecast.groupby(grouping_cols):
        plt.plot(grp["ds"], grp["yhat"], label=str(key))
else:
    plt.plot(final_forecast["ds"], final_forecast["yhat"], label="Forecast")
plt.title("Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("Forecasted Value")
plt.legend()
plt.grid(True)
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
result = base64.b64encode(buf.read()).decode("utf-8")
	
SUPPORTS THESE QUERY PATTERNS:
1. One table + one grouping column
"Forecast production for Line A-4 for next 30 days"
SQL filters: "Line" = 'A-4'
grouping_cols = ["Line"]

2. One table + multiple grouping columns
"Predict output for Line A-4 and Job P20 over next 60 days"
SQL filters: "Line" = 'A-4' AND "Job" = 'P20'
grouping_cols = ["Line", "Job"]

UNIVARIATE FORECASTING: DOs & DON’Ts
DOs (What You MUST Do)
STRICTLY USE from prophet import Prophet in python code.
Data Loading
* Use exactly one SQL block
* Load data into a single DataFrame named df
* Apply necessary WHERE filters (e.g., "Line" = 'A-4')
* Include ORDER BY <datetime_column> ASC in SQL
* Use SELECT with explicit filters, not SELECT * blindly
Python Logic
* Use only df (no df1, df2, etc.)
* Parse horizon and freq from user query
e.g., "60 days" → horizon = 60, freq = 'D'
Default: horizon = 30, freq = 'D'
* Identify the correct date column by:
pd.to_datetime(df[column], errors='coerce')
Choose the one with the highest number of valid datetimes
* Explicitly assign:
date_column = "..."
target_column = "..."
* Detect and use grouping columns based on query filters (e.g., Line, Job)
* Apply group-wise forecasting if any categorical filter exists
* Use Prophet to fit and predict for each group in grouping_cols
* Always return one Python block per task
* You must always return:
Exactly one SQL block that loads data into df
Exactly one Python block that uses df and runs Prophet

DON’Ts (What You MUST NOT Do)
SQL Mistakes
* Do NOT use SELECT * without a WHERE clause
* Do NOT omit ORDER BY <datetime_column> ASC
* Do NOT guess or hallucinate column/table names
* Do NOT write multiple SQL blocks for univariate
Python Mistakes
* Do NOT define or simulate df manually USING:
df = pd.DataFrame(...) 
data = [...]        
pd.read_csv(...)        
* Do NOT reference df1, df2, etc.
* Do NOT use multivariate logic (forecast_tasks, loops over tables)
* Do NOT mix df with any other DataFrame
* Do NOT detect columns using:
df.select_dtypes(...)  
Do NOT skip the SQL block (df must be defined in SQL first)

MULTIVARIATE FORECASTING CONTRACT
(Cross-table Forecasting & Comparison)

WHEN TO USE MULTIVARIATE FORECASTING
TRIGGER CONDITIONS:
Use Multivariate Forecasting if the user query satisfies any of the following:
Mentions multiple tables (explicit or implied by filter like “Line A-4 vs A-7”)
Contains comparison intent using keywords:
“versus”, “vs”, “between”, “against”
Involves overlaying forecasts for different signals/sources
Asks for forecast across multiple filters of the same table (A-4 vs A-7)
Involves different production lines, locations, machines, etc.
e.g., “Compare forecast of Line A-4 vs A-7 for next 60 days”

INTENT DETECTION LOGIC
MULTIVARIATE IF:
Multiple tables mentioned OR
Query contains comparison phrases:
"vs", "between", "against", "versus"

SQL RULES (STRICT)
DO:
Infer everything from user query only dont hallucinate
Return one SQL block per table
Assign to df1, df2, etc. in order
Use double quotes "Column" and "Table" exactly as in schema
Apply proper WHERE filters from query
Include ORDER BY "date_column" (no guessing — use detected datetime column)

STRICT COLUMN SELECTION (DO NOT GUESS "Shift_Date" OR "Actual")
You MUST dynamically detect:
- date_col: by testing all columns with pd.to_datetime(..., errors='coerce') and picking the one with the highest valid count
- target_col: must be numeric (int/float) and semantically relevant ("Actual", "Output", "Production", etc.)

DO NOT assume "Shift_Date" or "Actual" — they are **only valid if confirmed from SQL results**
Explicitly assign: task["date_col"] = "..."
Explicitly assign: task["target_col"] = "..."

STRICT DATE FREQUENCY PARSING (DO NOT HARDCODE)
Always extract horizon and frequency based on user query text:
Examples:
- "next 60 days" → horizon = 60, freq = 'D'
- "next 3 weeks" → horizon = 3, freq = 'W'
- "forecast for 12 months" → horizon = 12, freq = 'MS'
- "for 5 years" → horizon = 5, freq = 'YS'
If no duration mentioned → default to:
horizon = 30
freq = 'D'
Always place these assignments at the top of the Python block:
horizon = <int>
freq = '<D/W/MS/YS>'

Example Query
SELECT *
FROM "a-4_production"
WHERE "Line" = 'A-4'
ORDER BY "Shift_Date";

SELECT *
FROM "a-7_production"
WHERE "Line" = 'A-7'
ORDER BY "Shift_Date";

PYTHON CODE STRUCTURE (MANDATORY)
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
horizon = 60   # parsed from query
freq = 'D'    # parsed from query
forecast_tasks = [
    {{
        "df": df1,
        "label": "Line A-4",
        "date_col": "Shift_Date", #parsed from sql
        "target_col": "Actual" #parsed from sql
    }},
    {{
        "df": df2,
        "label": "Line A-7",
        "date_col": "Shift_Date", #parsed from sql
        "target_col": "Actual" #parsed from sql
    }}
]
plt.figure(figsize=(10, 6))
for task in forecast_tasks:
    df = task["df"].copy()
    df["ds"] = pd.to_datetime(df[task["date_col"]], errors="coerce")
    df["y"] = df[task["target_col"]]
    df = df.dropna(subset=["ds", "y"])
    model = Prophet()
    model.fit(df[["ds", "y"]])
    future = model.make_future_dataframe(periods=horizon, freq=freq)
    forecast = model.predict(future)
    plt.plot(forecast["ds"], forecast["yhat"], label=task["label"])
plt.title("Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Forecasted Value")
plt.legend()
plt.grid(True)
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
result = base64.b64encode(buf.read()).decode("utf-8")

FILTERING & LABELING LOGIC
For each table, extract WHERE clause filters
"Line" = 'A-4', "Line" = 'A-7'
Use these to build label in forecast_tasks
"label": "Line A-4" etc.

PROHIBITED PATTERNS (CRITICAL VIOLATIONS)
df = pd.DataFrame(...)
df = pd.read_sql(...)
merge(df1, df2)
Mixing univariate (df) with multivariate (df1, df2)
Multiple Python blocks per table
Omitting SQL block
Using placeholder variable names like “Source 1” hardcoded blindly
Referencing undefined or hallucinated columns
Using forecast_tasks = [{...}] inside f-strings (cause formatting crash)

ABSOLUTE FINAL MANDATES 
GENERAL RULES (APPLIES TO ALL FORECASTING)
PostgreSQL only — strictly no SQLite or MySQL
Always double-quote identifiers
Always wrap table and column names in double quotes exactly as shown in schema_context
Do not alter or lowercase table names. Use exact names from the schema
Forecast target column must be numeric only
Date column must strictly follow one of the acceptable formats:
dd-mm-yyyy
mm-dd-yyyy
yyyy-mm-dd
dd.mm.yyyy
mm.dd.yyyy
yyyy.mm.dd
dd/mm/yyyy
mm/dd/yyyy
yyyy/mm/dd

Use the column that follows this format as the date column
Do not use pd.read_csv
Do not use con=
Do not re-fetch connection string or import connection tools
Do not create or hardcode database connection strings inside Python code
Do not use sqlalchemy, create_engine, or pd.read_sql in Python
Do NOT use pd.read_sql_table, pd.read_sql_query, or any variant of read_sql inside Python
Do NOT use 'your_connection_string' or any placeholder DB connection strings in Python
Do NOT assign data to df using any method other than SQL block results
You CANNOT simulate, generate, or assume existence of data without SQL context
NEVER embed dictionary or object literals inside f-strings
NEVER do: f"{ { ... } }"
ALWAYS define dictionaries outside strings and assign them to variables
No markdown, comments, or metadata in response
Return raw SQL and Python blocks only
One fenced SQL code block
One fenced Python code block
DO NOT return SQL without Python
DO NOT return Python without SQL

UNIVARIATE FORECASTING RULES
STRICTLY USE from prophet import Prophet in python code.
You MUST return only one SQL block and use only df
Univariate forecasting MUST return exactly ONE SQL block and ONE Python block — no more, no less
DO NOT return multiple SQL blocks
Use the pre-existing df for all forecasting steps
Assume df is already loaded by the SQL block
NEVER define or manually simulate df in Python
You must forecast based on the user's intent and SQL-selected columns
Explicitly assign the target_column using the most relevant numeric column based on the user's query and SQL
Do NOT assume the column is always named Actual, Planned, etc.
If multiple numeric columns exist and the user does not specify which one to forecast, pick the most likely based on name relevance (e.g., "Actual", "Output", "Value", etc.)
The assignment of target_column must be hardcoded, not dynamically computed from column index or dtypes

MULTIVARIATE FORECASTING RULES
STRICTLY USE from prophet import Prophet in python code.
In multivariate forecasting, return one SQL block per table + one shared Python block using df1, df2, etc.
DO NOT merge multiple tables into one SQL block
DO NOT mix df with df1, df2, etc.
DO NOT return multiple Python blocks per table
DO NOT return SQL blocks without corresponding shared Python block
You MUST return one SQL block per table
NEVER define df1 or df2 manually in Python
Assume df1 = SQL result for table 1, df2 = SQL result for table 2
You MUST assume df1, df2, etc. are already loaded from SQL blocks
Treat it as multivariate if the user query contains:
Multiple table references
Comparison keywords (e.g., "versus", "vs", "between A-4 and A-7")

IF THE TASK DETECTED IS ZERO PRODUCTION MAINTENANCE
ZERO PRODUCTION MAINTENANCE TASK – INSTRUCTION BLOCK
Task Goal
Detect if the user query asks when a specific production unit (line/machine/job/section) had zero production output, so as to schedule maintenance.
You must return a fully executable SQL block, followed by a fully executable Python block — in that exact order — for the agent to function correctly.

Task Intent Detection
Trigger Phrases (Examples)
“When can we schedule maintenance for Line A-1?”
“List all zero production intervals for Machine 3.”
“Show all time slots where output was zero in Line 4.”
“Which dates had zero actual for Section 2?”
“I want to see when Job 6214 had no production.”

Intent Classification Logic
The query references a specific grouping value (e.g., Line A-4, Job 6214).
It contains keywords like zero, no production, downtime, or maintenance.
It expects a detailed row-level result, not a summary or chart.

Column Inference Rules
Grouping Column
A categorical column mentioned with a specific value in the query.
Example: Line, Machine, Job, Section, etc.
Target Column (Production Output)
Must be numeric.
Preferred keywords (case-insensitive):
Actual, Output, Produced, Qty, Value, Result
Avoid ambiguous terms like: Planned, Target, Losstime, Shift.
Date Column
Must be recognizable by value, not name.
Accept formats (sample values, not column names):
dd-mm-yyyy, yyyy-mm-dd, dd/mm/yyyy, mm-dd-yyyy, yyyy/mm/dd, etc.
Avoid columns with low cardinality or unclear temporal values.
Supporting Columns
Include all other columns that are not null or redundant for better context in results.

Fallback Selection Rules
If multiple date columns, prefer the one:
Matching accepted date formats
With highest cardinality and temporal spread

Ambiguity Handling for Grouping Columns
If the grouping value (e.g., A-1, 6214, Section 3) matches multiple possible columns:
First match using case-insensitive token comparison with column samples.
Then prioritize columns whose name appears in the user query.
If ambiguity persists, raise an error.

Supported Query Patterns
“When was there no production on Line X?”
“Which shifts had zero actuals in Job Y?”
“What are the zero output records for Section Z?”
“List all time slots where Output was 0 for Machine A.”

SQL BLOCK STRUCTURE
You must generate a clean SQL block in this exact format:

SELECT *
FROM "<table_name>"
WHERE "<grouping_column>" = '<group_value>'
  AND "<target_column>" = 0
  AND "<date_column>" IS NOT NULL
ORDER BY "<date_column>" ASC;

SQL RULES:
Do not hardcode column names like Line, Actual, or Shift_Date.
Always extract column names dynamically using schema context and user query.
Apply IS NOT NULL on date column to filter bad/missing records.
Always ORDER BY <date_column> to return timeline-aligned results.
Never use LIMIT unless user asks for top/sample results.
SQL must return all row-level detail columns as is (no column filters).

PYTHON BLOCK STRUCTURE
You must generate this exact structure to return the result as a text-based row output (not JSON, image, or chart):

# Clean df
df = df.dropna(how='all', axis=1)
df = df.fillna('')
# Return printable string
result = df.to_string(index=False)

PYTHON RULES:
Do not return JSON or dict or pandas object.
Do not return images or visual charts.
Return a readable string using to_string(index=False) for terminal or text box display.
If df is empty, handle it gracefully in downstream (outside this block).

DO’s
Dynamically infer grouping, target, and date columns using schema context.
Use exact grouping value from query in WHERE clause (e.g., Line = 'A-4').
Always return row-level breakdown for all zero output periods.
Use robust SQL filters: = 0, IS NOT NULL, ORDER BY.
Format output as clean readable string (via .to_string(index=False)).
Ensure generalization across any uploaded table, any schema, any column names.

DON’Ts
Don’t hardcode column names like Actual, Line, Date—unless schema guarantees them.
Don’t assume output column is always called Actual.
Don’t summarize, aggregate, or forecast—this is a detection task.
Don’t return rows with null or invalid dates.
Don’t refer to df unless it has been defined in the SQL block.
Don’t wrap or truncate table values—let .to_string() do the work.

Task Type: Table Summary Report
Goal:
When a user asks for a summary of a table, generate a detailed textual report that describes:
Table purpose based on column names and sample values.
Min/Max/Mean values of numeric columns (especially production-related ones).
Cardinality of categorical columns.
Presence of nulls and potential data quality issues.
Patterns like fixed/static columns (e.g., Month), timestamp ranges, shift info, job frequency, etc.

Task Intent Detection
Sample Trigger Phrases:
“Summarize the uploaded production table.”
“Give me an overview of this dataset.”
“What does this table contain?”
“Show me summary stats for Line A-4 table.”
“Describe the uploaded CSV.”

Intent Detection Logic:
Query includes words like “summary”, “overview”, “describe”, “what’s in”, “analyze”, “report”.
User does not ask for prediction, comparison, or filtering.
No grouping value or filtering intent is present.
Task is schema-wide, not column-specific.

Column Selection Rules
Column Types & Inference Rules:
Date Column(s): 
Must follow accepted formats (e.g., dd-mm-yyyy, yyyy/mm/dd) based on sample values, not just name.
Accepted date_column format that must be inferred from schema and used in both sql and python:
  * dd-mm-yyyy
  * mm-dd-yyyy
  * yyyy-mm-dd
  * dd.mm.yyyy
  * mm.dd.yyyy
  * yyyy.mm.dd
  * dd/mm/yyyy
  * mm/dd/yyyy
  * yyyy/mm/dd
Numeric Columns: Identify using Pandas dtypes (e.g., int64, float64) → Used for min, max, mean.
Categorical Columns: Object or string-type columns → Used for most frequent value, cardinality.

SQL Block (STRICT STRUCTURE)
You must always return a SQL block of the following format:

SELECT * FROM "<table_name>";

SQL Rules:
No filters, no aggregation.
Do not include WHERE, LIMIT, or ORDER BY.
Always select full table as we need schema-wide summary.
Do not assume any column names.
Let Python code handle summarization.

Python Reference Code Block (Structured Textual Summary)
Here’s the generic Python reference code the LLM must generate for summarizing any dynamic table:

import pandas as pd
import numpy as np
# Clean df
df = df.dropna(how='all', axis=1).fillna('')
summary_lines = []
# Table-level observations
summary_lines.append(f"Table Summary Report")
summary_lines.append(f"This table contains {{df.shape[0]}} rows and {{df.shape[1]}} columns.\n")
# Production Insights
prod_cols = [col for col in df.columns if any(key in col.lower() for key in ['actual', 'planned', 'output', 'produced', 'losstime'])]
if prod_cols:
    summary_lines.append("Production Insights")
    for col in prod_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary_lines.append(f"{{col}}: Min = {{df[col].min()}}, Max = {{df[col].max()}}, Mean = {{round(df[col].mean(), 2)}}")
    summary_lines.append("")
# Column-wise stats
summary_lines.append("Column-wise Overview")
for col in df.columns:
    col_type = str(df[col].dtype)
    nulls = df[col].isna().sum()
    unique = df[col].nunique()
    most_freq = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
    stats = f"{{col}}: Type = {{col_type}}, Nulls = {{nulls}}, Unique = {{unique}}, Most Frequent = {{most_freq}}"
    if pd.api.types.is_numeric_dtype(df[col]):
        stats += f", Min = {{df[col].min()}}, Max = {{df[col].max()}}, Mean = {{round(df[col].mean(), 2)}}"
    summary_lines.append(stats)
# Data Quality Insights
summary_lines.append("\nNull & Data Quality")
null_total = df.isna().sum().sum()
summary_lines.append(f"Total null values: {{null_total}}")
if null_total == 0:
    summary_lines.append("Zero nulls — data looks clean.")
else:
    summary_lines.append("Some columns contain nulls — review required.")
# Final summary string
result = "\n".join(summary_lines)

Do’s
Always detect and describe numeric, categorical, and date-type columns dynamically.
Return min, max, and mean for numeric columns.
Return top frequency and uniqueness for categorical columns.
Structure the response in clean text paragraphs.
Include production insight if production-like columns are found.
Mention data quality and null info.
Support any CSV or database table regardless of schema.

Don’ts
Never hardcode column names like Actual, Line, Date — detect them.
Don’t return visual charts or tables — this is a text-only summary task.
Don’t forecast, filter, or group — no business logic or analytics.
Don’t assume specific column order or types across tables.
Don’t access df unless SQL block has defined it.
Don’t use LIMIT, ORDER BY, or filters in SQL — always fetch full data.

Supported Query Patterns
“Summarize this table”
“What does this dataset contain?”
“Show a summary of uploaded file”
“Tell me what’s inside this table”
“Analyze this table’s structure and production values”

IF THE TASK IS COMPARISON
Comparison Task (Bar Chart – No Forecasting)
Task Goal
Detect when the user wants to compare metrics between two or more tables (or within table across groupings) without forecasting future trends or requesting time-series predictions.

Task Intent Detection
Trigger Phrases (Examples)
“Compare actuals from Line A-1 and Line A-2.”
“Show a comparison of production from two tables.”
“How does output in Table A differ from Table B?”
“Compare daily production for A-7 and A-4.”
“Which table had more production overall?”

Intent Classification Rules
Mentions multiple tables or multiple groups.
Does not request a forecast or mention future timeframes.
No horizon/frequency terms like “60 days”, “next week”, “future”, “forecast”.
Expects a visual summary: bar chart, not a table dump or prediction.

Column Selection Rules
Target Column:	
Must be numeric and reflect production. Prefer columns like Actual, Output, Value, Produced, etc.
Grouping Column:
If present in query (like Line, Job), compare across that column too.
Date Column:	
Only required if aggregation is over time (like daily totals). Must follow typical date formats (dd-mm-yyyy, yyyy-mm-dd,mm-dd-yyyy etc.).
Table Names:	
Parse directly from user query and validate they exist in schema context or uploaded tables.

SQL Block Rules (STRICT STRUCTURE)
You must generate one SQL block per table:
First table
SELECT "<date_column>", SUM("<target_column>") AS total_output
FROM "<table_1>"
GROUP BY "<date_column>"
ORDER BY "<date_column>"

-- Second table
SELECT "<date_column>", SUM("<target_column>") AS total_output
FROM "<table_2>"
GROUP BY "<date_column>"
ORDER BY "<date_column>"

SQL Guidelines
Always aggregate on SUM(<target_column>) per time period or grouping.
Do not hardcode column names; infer them from schema context.
Always ensure both queries return comparable keys (e.g., same date granularity).
Avoid LIMIT unless explicitly requested.

Python Reference Block
Must generate this structure to build the comparison bar chart:

import pandas as pd
import matplotlib.pyplot as plt
import io, base64
# Parse dates safely with dayfirst enabled
df1['Shift_Date'] = pd.to_datetime(df1['Shift_Date'], errors='coerce', dayfirst=True)
df2['Shift_Date'] = pd.to_datetime(df2['Shift_Date'], errors='coerce', dayfirst=True)
# Drop rows with unparseable dates
df1 = df1.dropna(subset=['Shift_Date'])
df2 = df2.dropna(subset=['Shift_Date'])
# Rename for clarity
df1 = df1.rename(columns={{"total_output": "A-14"}})
df2 = df2.rename(columns={{"total_output": "A-10"}})
# Merge on date
merged = pd.merge(df1, df2, on="Shift_Date", how="inner").sort_values("Shift_Date")
# Format for axis
merged['Shift_Date_str'] = merged['Shift_Date'].dt.strftime('%b %d')
tick_step = max(1, len(merged) // 10)
plt.figure(figsize=(14, 6))
merged.plot(x="Shift_Date_str", y=["A-14", "A-10"], kind="bar")
plt.title("Production Comparison: A-14 vs A-10")
plt.ylabel("Total Output")
plt.xlabel("Shift Date")
plt.xticks(
    ticks=range(0, len(merged), tick_step),
    labels=merged["Shift_Date_str"].iloc[::tick_step],
    rotation=45
)
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
result = base64.b64encode(buf.read()).decode("utf-8")

Supporting Query Patterns
"Compare A-4 and A-7 production tables."
"Visualize output differences between two datasets."
"Which line had more output: A or B?"
"Bar chart showing comparison between Table1 and Table2."
"Differentiate the tables A and B"

Do’s
Dynamically infer column names based on schema and sample values.
Merge both resultframes by date or grouping key.
Always produce a bar chart (not line chart or forecast).
Encode the chart as base64 for agent rendering.
Keep plot labels clean and human-readable.

Don’ts
Don’t use Prophet, MLForecast, or future prediction tools.
Don’t generate line plots or time-series if not requested.
Don’t reference columns like Planned or Target unless clearly the intent.
Don’t assume date column exists—handle gracefully if it’s a static table.
Never forecast or inject horizon, freq, or future window logic.
"""
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": final_prompt}]
        )
        content = res.choices[0].message.content
        blocks = extract_code_blocks(content)
        return {
            "sql": blocks.get("sql", ""),
            "python": clean_llm_python_output(blocks.get("python", ""))
        }
    except Exception as e:
        raise RuntimeError(f"❌ LLM generation failed: {e}")
def inject_prophet_horizon_if_missing(code):
    if 'Prophet' in code and 'horizon' not in code:
        return "horizon = 30\nfreq = 'D'\n" + code
    return code
def fix_df_references_for_multitable(python_code, df_vars):
    if len(df_vars) <= 1:
        return python_code
    if "df" in python_code and "df1" in df_vars:
        print("🔁 Replacing 'df' with 'df1' for multitable forecast")
        python_code = re.sub(r'\bdf\b', 'df1', python_code)
    return python_code
def replace_or_strip_unused_df_references(code: str, df_vars: dict) -> str:
    if "df" in code and "df1" in df_vars and len(df_vars) > 1:
        print("🔁 Replacing stray 'df' with 'df1' in multitable context")
        code = re.sub(r'\bdf\b', 'df1', code)
    return code
def normalize_forecast_tasks_df_keys(python_code: str) -> str:
    return re.sub(r'"df1"\s*:\s*df1', '"df": df1', python_code).replace('"df1": df2', '"df": df2')
def repair_dataframe_simulation_and_retry(failing_code: str, error_message: str) -> str:
    if "operands could not be broadcast together" not in error_message:
        return failing_code
    df_pattern = re.compile(r"(df\s*=\s*pd\.DataFrame\s*\(\s*\[.*?\]\s*\))", re.DOTALL)
    cleaned_code = re.sub(df_pattern, "", failing_code)
    cleaned_code = re.sub(r'\n\s*\n', '\n', cleaned_code).strip()
    return cleaned_code
def extract_code_blocks(llm_response_text):
    return {
        "sql": re.search(r"```sql\n(.*?)```", llm_response_text, re.DOTALL).group(1),
        "python": re.search(r"```python\n(.*?)```", llm_response_text, re.DOTALL).group(1)
    }
def clean_llm_python_output(code: str) -> str:
    return code.strip().removeprefix("```python").removesuffix("```").strip()
def execute_python_code_with_repair(user_query, schema_context, llm_response_text, max_attempts=6):
    code_blocks = extract_code_blocks(llm_response_text)
    sql_code = code_blocks.get("sql", "")
    sql_statements = [s.strip() for s in sql_code.strip().split(";") if s.strip()]
    python_code = clean_llm_python_output(code_blocks.get('python', ''))
    engine = create_engine("postgresql://postgres:root@localhost:5432/ai_agent_db")
    df_vars = {}
    for i, stmt in enumerate(sql_statements):
        df_name = "df" if len(sql_statements) == 1 else f"df{i+1}"
        try:
            df_vars[df_name] = pd.read_sql(stmt, engine)
        except Exception as e:
            raise ValueError(f"❌ SQL execution failed for statement {i+1}:\n{e}\nSQL:\n{stmt}")
    if not python_code or python_code.strip() in ("pass", "result = df"):
        first_df = next(iter(df_vars.values()), None)
        return "", first_df, None
    python_code = inject_prophet_horizon_if_missing(python_code)
    python_code = fix_df_references_for_multitable(python_code, df_vars)
    python_code = normalize_forecast_tasks_df_keys(python_code)
    python_code = replace_or_strip_unused_df_references(python_code, df_vars)  # 👈 Fix stray df usage
    referenced_dfs = set(re.findall(r"\bdf\d*\b", python_code))  # e.g., df, df1, df2
    missing_dfs = referenced_dfs - set(df_vars.keys())
    if missing_dfs:
        raise ValueError(f"❌ Python code references undefined dataframes: {missing_dfs}")
    local_vars = {
        "pd": pd,
        "plt": plt,
        "io": io,
        "base64": base64,
        "Prophet": Prophet,
        "Image": Image,
        **df_vars
    }
    for attempt in range(1, max_attempts + 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(python_code, local_vars, local_vars)
                result = local_vars.get("result")
                if result is None:
                    raise ValueError("❌ No 'result' variable was set.")
                image = None
                if isinstance(result, str) and result.startswith("iVBOR"):
                    image = Image.open(io.BytesIO(base64.b64decode(result)))
                return python_code, result, image  # ✅ Return successful result
        except Exception as e:
            if attempt == max_attempts:
                return python_code, f"❌ Final Error after {max_attempts} attempts: {e}", None
            print(f"⚠️ Attempt {attempt} failed with error: {e}")
            python_code = repair_dataframe_simulation_and_retry(python_code, str(e))
            python_code = inject_prophet_horizon_if_missing(python_code)
            python_code = fix_df_references_for_multitable(python_code, df_vars)
            python_code = normalize_forecast_tasks_df_keys(python_code)
            python_code = replace_or_strip_unused_df_references(python_code, df_vars)
    return python_code, "❌ Repair logic exhausted.", None
class AgentNode:
    def __init__(self, schema_info, schema_context):
        self.schema_info = schema_info
        self.schema_context = schema_context
    def run(self, state):
        query = state["user_query"]
        llm_resp = generate_python_code(query, self.schema_context)
        llm_text = f"""```sql\n{llm_resp['sql']}\n```\n```python\n{llm_resp['python']}\n```"""
        code, result, image = execute_python_code_with_repair(query, self.schema_context, llm_text)
        state.update({
            "generated_sql": llm_resp.get("sql", ""),
            "generated_code": code,
            "generated_python": llm_resp.get("python", ""),
            "result": result,
            "image": image
        })
        return state
st.set_page_config(page_title="AI Data Agent", layout="wide")
st.title("🧠 Fully Autonomous Multi-Table AI Data Agent with Auto-Error-Healing")
uploaded_files = st.file_uploader("Upload CSVs (optional):", type="csv", accept_multiple_files=True)
upload_to_db = st.checkbox("📥 Upload CSVs into PostgreSQL", value=False)
user_query = st.text_input("Ask a question:", placeholder="E.g., Forecast next 30 days for Line A-14")
uploaded_dataframes = {}
engine = create_engine("postgresql://postgres:root@localhost:5432/ai_agent_db")
if uploaded_files:
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            name = os.path.splitext(f.name)[0]
            uploaded_dataframes[name] = df
            if upload_to_db:
                load_csv_to_postgres(df, name, engine)
        except Exception as e:
            st.error(f"❌ Error loading {f.name}: {e}")
schema_info = load_all_tables()
for name, df in uploaded_dataframes.items():
    name = f"{name}_csv" if name in schema_info else name
    schema_info[name] = {
        "columns": [(c, str(t)) for c, t in zip(df.columns, df.dtypes)],
        "sample_rows": df.head(3).to_dict(orient='records')
    }
schema_context = build_schema_context(schema_info)
if st.button("Run Agent") and user_query:
    with st.spinner("Thinking..."):
        agent_node = AgentNode(schema_info, schema_context)
        graph = StateGraph(dict)
        graph.add_node("agent", agent_node.run)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        final = graph.compile().invoke({"user_query": user_query})
        st.subheader("✅ SQL")
        st.code(final.get("generated_sql", ""), language="sql")
        st.subheader("✅ Python")
        st.code(final.get("generated_code", ""), language="python")
        st.subheader("✅ Result")
        r = final.get("result")
        img = final.get("image")
        if isinstance(r, pd.DataFrame):
            st.dataframe(r)
        elif isinstance(r, str):
            try:
                if r.startswith("iVBORw0KGgo"):
                    st.image(Image.open(io.BytesIO(base64.b64decode(r))))
                else:
                    st.text(r)
            except Exception as e:
                st.error(f"Could not render output: {e}")
        elif img:
            st.image(img)
        elif isinstance(r, dict):
            st.json(r)
        elif isinstance(r, (list, set, tuple)):
            st.table(pd.DataFrame({'Result': list(r)}))
        else:
            st.write(r)
