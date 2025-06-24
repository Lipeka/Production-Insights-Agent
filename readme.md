
# 🏭 AI Production Agent

An AI-powered production insights assistant that:
- ✅ Accepts multiple CSV uploads and stores them in PostgreSQL
- 🧠 Understands natural language queries using OpenRouter (Mistral)
- 📊 Detects the intent through user's natural language query and performs forecasting, comparisons, or zero-production analysis
- 📦 Visualizes results with Matplotlib + Prophet
- 🚀 Fully dynamic: no hardcoded column names or formats

---

## 📦 Features

| Feature                     | Description                                                              |
|----------------------------|--------------------------------------------------------------------------|
| Upload multiple CSVs       | CSVs are inserted into PostgreSQL with sanitized table names             |
| Auto schema extraction     | Detects columns dynamically using PostgreSQL schema                      |
| LLM-powered query parsing  | Uses OpenRouter’s `mistral-7b-instruct` to detect task + table + filters |
| Forecasting                | Forecasts future values using Facebook Prophet                           |
| Comparison                 | Compares planned vs actual (auto-detects columns if not specified)       |
| Zero-production detection  | Counts zero-entries in the selected target column                        |
| Visual & textual output    | Generates insights + graphs using Gradio                                 |

---

## 🛠️ Installation

### 1. Clone this repo

```bash
git clone https://github.com/your-username/ai-production-agent.git
cd ai-production-agent
````

### 2. Set up virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install gradio pandas numpy matplotlib prophet sqlalchemy openai psycopg2-binary
```

---

## 🧠 OpenRouter Setup

1. Get your OpenRouter API key from [https://openrouter.ai/](https://openrouter.ai/)
2. Replace the `api_key` value in this line in `prod.py`:

```python
client = OpenAI(
    api_key="sk-or-...",
    base_url="https://openrouter.ai/api/v1"
)
```

---

## 🐘 PostgreSQL Setup

1. Install PostgreSQL locally and create a DB named `prod_insights`.
2. Use the default user (`postgres`) and password (`root`) or update the connection string in:

```python
def get_engine():
    return create_engine("postgresql://postgres:root@localhost:5432/prod_insights")
```

---

## 🚀 Running the App

```bash
python prod.py
```

You’ll get a **Gradio interface** link like:

```
Running on local URL:  http://127.0.0.1:7860
```

---

## 🧪 Example Queries

* `Forecast production for next 60 days from garments table`
* `Compare planned vs actual in monthly_data table`
* `Find zero production entries from shift_analysis`
* `Forecast Line A-12 from table garments for 30 days`
* `How many zero entries for Actual in March from quality_check`

---

## 📁 File Structure

```
├── prod.py              # Main application code
├── README.md            # This file
```

---

## 🧠 Behind the Scenes

* **LLM Prompting**: Your query is converted into structured JSON like:

```json
{
  "task": "forecast",
  "table": "garments",
  "date_col": "shift_date",
  "target_col": "actual",
  "filters": { "line": "A-12" },
  "period_days": 30
}
```

* **No hardcoded logic**: Dates and filters are dynamically detected using column names + values.
* **Robust auto-fixes**: Handles invalid LLM outputs with auto-fix attempts on the returned JSON.

---

## ✅ To Do

* [ ] Add SHAP feature importance
* [ ] Add multi-table analysis
* [ ] Export to PDF/Excel
* [ ] Deploy on Hugging Face Spaces / Streamlit Cloud

---

## 📝 License

MIT License. Free to use and modify.
