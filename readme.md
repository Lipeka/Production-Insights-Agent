# 🏭 AI Production Agent

An LLM-orchestrated autonomous production data agent that:

* ✅ Accepts multiple CSV uploads and ingests them into PostgreSQL
* 🧠 Understands free-form natural language queries with no task labels, keywords, or JSON parsing
* 🔎 Automatically detects task type (forecast, compare, zero production, summary) through LLM reasoning
* 📊 Visualizes outputs via bar plots, line graphs, and textual summaries with complete autonomy
* 🚫 Has **zero hardcoded logic**, column names, task labels, or tool references
* ♻️ Includes auto-error healing, retry mechanisms, dynamic schema adaptation, and multistep execution support

---

## 📦 Features

| Feature                          | Description                                                               |
| -------------------------------- | ------------------------------------------------------------------------- |
| **Zero-Hardcoded Logic**         | No fixed task keywords, column names, forecast tools, or control flags    |
| Dynamic multi-table support      | Handles multiple PostgreSQL and CSV tables in a single query              |
| Intent recognition via LLM       | No JSON returned — intent and column inference done via free-text parsing |
| Forecasting (via Prophet)        | Predicts future production using LLM-generated Python with Prophet        |
| Comparison (Bar Plot)            | Automatically generates grouped comparisons via bar charts                |
| Zero production detection        | Returns rows or jobs where output/actuals are zero                        |
| Auto-summary generation          | Full statistical + textual insights from any uploaded or existing table   |
| Multi-step agent logic           | Orchestrated with LangGraph to support retries, multi-code-block routing  |
| Natural language horizon parsing | Understands phrases like “next month”, “next 60 days”, “30 working days”  |
| Group-aware filtering            | Dynamically filters by lines, jobs, shifts, dates — no config needed      |
| Dynamic SQL + Python blocks      | Generated entirely by LLM based on query + live schema                    |
| SQL-first strict enforcement     | Data must always be queried via SQL; no use of `read_csv` or hardcoded df |
| Date auto-format repair          | Handles formats like `dd-mm-yyyy`, `mm-dd-yyyy`, and ISO intelligently    |
| Error recovery via auto-healing  | Retries failed Python blocks up to 6 times with self-repair heuristics    |
| Dynamic result rendering         | Auto-detects base64 images, text, or dataframes for Streamlit display     |
| Streamlit UI                     | Simple upload + query interface, optimized for debugging and analysis     |
| Multi-table comparison           | Compare tables even if they come from different schemas or file sources   |

---

## 🔥 Unique Innovations

| Innovation Area                  | Your Implementation                                                       |
| -------------------------------- | ------------------------------------------------------------------------- |
| **Prompt-Only Training**         | Agent is **strictly trained via prompt engineering only**, no finetuning  |
| **Tool-less routing**            | No need to invoke predefined tools like `forecast()`, `compare()`, etc.   |
| **No JSON parsing**              | LLM responses are free-form prose and parsed semantically by LangGraph    |
| **Fully dynamic schema binding** | Works with *any* table and column combo without user config               |
| **Auto-fix logic**               | Python failures auto-corrected and retried using dynamic simulation       |
| **Multimodal fallback**          | Handles partial answers (e.g., if Python is `pass`, fallback to SQL only) |

---

## 🛠️ Installation

### 1. Clone this repo

```bash
git clone https://github.com/your-username/ai-production-agent.git
cd ai-production-agent
```

### 2. Set up virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas numpy matplotlib prophet sqlalchemy openai psycopg2-binary
```

---

## 🧠 OpenRouter Setup

1. Get your API key from [https://openrouter.ai/](https://openrouter.ai/)
2. Add this to your `.env` or replace directly in `prod.py`:

```python
client = OpenAI(
    api_key="sk-or-...",
    base_url="https://openrouter.ai/api/v1"
)
```

---

## 🐘 PostgreSQL Setup

1. Install PostgreSQL and create a database:

```sql
CREATE DATABASE prod_insights;
```

2. Update DB credentials in the code or use `.env`:

```python
postgresql://postgres:root@localhost:5432/prod_insights
```

---

## 🚀 Run the App

```bash
streamlit run prod.py
```

---

## 💡 Example Queries

| Task Type       | Example Query                                                   |
| --------------- | --------------------------------------------------------------- |
| Forecast        | `Forecast next 60 days for Line A-7 from daily_output table`    |
| Comparison      | `Compare production in table_a and table_b by date`             |
| Zero Detection  | `Show when Line A-12 had zero Actual in March`                  |
| Summary         | `Summarize the table garments_upload.csv`                       |
| Grouped Compare | `Compare actuals for Line A-14 vs A-10 in a bar chart`          |
| Multivariate    | `Forecast Actual and Planned for next 30 days from shift_table` |

---

## 🧠 How It Works

Your natural language query is:

* Interpreted via LLM (Qwen 2.5 72B on OpenRouter)
* No predefined JSON plan returned
* Instead, the LLM replies with full natural language explanations and:

  * ✅ Dynamic SQL block (always first)
  * ✅ Dynamic Python block (must reference `df` created from SQL)
* LangGraph executes SQL → Python in order
* If Python fails, it:

  * Triggers auto-repair heuristics (e.g., fix undefined df, remove simulation)
  * Retries up to 6 times

---

## ✅ To Do

* [ ] Add LLM-driven PDF summary export
* [ ] Add SHAP-style feature attribution (for forecasting)
* [ ] Hugging Face / Streamlit Cloud deployment
* [ ] CSV preview and chart download options

---

## 📝 License

MIT License — Use freely and modify.
