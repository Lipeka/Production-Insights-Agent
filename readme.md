
# 🏭 AI-Powered Production Insights Agent

Welcome to the **Production Insights AI Agent** — your personal AI data analyst for manufacturing operations! This powerful app allows you to explore, forecast, visualize, and analyze production data using natural language queries. From real-time trends to explainable AI, it's built to deliver **data-backed decisions on demand**.

---

## 🚀 Features

🔍 **Natural Language Querying**
- Ask things like:
  - “What was the average production for A-4 last month?”
  - “Compare planned vs actual for June”
  - “Forecast next 30 days of production for Line A-2”

📊 **Planned vs Actual Production Visualization**
- Instantly renders side-by-side bar charts
- Helps identify gaps in execution vs expectations

🔮 **Time Series Forecasting**
- Powered by Facebook Prophet
- Predicts future production based on historical trends
- Supports dynamic time windows (e.g. 15 days, 2 months)

🧠 **MLOps and Explainable AI with MLflow + SHAP + XGBoost**
- Feature importance comparisons across lines
- Understand what drives your production outcomes

🧼 **Zero Production Detection**
- Flags time slots with actual = 0
- Great for downtime analysis and maintenance planning

🧑‍💻 **OpenRouter RAG (Retrieval-Augmented Generation)**
- Seamlessly answers complex queries using `Mistral-7B Instruct`
- Context-aware insights through language + data fusion

🧬 **Gradio UI Dashboard**
- Intuitive and interactive frontend
- No need to write code—just ask and get insights with visuals!

---

## 🛠 Tech Stack

| Layer            | Tech Used                              |
|------------------|----------------------------------------|
| 📦 Backend       | Python, Pandas, SQLAlchemy              |
| 🔮 ML & Forecast | Prophet, XGBoost, SHAP                  |
| 🎨 Visualization | Matplotlib, PIL                         |
| 🤖 AI Assistant  | OpenRouter + Mistral-7B (free tier)     |
| 🌐 Frontend      | Gradio                                 |
| 🧠 MLOps         | MLflow (tracking forecasts & metrics)   |
| 🗃️ Database       | PostgreSQL                             |
| 📄 Data Input    | CSV Loader via `loader.py`              |

---

## ⚙️ How It Works

1. **Load Data**
   - Drop your `.csv` files into the `csvs/` folder.
   - Run `python loader.py` to upload into PostgreSQL as `production_data`.

2. **Start the Agent**
   ```bash
   python agent.py
````

3. **Ask Anything**

   * Use the Gradio interface to ask production-related queries in plain English.
   * Get answers, insights, forecasts, and SHAP visualizations!

---

## 🔐 OpenRouter Setup

This app uses [OpenRouter.ai](https://openrouter.ai) to run the Mistral-7B model for free!

Set your OpenRouter API key in `agent.py`:

```python
client = OpenAI(
    api_key="sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://openrouter.ai/api/v1"
)
```

---

## 💬 Example Queries

* "average production for A-2 in April"
* "forecast next 15 days for A-4"
* "compare planned vs actual for job 301"
* "show shap for A-2 and A-3"
* "zero production in May 2024"

---

## 📜 License

This project is licensed under the MIT License.

```
