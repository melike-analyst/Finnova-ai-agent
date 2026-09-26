# FinNova Bank — Autonomous Business Intelligence Agent

An end-to-end system that uses an LLM-based, multi-step **AI agent** to translate
natural-language finance/banking questions into SQL queries, analyze the
results, interpret them, and automatically send weekly reports.

> **Why this project?** Most "AI  projects" are one-off chatbot demos.
> This project instead demonstrates three things at once: (1) **agent
> orchestration** that combines an LLM with external tools (SQL, charts),
> (2) wiring that into a real workflow through **automation** (weekly report →
> Slack), and (3) a **deployed**, running product — not just a Jupyter
> notebook.

---

## 1) Architecture

```
User question / scheduled trigger
            │
            ▼
   Agent orchestrator (Google Gemini + native function-calling)
            │
            ├──▶ run_sql   → SQLite database (SELECT-only, safe)
            └──▶ make_chart → automatic chart generation
            │
            ▼
   Natural-language interpretation + chart (if applicable)
            │
            ├──▶ Streamlit UI (live demo)
            └──▶ Weekly automation → Slack / email
```

The agent loop is framework-free, written directly against Google Gemini's
native function-calling API (`src/agent.py`). This is a deliberate choice:
libraries like LangChain/LangGraph handle this logic "inside a black box";
here the logic is kept transparent to show how the agent concept actually
works under the hood. If you'd rather use a framework, the same flow can be
rebuilt with LangGraph's `StateGraph` (see §7, "Extension ideas").

**LLM provider note:** The project uses the Google Gemini API
(`gemini-3.5-flash`) because Gemini offers a genuine free tier with no
credit card required — a practical choice for portfolio/demo projects. The
same agent architecture can be ported to Anthropic Claude or the OpenAI API
with minimal changes; the tool-calling logic in `src/agent.py` was designed
to be provider-agnostic.

## 2) Dataset

The data is **entirely synthetic**, generated for a fictional "FinNova Bank"
(`data/generate_dataset.py`, using Faker + numpy). It contains no real
individual or institutional data. Note that this is a deliberate choice:
real financial data isn't suitable for portfolio projects due to data
protection (KVKK/GDPR) and licensing constraints; synthetic data both
removes that problem and gives full freedom to design the business
scenarios you want on top of it (seasonality, regional growth, risk
signals).

| Table | Row count | Description |
|---|---|---|
| `branches` | 20 | Branch info across 8 cities |
| `customers` | 900 | Customer demographics, credit score, segment |
| `accounts` | 1,470 | Account type, balance, interest rate, loan amount |
| `transactions` | 75,616 | 24 months of transaction history |

Realistic business patterns are deliberately embedded in the data (the
agent is expected to "discover" them):

- Flagged (suspicious) transaction rate is **4.51%** for credit card
  purchases vs. an average of **~1%** for other transaction types
  (verified)
- Bill-payment rate is **38.5%** in winter months (Dec–Feb) vs. **12.7%**
  in other seasons (verified)
- A noticeable increase in average monthly transaction volume at the
  Istanbul and Izmir branches over the last 6 months (verified)

## 3) Setup

```bash
git clone <repo-url>
cd finnova-ai-agent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate the dataset and load it into the database
python data/generate_dataset.py
python data/load_to_sqlite.py

# Prepare your .env file
cp .env.example .env
# Enter your own GEMINI_API_KEY in .env (free, no credit card required)
# (https://aistudio.google.com/apikey)
```

## 4) Running the project

**To ask a single question from the terminal:**
```bash
python -m src.agent "Which transaction type has the highest suspicious-activity rate?"
```

**Streamlit UI (live demo):**
```bash
streamlit run app/app.py
```

**To test the weekly automation (one-off run):**
```bash
python automation/scheduler.py --once
```

**To run the weekly automation as a persistent service:**
```bash
python automation/scheduler.py
```

## 5) Example usage

**Question:** *"Which transaction type has the highest suspicious-activity rate?"*

The agent's steps:
1. Calls the `run_sql` tool → generates SQL that computes the flagged rate
   by `transaction_type`
2. Gets the result (Credit Card Purchase: 4.51% — about 4.5× higher than
   the rest)
3. Interprets it in natural language: *"Credit card purchases show a
   noticeably higher suspicious-activity rate compared to other
   transaction types. This suggests card-based fraud controls should be
   prioritized."*

## 5.1) Classical machine learning component: fraud prediction model

While the main project is an LLM-based agent,
`notebooks/fraud_detection_model.ipynb` showcases the project's **classical
ML** side: EDA, feature engineering, model training (Logistic Regression +
Random Forest), and evaluation (precision/recall/F1/ROC-AUC/feature
importance), presented as an end-to-end pipeline.

**Results** (reported honestly): Random Forest ROC-AUC ~0.66, Logistic
Regression ~0.63. The modest performance is because the fraud label in the
synthetic dataset was assigned largely at random, conditioned on
transaction type (see the "Results and Discussion" section in the notebook
for the full write-up). This is a deliberate transparency choice:
interpreting a model's limitations honestly, rather than overstating
results, is what real data-science practice looks like.

To run it:
```bash
cd notebooks
jupyter notebook fraud_detection_model.ipynb
```

## 6) Security notes

- `src/tools/sql_tool.py` validates the LLM-generated SQL before executing
  it: only `SELECT` statements are allowed; `INSERT/UPDATE/DELETE/DROP` and
  multiple statements (`;`) are rejected.
- The `.env` file is in `.gitignore`: your API key never gets committed to
  the repo.
- Results are capped at 200 rows (to avoid bloating the agent's context
  window and to control potential cost increases).

## 7) Extension ideas (next steps)

- **Add a RAG layer:** integrate a vector database (Chroma) that can also
  query company policy/procedure PDFs, so the agent can answer questions
  like "Is this customer eligible under our credit approval policy?"
- **Migrate to LangGraph:** remodel the loop in `src/agent.py` as a
  `StateGraph`; extend it into a multi-agent architecture (planner +
  analyst + writer), which could also be done with CrewAI.
- **Evaluation (eval) set:** build a test set of 20-30 question/expected-SQL
  pairs and wire a "correct SQL generation rate" metric into CI/CD (GitHub
  Actions).
- **n8n integration:** move what `automation/scheduler.py` does into a
  no-code n8n workflow, and write to Notion/Google Sheets in addition to
  Slack.

## 8) Project structure

```
finnova-ai-agent/
├── data/
│   ├── generate_dataset.py   # synthetic data generation
│   ├── load_to_sqlite.py     # CSV → SQLite
│   └── finnova_bank.db
├── src/
│   ├── db.py                 # connection + schema introspection
│   ├── agent.py               # agent orchestration loop
│   └── tools/
│       ├── sql_tool.py        # safe SQL execution
│       └── chart_tool.py      # automatic chart generation
├── app/
│   └── app.py                 # Streamlit UI
├── automation/
│   └── scheduler.py           # weekly automated report
├── requirements.txt
├── .env.example
└── README.md
```

## 9) Tech stack

`Python` · `Google Gemini API (function-calling / agentic loop)` · `SQLite` ·
`pandas` · `Streamlit` · `APScheduler` · `Slack Webhooks` · (optional
extensions: `LangGraph`, `Chroma`, `n8n`)

---

*Note: all data used is synthetic and bears no relation to any real bank or customer data.*
