# INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT 

An end-to-end data analytics, machine learning and AI-powered customer support project built using Python, SQL, Pandas, Scikit-learn, TF-IDF and a simulated LLM.

## Overview

This project develops a customer intelligence and support system for a simulated Instamart business.

It covers four components:

- **Part A – Data Engineering and Analytics:** Data quality auditing, cleaning, SQLite database creation, SQL analysis and Pandas EDA.
- **Part B – Customer Intelligence:** RFM segmentation, K-Means clustering and customer lapse prediction.
- **Part C – Support Ticket Intelligence:** PII redaction, ticket classification, prioritisation, routing, retries and token-budget controls.
- **Part D – RAG and AI Support Agent:** Policy retrieval, tool execution, refund decisions, human escalation, checkpointing and agent recovery.

- Python 3.11
- Pandas and NumPy
- SQLite and SQL
- Scikit-learn
- TF-IDF and cosine similarity
- K-Means clustering
- Logistic Regression and Random Forest
- Joblib
- JSON and JSONL
- Mock LLM supplied with the project

No external LLM API key or network connection is required.

```text
instamart-customer-copilot/
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
│
├── outputs/
│   ├── part_a_summary.csv
│   ├── customer_segments.csv
│   ├── lapse_model.joblib
│   ├── retention_list.csv
│   ├── triage_results.csv
│   ├── triage_budget_demo.csv
│   ├── agent_runs.jsonl
│   ├── refund_ledger.jsonl
│   └── agent_architecture_comparison.txt
│
├── sql/
│   └── queries.sql
│
├── state/
│
├── audit_raw.py
├── audit_quality.py
├── clean_data.py
├── build_database.py
├── validate_data.py
├── pandas_analysis.py
├── customer_segmentation.py
├── lapse_prediction.py
├── ticket_triage.py
├── policy_rag.py
├── support_agent.py
├── generate_data.py
├── mock_llm.py
├── requirements.txt
└── README.md
```

## Reproduce

### 1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Generate the input data

```bash
python generate_data.py
```

The simulated dataset contains customers, products, orders, support tickets, policy evaluation records and agent scenarios.

### 4. Run Part A – Data Engineering and Analytics

```bash
python audit_raw.py
python audit_quality.py
python clean_data.py
python build_database.py
python validate_data.py
python pandas_analysis.py
```

### 5. Run Part B – Customer Intelligence

```bash
python customer_segmentation.py
python lapse_prediction.py
```

### 6. Run Part C – Support Ticket Intelligence

```bash
python ticket_triage.py
```

### 7. Run Part D – RAG and Support Agent

```bash
python policy_rag.py
python support_agent.py
```

The project uses a simulated LLM and does not require an external API key.

## Part A 

### Data quality auditing

Scripts:
- `audit_raw.py`
- `audit_quality.py`

The raw data was inspected for missing values, duplicates, invalid values, inconsistent categories and formatting issues.

### Data cleaning

Script: `clean_data.py`

Key operations:
- Standardised city and gender values.
- Handled missing and invalid customer ages.
- Removed duplicate orders.
- Retained the latest record for each order ID.
- Repaired missing order amounts using product price, quantity and discount.
- Handled invalid delivery times.

### SQLite database

Script: `build_database.py`

Created a relational database with three tables:

|   Table   | Records |
|-----------|---------|
| Customers |  1,200  |
| Products  |    40   |
| Orders    |  12,309 |

### Validation

Script: `validate_data.py`

All 12 data validation checkpoints passed.

### SQL and Pandas analysis

- Seven SQL analytical queries.
- Joins, aggregations and HAVING conditions.
- Ranking and window functions.
- Daily revenue and cumulative revenue.
- Pivot tables and grouped analysis.
- Distribution statistics and correlation analysis.

Total cleaned completed-order revenue: ₹29,98,668.80.

## Part B 

### RFM customer segmentation

Script: `customer_segmentation.py`

Features:
- Recency
- Frequency
- Monetary value

Applied StandardScaler and K-Means clustering.

Compared K values from 2 to 8 and selected K=4 based on silhouette score and interpretability.

The four business segments:

1. Regular
2. High Value
3. Loyal
4. At Risk

PCA was used for dimensionality reduction and visualisation analysis.

### Customer lapse prediction

Script: `lapse_prediction.py`

Compared:
- Dummy Classifier baseline
- Logistic Regression
- Random Forest
- Oversampled Logistic Regression

The Random Forest model achieved:

- Accuracy: 70.3%
- Lapsed-customer precision: 0.60
- Lapsed-customer recall: 0.66
- Lapsed-customer F1-score: 0.63

The oversampled Logistic Regression model achieved a lapsed-customer F1-score of 0.66 and recall of 0.79, with more false positives.

The selected Random Forest model was saved using Joblib.

### Retention list

Generated `outputs/retention_list.csv` containing high-priority customers for retention analysis.

**Important limitation:** The premium segment labels used for this illustrative list were created using full-period RFM data, including September. Therefore, this list should not be treated as a leakage-free prospective evaluation.

## Part C

Script: `ticket_triage.py`

### Implementation

- Empty-ticket validation and long-ticket detection before model calls.
- Email, phone and order ID PII redaction.
- Zero-shot and few-shot classification across billing, delivery, technical and fraud.
- JSON validation and keyword-based fallback validation.
- Exponential backoff with jitter for transient errors.
- Customer-segment-aware routing and priority assignment.
- Actual token usage tracking and token-budget enforcement.
- CSV outputs for classification and budget demonstrations.

### Input validation and PII redaction

|            Metric           | Result |
|-----------------------------|--------|
| Total tickets               | 164    |
| Empty tickets               | 4      |
| Long tickets                | 4      |
| Email replacements          | 55     |
| Phone replacements          | 91     |
| Order ID replacements       | 132    |
| Remaining email patterns    | 0      |
| Remaining phone patterns    | 0      |
| Remaining order ID patterns | 0      |

All detected PII patterns were removed before classification.

### Classification results

| Metric                           | Zero-shot | Few-shot |
|----------------------------------|-----------|----------|
| Valid JSON classifications       | 135       | 135      |
| Keyword fallback classifications | 12        | 12       |
| Total classified tickets         | 147       | 147      |
| Invalid model outputs            | 9         | 9        |
| Accuracy                         | 85.71%    | 100.00%  |
| Actual tokens used               | 9,712     | 18,292   |
| Tokens per classified ticket     | 66.07     | 124.44   |

Few-shot prompting improved accuracy by 14.29 percentage points and used 8,580 additional tokens.

### Confusion matrices

Rows represent actual categories; columns represent predicted categories.

**Zero-shot**

| Actual / Predicted | Billing  | Delivery | Technical | Fraud |
|--------------------|----------|----------|-----------|-------|
| Billing            | 38       | 0        | 0         | 0     |
| Delivery           | 0        | 36       | 0         | 0     |
| Technical          | 10       | 0        | 29        | 0     |
| Fraud              | 11       | 0        | 0         | 23    |

**Few-shot**

| Actual / Predicted | Billing | Delivery | Technical | Fraud |
|--------------------|---------|----------|-----------|-------|
| Billing            | 38      | 0        | 0         | 0     |
| Delivery           | 0       | 36       | 0         | 0     |
| Technical          | 0       | 0        | 39        | 0     |
| Fraud              | 0       | 0        | 0         | 34    |

The zero-shot errors were 10 technical tickets and 11 fraud tickets incorrectly classified as billing. Few-shot classification correctly classified all 147 evaluated tickets.

### Retry handling

Transient errors were recovered using exponential backoff with jitter. The supplied mock LLM generated 19 transient errors per full classification run; all were recovered. The retry log records the ticket ID, retry number and wait duration.

### Token-budget demonstration

The zero-shot budget demonstration used a 6,000-token budget.

|            Status              | Tickets |
|--------------------------------|---------|
| Classified JSON                | 85      |
| Classified by keyword fallback | 7       |
| Invalid model output           | 5       |
| Budget exhausted               | 59      |
| Empty tickets skipped          | 4       |
| Long tickets flagged           | 4       |
| Actual tokens consumed         | 6,049   |

The budget is checked before each model call. A final call can take the running total slightly above the configured budget; subsequent calls are stopped.

### Routing results

|     Route     | Tickets |
|---------------|---------|
| Support queue | 102     |
| Security team | 34      |
| Human review  | 17      |
| Premium desk  | 11      |

| Priority | Tickets |
|----------|---------|
| NORMAL   | 119     |
| HIGH     | 45      |

Outputs: `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.

These results are from the supplied simulated evaluation dataset, not a real-world production benchmark.

## Part D

Scripts:
- `policy_rag.py`
- `support_agent.py`

### Retrieval-Augmented Generation

Five policy documents were loaded and divided into 40-word chunks with 10-word overlap.

- Total overlapping chunks: 15
- TF-IDF vectorisation
- Cosine similarity retrieval
- Top-3 policy passage retrieval
- Fallback when no relevant policy match is found

The implementation uses a lexical similarity threshold as a retrieval safeguard. It is not a calibrated semantic-confidence score.

### Retrieval evaluation

The retrieval system was evaluated using 14 questions: 12 answerable policy questions and 2 unanswerable questions.

| Evaluation metric | 10-word overlap | Zero overlap |
|---|---:|---:|
| Expected document retrieved | 12/12 (100%) | 12/12 (100%) |
| Expected fact retrieved | 12/12 (100%) | 11/12 (91.67%) |
| Correct unanswerable fallbacks | 2/2 (100%) | 2/2 (100%) |

The 10-word overlap configuration retrieved all expected facts. Removing overlap reduced expected-fact retrieval by one question, demonstrating the benefit of overlapping chunks for this dataset.

### Support agent tools

The agent defines five tools in `TOOLS_SPEC` and maps them to their implementations through `TOOL_MAP`:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`
5. `escalate_to_human`

### Refund business rules

- Only completed orders are eligible.
- Delivery taking more than 45 minutes qualifies for a full refund.
- Delivery taking 45 minutes or less receives no refund.
- Refunds above ₹1,000 are escalated for human approval.

### Agent execution and checkpoint recovery

The agent uses a ReAct-style tool execution loop.

The `run_agent` interface supports:
- `user_message`
- `scenario_id`
- `max_iterations` (default: 6)
- `token_limit` (default: 6,000)
- `time_limit_seconds` (default: 10)

The agent records its conversation messages, token usage, step number and executed tool steps. Each tool step stores the tool name, arguments and observation.

An S1 crash was simulated after the second tool execution. The agent resumed from its checkpoint without repeating completed tools. The recovered workflow executed `get_order`, `search_policy`, `calculate_refund` and `issue_refund` exactly once each.

### Scenario results

Results from the latest complete seven-scenario run:

| Scenario | Actual status | Outcome | Model calls | Tokens |
|---|---|---|---:|---:|
| S1 | REFUND | ₹128.25 refund issued | 5 | 1,460 |
| S2 | REFUND | ₹135 refund issued | 5 | 1,460 |
| S3 | ESCALATE | ₹1,999 sent for human approval | 5 | 1,447 |
| S4 | DENIED | Refund denied | 4 | 1,055 |
| S5 | DENIED | Refund denied | 3 | 674 |
| S6 | NOT_FOUND | Order not found | 2 | 291 |
| S7 | FALLBACK_MAX_ITERATIONS | Safe fallback after six iterations | 6 | 3,666 |

The run results are appended to `outputs/agent_runs.jsonl`. Each record includes the scenario ID, status, final answer, model-call count, token usage and detailed tool steps with arguments and observations.

### Safety and fallback tests

The agent includes three execution safeguards:

- Maximum iteration limit: 6.
- Token budget: checked before and after model calls. If the budget is exceeded after a call, execution stops before the next tool is executed.
- Time limit: checked using a monotonic clock to avoid wall-clock adjustments affecting elapsed-time measurement.

Additional tests were run using S1:

| Test | Test configuration | Actual result |
|---|---|---|
| Token-limit fallback | `token_limit=200` | `FALLBACK_TOKEN_LIMIT` |
| Timeout fallback | `time_limit_seconds=0` | `FALLBACK_TIMEOUT` |
| Iteration fallback | Repeated tool-call scenario | `FALLBACK_MAX_ITERATIONS` |

Both the token-limit and timeout tests returned safe messages directing the customer to human support.

### ReAct versus Plan-and-Execute

ReAct (Reason + Act) combines reasoning and tool execution in an iterative loop. The agent examines the latest tool result and decides what action to take next.

Plan-and-Execute separates planning from execution. It first creates a sequence of steps and then executes that plan, potentially reducing repeated planning decisions.

This project uses a ReAct-style workflow. The mock agent chooses tools based on the latest tool result.

Measured results:

| Scenario | Model calls | Tokens |
|---|---:|---:|
| S1 | 5 | 1,460 |
| S5 | 3 | 674 |
| S6 | 2 | 291 |

S1 required order lookup, policy search, refund calculation, refund issuance and a final response. S5 stopped after identifying a cancelled order. S6 stopped after the order lookup returned NOT_FOUND.

ReAct handles unexpected tool results naturally. For example, a missing order stops processing, while a cancelled order is denied before refund calculation. However, iterative reasoning can consume more calls and may loop. The six-iteration limit prevents indefinite execution.

Plan-and-Execute may be more efficient for predictable, multi-step workflows, while ReAct is useful when the next action depends on intermediate tool results.

See `outputs/agent_architecture_comparison.txt` for the architecture comparison.

## Limitations and Next Steps

- The dataset is simulated and does not represent actual Instamart customer records.
- The LLM is a supplied mock implementation rather than a live production model.
- The retrieval system uses TF-IDF rather than a neural embedding model.
- Model performance depends on the supplied synthetic dataset.
- The agent is a demonstration, not a production payment or refund system.
- Human approval is represented by a simulated escalation record.

This project demonstrates an end-to-end workflow connecting data engineering, SQL analytics, customer segmentation, predictive modelling, privacy-aware support ticket classification and a tool-using RAG support agent.

It combines analytical insights with automated customer support workflows while incorporating validation, fallback handling, audit logging and recovery mechanisms.