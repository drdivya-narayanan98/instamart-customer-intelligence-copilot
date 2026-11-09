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

The implementation uses a lexical similarity threshold as a simple retrieval safeguard. It is not a calibrated semantic-confidence score.

### Retrieval evaluation

The retrieval system was evaluated using 14 questions: 12 answerable policy questions and 2 unanswerable questions.

| Evaluation metric              | 10-word overlap | Zero overlap   |
|--------------------------------|-----------------|----------------|
| Expected document retrieved    | 12/12 (100%)    | 12/12 (100%)   |
| Expected fact retrieved        | 12/12 (100%)    | 11/12 (91.67%) |
| Correct unanswerable fallbacks | 2/2 (100%)      | 2/2 (100%)     |

The 10-word overlap configuration retrieved all expected facts. Removing overlap reduced expected-fact retrieval by one question, demonstrating the benefit of overlapping chunks for this dataset.

### Support agent tools

The agent uses five tools:

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

### Checkpoint and recovery

The agent saves:
- Run ID
- Scenario ID
- Full conversation messages
- Token usage
- Step number
- Executed tools

An S1 crash was simulated after the second tool execution. The agent resumed from its checkpoint without repeating the completed tools.

### Scenario results

| Scenario | Expected outcome | Actual outcome |
|----------|------------------|----------------|
| S1       | Refund ₹128.25   | Refund ₹128.25 |
| S2       | Refund ₹135      | Refund ₹135    |
| S3       | Escalate ₹1,999  | Escalated      |
| S4       | Denied           | Denied         |
| S5       | Denied           | Denied         |
| S6       | Not found        | Not found      |
| S7       | Fallback         | Fallback       |

### Safety tests

- Token-limit test: `TOKEN_LIMIT`
- Timeout test: `TIMEOUT`
- Fallback iteration limit: 5

The token limit is checked after model calls, so a call can cause usage to exceed the configured threshold. It is a soft limit in this implementation.

### ReAct versus Plan-and-Execute

The agent uses a ReAct-style workflow, selecting the next tool based on the latest tool result.

Measured model calls:
- S1: 5
- S5: 3
- S6: 2

See `outputs/agent_architecture_comparison.txt` for the comparison.

Run the scripts from the project root with the virtual environment activated.

**Note:** Some scripts regenerate data or append to JSONL output files. For a fresh complete run, use a clean copy of the project or back up and clear generated outputs and database files before rerunning the pipeline.

## Limitations and Next Steps

- The dataset is simulated and does not represent actual Instamart customer records.
- The LLM is a supplied mock implementation rather than a live production model.
- The retrieval system uses TF-IDF rather than a neural embedding model.
- Model performance depends on the supplied synthetic dataset.
- The agent is a demonstration, not a production payment or refund system.
- Human approval is represented by a simulated escalation record.

This project demonstrates an end-to-end workflow connecting data engineering, SQL analytics, customer segmentation, predictive modelling, privacy-aware support ticket classification and a tool-using RAG support agent.

It combines analytical insights with automated customer support workflows while incorporating validation, fallback handling, audit logging and recovery mechanisms.