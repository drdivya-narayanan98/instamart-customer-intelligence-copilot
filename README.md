# INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT 

An end-to-end data analytics, machine learning and AI-powered customer support project built using Python, SQL, Pandas, Scikit-learn, TF-IDF and a simulated LLM.

## Project Overview

This project develops a customer intelligence and support system for a simulated Instamart business.

It covers four components:

- **Part A – Data Engineering and Analytics:** Data quality auditing, cleaning, SQLite database creation, SQL analysis and Pandas EDA.
- **Part B – Customer Intelligence:** RFM segmentation, K-Means clustering and customer lapse prediction.
- **Part C – Support Ticket Intelligence:** PII redaction, ticket classification, prioritisation, routing, retries and token-budget controls.
- **Part D – RAG and AI Support Agent:** Policy retrieval, tool execution, refund decisions, human escalation, checkpointing and agent recovery.

## Technology Stack

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

## Project Structure

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

## Setup

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

## Part A – Data Engineering and Analytics

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

## Part B – Customer Intelligence

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

## Part C – Support Ticket Intelligence

Script: `ticket_triage.py`

Implemented:

- Empty-ticket validation.
- Long-ticket detection.
- Email, phone and order ID redaction.
- Zero-shot and few-shot classification.
- Retry handling for transient errors.
- Invalid-output handling.
- Customer segment-aware routing.
- Priority assignment.
- Token-budget enforcement.

### Results

|          Metric        | Result |
|------------------------|--------|
| Total tickets          | 164    |
| Classified tickets     | 135    |
| Empty tickets          | 4      |
| Long tickets           | 4      |
| Invalid model outputs  | 21     |
| Email replacements     | 55     |
| Phone replacements     | 91     |
| Order ID replacements  | 132    |
| Remaining PII patterns | 0      |

Few-shot classification accuracy: 100% on the supplied evaluation cases.

Zero-shot classification accuracy: 84.44%.

These results are from the supplied simulated evaluation dataset, not a real-world production benchmark.

## Part D – RAG and AI Support Agent

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

## Running the project

Run the scripts from the project root with the virtual environment activated.

```bash
python audit_raw.py
python audit_quality.py
python clean_data.py
python build_database.py
python validate_data.py
python pandas_analysis.py
python customer_segmentation.py
python lapse_prediction.py
python ticket_triage.py
python support_agent.py
```

**Note:** Some scripts regenerate data or append to JSONL output files. For a fresh complete run, use a clean copy of the project or back up and clear generated outputs and database files before rerunning the pipeline.

## Limitations

- The dataset is simulated and does not represent actual Instamart customer records.
- The LLM is a supplied mock implementation rather than a live production model.
- The retrieval system uses TF-IDF rather than a neural embedding model.
- Model performance depends on the supplied synthetic dataset.
- The agent is a demonstration, not a production payment or refund system.
- Human approval is represented by a simulated escalation record.

## Conclusion

This project demonstrates an end-to-end workflow connecting data engineering, SQL analytics, customer segmentation, predictive modelling, privacy-aware support ticket classification and a tool-using RAG support agent.

It combines analytical insights with automated customer support workflows while incorporating validation, fallback handling, audit logging and recovery mechanisms.