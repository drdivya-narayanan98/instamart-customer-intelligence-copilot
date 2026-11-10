import pandas as pd
import re
import json
import time
import random

from mock_llm import reset

from mock_llm import chat, est_tokens, TransientLLMError

# Load ticket data
tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

print("TICKETS LOADED")
print("Total tickets:", len(tickets))

# Identify empty and long tickets
tickets["text_length"] = tickets["text"].str.len()

tickets["is_empty"] = (
    tickets["text"].str.strip() == ""
)

tickets["estimated_tokens"] = (
    tickets["text"].apply(est_tokens)
)

tickets["flagged_long"] = (
    tickets["estimated_tokens"] > 400
)

print("\nINPUT VALIDATION")
print("Empty tickets:", tickets["is_empty"].sum())
print("Long tickets:", tickets["flagged_long"].sum())

# Redact personal information
def redact_pii(text):
    text = re.sub(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        "[EMAIL]",
        text
    )

    text = re.sub(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        text
    )

    text = re.sub(
        r"ORD-\d{6}",
        "[ORDER_ID]",
        text
    )

    return text

tickets["redacted_text"] = (
    tickets["text"].apply(redact_pii)
)

print("\nPII REDACTION")
print(
    "Email replacements:",
    tickets["text"].str.count(
        r"[\w.+-]+@[\w-]+\.[\w.]+"
    ).sum()
)

print(
    "Phone replacements:",
    tickets["text"].str.count(
        r"\b[6-9]\d{9}\b"
    ).sum()
)

print(
    "Order ID replacements:",
    tickets["text"].str.count(
        r"ORD-\d{6}"
    ).sum()
)

# Verify that no original PII patterns remain
print("\nPII VERIFICATION")

email_remaining = tickets["redacted_text"].str.contains(
    r"[\w.+-]+@[\w-]+\.[\w.]+",
    regex=True
).sum()

phone_remaining = tickets["redacted_text"].str.contains(
    r"\b[6-9]\d{9}\b",
    regex=True
).sum()

order_id_remaining = tickets["redacted_text"].str.contains(
    r"ORD-\d{6}",
    regex=True
).sum()

print("Emails remaining:", email_remaining)
print("Phone numbers remaining:", phone_remaining)
print("Order IDs remaining:", order_id_remaining)

print(
    "All PII removed:",
    email_remaining + phone_remaining + order_id_remaining == 0
)

assert (
    email_remaining + phone_remaining + order_id_remaining == 0
), "PII detected after redaction!"

# Zero-shot classification prompt
ZERO_SHOT_SYSTEM = """
CLASSIFY the customer support ticket.

Choose exactly one category:
billing, delivery, technical, fraud.

Return only valid JSON in this format:
{"category": "billing", "reason": "brief explanation"}

Do not include markdown or extra text.
"""

# Few-shot classification prompt
FEW_SHOT_SYSTEM = """
CLASSIFY the customer support ticket.

Choose exactly one category:
billing, delivery, technical, fraud.

EXAMPLES:

Ticket: I was charged twice for my order.
Output: {"category": "billing", "reason": "duplicate charge"}

Ticket: My delivery has not arrived.
Output: {"category": "delivery", "reason": "late delivery"}

Ticket: The app keeps crashing when I open it.
Output: {"category": "technical", "reason": "app issue"}

Ticket: Someone used my card without permission.
Output: {"category": "fraud", "reason": "unauthorized payment"}

Return only valid JSON.
Do not include markdown or extra text.
"""

# Build messages for the model
def build_messages(ticket_text, few_shot=False):

    system_prompt = (
        FEW_SHOT_SYSTEM if few_shot
        else ZERO_SHOT_SYSTEM
    )

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": ticket_text
        }
    ]

# Ticket classification prompts

ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return only valid JSON in this format:
{"category": "category_name", "reason": "short reason"}
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

EXAMPLES:
Ticket: I was charged twice for my order.
{"category": "billing", "reason": "duplicate charge"}

Ticket: My package has not arrived.
{"category": "delivery", "reason": "late delivery"}

Ticket: The app crashes when I open it.
{"category": "technical", "reason": "app issue"}

Ticket: Someone used my card without permission.
{"category": "fraud", "reason": "unauthorized payment"}

Return only valid JSON.
"""


def classify_ticket(text, prompt, ticket_id="TEST"):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]

    allowed_categories = {
        "billing", "delivery", "technical", "fraud"
    }

    for attempt in range(4):
        try:
            response = chat(messages)
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            try:
                result = json.loads(response_text)

                category = result.get("category")

                if category in allowed_categories:
                    return category, "classified_json", tokens_used

                return None, "flagged_invalid_output", tokens_used

            except (json.JSONDecodeError, AttributeError, TypeError):
                words = set(
                    re.findall(r"[a-z]+", response_text.lower())
                )

                matches = words.intersection(allowed_categories)

                if len(matches) == 1:
                    return matches.pop(), "classified_keyword", tokens_used

                return None, "flagged_invalid_output", tokens_used

        except TransientLLMError:
            if attempt == 3:
                return None, "flagged_invalid_output", 0

            base_delay = 0.05 * (2 ** attempt)

            delay = random.uniform(
                base_delay * 0.9,
                base_delay * 1.15
            )

            print(
                f"Retry: ticket {ticket_id}, retry number {attempt + 1}, "
                f"wait {delay:.4f} seconds"
            )

            time.sleep(delay)

    return None, "flagged_invalid_output", 0


# Test zero-shot classification on one ticket

test_text = tickets.loc[
    tickets["is_empty"] == False,
    "redacted_text"
].iloc[0]

test_result = classify_ticket(
    test_text,
    ZERO_SHOT_PROMPT
)

print("\nCLASSIFICATION TEST")
print("Ticket text:", test_text)
print("Predicted category:", test_result)

# Test few-shot classification on the same ticket

few_shot_result = classify_ticket(
    test_text,
    FEW_SHOT_PROMPT
)

print("\nFEW-SHOT CLASSIFICATION TEST")
print("Predicted category:", few_shot_result)


# Load customer segment information

segments = pd.read_csv(
    "outputs/customer_segments.csv"
)

premium_customers = set(
    segments.loc[
        segments["is_premium"] == True,
        "customer_id"
    ]
)

print("\nCUSTOMER SEGMENTS")
print("Total customers:", len(segments))
print("Premium customers:", len(premium_customers))


def process_tickets(prompt, token_budget=100000):
    reset()

    results = []
    total_tokens = 0

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        text = ticket["redacted_text"]

        category = None
        tokens_used = 0

        if ticket["is_empty"]:
            status = "skipped_empty"

        elif ticket["flagged_long"]:
            status = "flagged_long"

        else:
            estimated_cost = est_tokens(prompt) + est_tokens(text)

            if total_tokens >= token_budget:
                status = "budget_exhausted"

            else:
                category, status, tokens_used = classify_ticket(text, prompt, ticket_id)

                total_tokens += tokens_used

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "tokens_used": tokens_used
        })

    return pd.DataFrame(results), total_tokens


# Run zero-shot classification

zero_shot_results, zero_shot_tokens = process_tickets(
    ZERO_SHOT_PROMPT
)

print("\nZERO-SHOT RESULTS")
print(zero_shot_results["status"].value_counts())
print("Actual tokens used:", zero_shot_tokens)

# Calculate accuracy using successfully classified tickets
zero_shot_check = zero_shot_results.merge(
    tickets[["ticket_id", "true_category"]],
    on="ticket_id"
)

valid_zero_shot = zero_shot_check[
    zero_shot_check["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

accuracy = (
    valid_zero_shot["category"]
    == valid_zero_shot["true_category"]
).mean() * 100

print("Zero-shot accuracy:", round(accuracy, 2), "%")

zero_shot_classified_count = len(valid_zero_shot)

print(
    "Tokens per classified ticket:",
    round(zero_shot_tokens / zero_shot_classified_count, 2)
)

from sklearn.metrics import confusion_matrix

print("\nZERO-SHOT CONFUSION MATRIX")
print(
    pd.DataFrame(
        confusion_matrix(
            valid_zero_shot["true_category"],
            valid_zero_shot["category"],
            labels=["billing", "delivery", "technical", "fraud"]
        ),
        index=["Actual billing", "Actual delivery", "Actual technical", "Actual fraud"],
        columns=["Pred billing", "Pred delivery", "Pred technical", "Pred fraud"]
    )
)


# Run few-shot classification

few_shot_results, few_shot_tokens = process_tickets(
    FEW_SHOT_PROMPT
)

print("\nFEW-SHOT RESULTS")
print(few_shot_results["status"].value_counts())
print("Actual tokens used:", few_shot_tokens)

few_shot_check = few_shot_results.merge(
    tickets[["ticket_id", "true_category"]],
    on="ticket_id"
)

valid_few_shot = few_shot_check[
    few_shot_check["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

few_shot_accuracy = (
    valid_few_shot["category"]
    == valid_few_shot["true_category"]
).mean() * 100

print("Few-shot accuracy:", round(few_shot_accuracy, 2), "%")

print("\nFEW-SHOT CONFUSION MATRIX")
print(
    pd.DataFrame(
        confusion_matrix(
            valid_few_shot["true_category"],
            valid_few_shot["category"],
            labels=["billing", "delivery", "technical", "fraud"]
        ),
        index=["Actual billing", "Actual delivery", "Actual technical", "Actual fraud"],
        columns=["Pred billing", "Pred delivery", "Pred technical", "Pred fraud"]
    )
)

few_shot_classified_count = len(valid_few_shot)

print(
    "Tokens per classified ticket:",
    round(few_shot_tokens / few_shot_classified_count, 2)
)

# Add customer segment information
few_shot_results = few_shot_results.merge(
    segments[["customer_id", "segment_name", "is_premium"]],
    on="customer_id",
    how="left"
)

few_shot_results["is_premium"] = (
    few_shot_results["is_premium"].fillna(False)
)

def assign_route(row):

    if row["status"] not in [
    "classified_json",
    "classified_keyword"
    ]:
        return "human_review", "NORMAL"

    if row["category"] == "fraud":
        return "security_team", "HIGH"

    if row["is_premium"]:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


few_shot_results[["route", "priority"]] = (
    few_shot_results.apply(
        lambda row: pd.Series(assign_route(row)),
        axis=1
    )
)

print("\nTICKET ROUTING")
print(few_shot_results["route"].value_counts())

print("\nPRIORITY COUNTS")
print(few_shot_results["priority"].value_counts())


# Save final few-shot triage results

final_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

final_results = few_shot_results[final_columns]

final_results.to_csv(
    "outputs/triage_results.csv",
    index=False
)

print("\nFINAL OUTPUT")
print("Saved: outputs/triage_results.csv")
print("Rows:", len(final_results))
print("\nSample results:")
print(final_results.head(10).to_string(index=False))


# Demonstrate token budget protection

budget_demo, budget_tokens = process_tickets(
    ZERO_SHOT_PROMPT,
    token_budget=6000
)

budget_demo.to_csv(
    "outputs/triage_budget_demo.csv",
    index=False
)

print("\nTOKEN BUDGET DEMO")
print(budget_demo["status"].value_counts())
print("Tokens used:", budget_tokens)
print("Saved: outputs/triage_budget_demo.csv")
