import os
import json
import sqlite3
import uuid
import time

from datetime import datetime
from mock_llm import chat, est_tokens, reset
from policy_rag import retrieve


DB_PATH = "data/instamart.db"
STATE_DIR = "state"
OUTPUT_DIR = "outputs"

os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FALLBACK_MAX_ITERATIONS = 6
DEFAULT_TOKEN_LIMIT = 6000
DEFAULT_TIMEOUT = 10


def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        """
        SELECT order_id, customer_id, status,
               order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
        """,
        (order_id,)
    ).fetchone()

    conn.close()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "customer_id": row[1],
        "status": row[2],
        "order_amount": row[3],
        "delivery_minutes": row[4]
    }

def search_policy(query):
    results = retrieve(query, k=3)

    if not results:
        return {
            "error": "NO_RELEVANT_POLICY"
        }

    return {
        "results": [
            {
                "source": item["source"],
                "score": item["score"],
                "text": item["text"]
            }
            for item in results
        ]
    }

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        refund = 0.0
    elif delivery_minutes > 45:
        refund = float(order_amount)
    else:
        refund = 0.0

    return {
        "refund_inr": round(refund, 2)
    }


def issue_refund(order_id, refund_inr):
    record = {
        "tool_name": "issue_refund",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "timestamp": datetime.now().isoformat()
    }

    with open("outputs/refund_ledger.jsonl", "a") as file:
        file.write(json.dumps(record) + "\n")

    return {"refund_inr": record["refund_inr"]}


def escalate_to_human(order_id, refund_inr):
    record = {
        "tool_name": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "timestamp": datetime.now().isoformat()
    }

    with open("outputs/refund_ledger.jsonl", "a") as file:
        file.write(json.dumps(record) + "\n")

    return {"refund_inr": record["refund_inr"]}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

def execute_tool(tool_name, arguments):
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    tool = TOOL_MAP.get(tool_name)

    if tool is None:
        return {"error": "UNKNOWN_TOOL"}

    return tool(**arguments)


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search refund and support policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate the eligible refund amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": ["number", "null"]}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an approved refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Send a refund requiring approval to a human.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


def save_checkpoint(state):
    path = os.path.join(
        STATE_DIR,
        f"{state['run_id']}.json"
    )

    with open(path, "w") as file:
        json.dump(state, file, indent=2)


SYSTEM_PROMPT = """
You are an Instamart customer support AGENT.

Follow these rules:

1. Always look up the order using get_order.
2. If the order does not exist, return NOT_FOUND.
3. Search the refund policy using search_policy.
4. Only completed orders are eligible for refunds.
5. If delivery took more than 45 minutes, calculate a full refund.
6. If delivery took 45 minutes or less, deny the refund.
7. If the refund exceeds INR 1000, escalate to a human.
8. Otherwise, issue the refund.
9. Never invent order information or policy rules.
10. Give a clear final answer.
"""

def load_checkpoint(run_id):
    path = os.path.join(
        STATE_DIR,
        f"{run_id}.json"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint found for {run_id}"
        )

    with open(path, "r") as file:
        return json.load(file)
    

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    customer_id=None,
    crash_after_tool=None
):
    run_id = str(uuid.uuid4())

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "customer_id": customer_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "status": "running",
        "final_answer": None,
        "started_at": time.monotonic()
    }

    save_checkpoint(state)

    return continue_agent(
        state,
        token_limit,
        time_limit_seconds,
        crash_after_tool,
        max_iterations
    )

def continue_agent(
    state,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tool=None,
    max_iterations=6
):
    start_time = state.get("started_at", time.monotonic())

    while state["step_number"] < max_iterations:

        if time.monotonic() - start_time >= time_limit_seconds:
            state["status"] = "FALLBACK_TIMEOUT"
            state["final_answer"] = (
                "FALLBACK_TIMEOUT: time limit reached. "
                "Please contact human support."
            )
            break

        if state["tokens_used"] >= token_limit:
            state["status"] = "FALLBACK_TOKEN_LIMIT"
            state["final_answer"] = (
                "FALLBACK_TOKEN_LIMIT: token budget exceeded. "
                "Please contact human support."
            )
            break

        try:
            response = chat(state["messages"], tools=TOOLS_SPEC)
        except Exception as error:
            state["status"] = "ERROR"
            state["final_answer"] = str(error)
            break

        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        state["step_number"] += 1

        if state["tokens_used"] > token_limit:
            state["status"] = "FALLBACK_TOKEN_LIMIT"
            state["final_answer"] = (
                "FALLBACK_TOKEN_LIMIT: token budget exceeded. "
                "Please contact human support."
            )
            break

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        if not tool_calls:
            state["final_answer"] = assistant_message.content or ""
            state["status"] = (
                state["final_answer"].split(":", 1)[0].strip()
                if ":" in state["final_answer"]
                else "ERROR"
            )
            break

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })

        for tool_call in tool_calls:
            tool_name = tool_call.name
            arguments = tool_call.arguments

            result = execute_tool(tool_name, arguments)

            state["tools_executed"].append(tool_name)
            state["steps"].append({
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": result
            })

            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result)
            })

            save_checkpoint(state)

            if (
                crash_after_tool is not None
                and len(state["tools_executed"]) == crash_after_tool
            ):
                state["status"] = "interrupted"
                save_checkpoint(state)
                return state

    if state["status"] == "running":
        state["status"] = "FALLBACK_MAX_ITERATIONS"
        state["final_answer"] = (
            "FALLBACK_MAX_ITERATIONS: maximum agent iterations reached. "
            "Please contact human support."
        )

    save_checkpoint(state)
    return state


def resume_agent(run_id):
    state = load_checkpoint(run_id)

    if state["status"] != "interrupted":
        raise ValueError(
            "Only interrupted runs can be resumed."
        )

    state["status"] = "running"

    save_checkpoint(state)

    return continue_agent(
        state,
        token_limit=DEFAULT_TOKEN_LIMIT,
        time_limit_seconds=DEFAULT_TIMEOUT
    )


def save_run_result(state):
    record = {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": state["status"],
        "final_answer": state["final_answer"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "steps": state["steps"],
        "tools_executed": state["tools_executed"]
    }

    with open("outputs/agent_runs.jsonl", "a") as file:
        file.write(json.dumps(record) + "\n")


def run_all_scenarios():
    import csv

    reset()

    with open("data/raw/agent_scenarios.csv", newline="") as file:
        scenarios = list(csv.DictReader(file))

    for scenario in scenarios:
        print("\nRunning:", scenario["scenario_id"])

        state = run_agent(
            scenario_id=scenario["scenario_id"],
            customer_id=scenario["customer_id"],
            user_message=scenario["message"],
            crash_after_tool=(
                2 if scenario["scenario_id"] == "S1" else None
            )
        )

        if state["status"] == "interrupted":
            print("Simulated crash. Resuming...")
            state = resume_agent(state["run_id"])

        save_run_result(state)

        print("Status:", state["status"])
        print("Answer:", state["final_answer"])
        print("Tools:", state["tools_executed"])
        print("LLM calls:", state["llm_calls"])

if __name__ == "__main__":
    run_all_scenarios()