import os
import json
import requests
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

API_URL = os.environ["LEDGER_API_URL"]

# AI Pipe / OpenAI-compatible settings
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()


class Question(BaseModel):
    question: str


# ============================================================
# LEDGER API
# ============================================================

def make_url(path, page=None):
    parsed = urlparse(API_URL)
    params = parse_qs(parsed.query)

    params["path"] = [path]

    if page is not None:
        params["page"] = [str(page)]
    else:
        params.pop("page", None)

    query = urlencode(params, doseq=True)

    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        query,
        parsed.fragment
    ))


def fetch_all_orders():
    all_orders = []

    for page in range(1, 86):
        response = requests.get(
            make_url("/orders", page),
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        all_orders.extend(data["orders"])

    return all_orders


def fetch_rates():
    response = requests.get(
        make_url("/rates"),
        timeout=10
    )
    response.raise_for_status()

    return response.json()["usd_per_unit"]


def deduplicate_orders(orders):
    latest = {}

    for order in orders:
        order_id = order["id"]

        if (
            order_id not in latest
            or order["updated_at"] > latest[order_id]["updated_at"]
        ):
            latest[order_id] = order

    return list(latest.values())


# ============================================================
# LOAD DATA ON STARTUP
# ============================================================

print("Loading ledger...")

orders = deduplicate_orders(fetch_all_orders())
rates = fetch_rates()

print(f"Loaded {len(orders)} unique orders")
print(f"Rates: {rates}")


# ============================================================
# LLM
# ============================================================

def understand_question(question):

    system_prompt = """
You are a query parser for an appliance order ledger.

Convert the user's question into JSON.

Possible intent values:

- revenue
- refunds
- product_sales
- customer_sales

Return ONLY valid JSON.

Use this exact structure:

{
  "intent": "revenue",
  "region": null,
  "product": null,
  "customer": null,
  "year": null,
  "month": null
}

Rules:

- Revenue means paid orders and their amount converted to USD.
- Refunds means orders whose status is refunded.
- For a month, use a number from 1 to 12.
- If the question does not specify a filter, use null.
- Preserve region/product/customer names exactly as stated.
"""

    response = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        timeout=10
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    return json.loads(content)


# ============================================================
# FILTERING
# ============================================================

def matches_date(order, year, month):

    if year is None and month is None:
        return True

    date = datetime.fromisoformat(
        order["created_at"].replace("Z", "+00:00")
    )

    if year is not None and date.year != year:
        return False

    if month is not None and date.month != month:
        return False

    return True


def filter_orders(parsed):

    result = orders

    if parsed["region"]:
        result = [
            o for o in result
            if o["region"].lower() == parsed["region"].lower()
        ]

    if parsed["product"]:
        result = [
            o for o in result
            if o["product"].lower() == parsed["product"].lower()
        ]

    if parsed["customer"]:
        result = [
            o for o in result
            if o["customer"].lower() == parsed["customer"].lower()
        ]

    result = [
        o for o in result
        if matches_date(
            o,
            parsed["year"],
            parsed["month"]
        )
    ]

    return result


# ============================================================
# CALCULATIONS
# ============================================================

def usd_amount(order):
    return order["amount"] * rates[order["currency"]]


def calculate(parsed):

    filtered = filter_orders(parsed)

    intent = parsed["intent"]

    if intent == "revenue":

        filtered = [
            o for o in filtered
            if o["status"] == "paid"
        ]

        total = sum(
            usd_amount(o)
            for o in filtered
        )

        return round(total, 2)

    if intent == "refunds":

        filtered = [
            o for o in filtered
            if o["status"] == "refunded"
        ]

        total = sum(
            usd_amount(o)
            for o in filtered
        )

        return round(total, 2)

    if intent == "product_sales":

        return sum(
            o["qty"]
            for o in filtered
            if o["status"] == "paid"
        )

    if intent == "customer_sales":

        return round(
            sum(
                usd_amount(o)
                for o in filtered
                if o["status"] == "paid"
            ),
            2
        )

    raise ValueError(f"Unknown intent: {intent}")


# ============================================================
# API ENDPOINT
# ============================================================

@app.post("/")
def answer_question(request: Question):

    parsed = understand_question(request.question)

    answer = calculate(parsed)

    return {
        "answer": answer
    }
