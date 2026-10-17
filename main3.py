import os
import json
import requests
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

API_URL = os.environ["LEDGER_API_URL"]

LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

BUSINESS_TZ = ZoneInfo("Asia/Kolkata")


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
# LOAD DATA
# ============================================================

print("Loading ledger...")

orders = deduplicate_orders(fetch_all_orders())
rates = fetch_rates()

print(f"Loaded {len(orders)} unique orders")
print(f"Rates: {rates}")


# ============================================================
# LLM QUESTION PARSER
# ============================================================

def understand_question(question):

    system_prompt = """
You are a query parser for an appliance order ledger.

Convert the user's question into JSON.

Return ONLY valid JSON.

Use EXACTLY this structure:

{
  "intent": "revenue",
  "region": null,
  "product": null,
  "customer": null,
  "year": null,
  "month": null
}

Possible intent values:

1. "revenue"
   Use for questions asking total revenue.

2. "refunds"
   Use for questions asking total refunded amount.

3. "product_sales"
   Use for questions asking how many units of a product were sold.

4. "customer_sales"
   Use for questions asking how much a customer spent.

5. "top_product_revenue"
   Use when the question asks WHICH PRODUCT earned/generated/made the most
   revenue, sales revenue, or USD revenue.

   Examples:
   - Which product earned the most USD revenue?
   - What product generated the highest revenue?
   - Which product made the most money?
   - What was the top product by revenue?

6. "top_product_sales"
   Use when the question asks WHICH PRODUCT sold the most units.

   Examples:
   - Which product sold the most units?
   - What was the best-selling product?
   - Which product had the highest number of sales?

7. "distinct_customers"
   Use when the question asks how many UNIQUE or DISTINCT customers
   bought a product.

   Examples:
   - How many distinct customers bought an Air Fryer?
   - How many unique customers purchased a Kettle?

Rules:

- Revenue counts ONLY orders whose current status is "paid".
- Refunds count ONLY orders whose current status is "refunded".
- Revenue must be converted to USD using the supplied exchange rates.
- Product sales count quantity from paid orders.
- For "top_product_revenue", return intent "top_product_revenue".
- For "top_product_sales", return intent "top_product_sales".
- For "distinct_customers", return intent "distinct_customers".
- If a product is specified, preserve its name exactly.
- If a customer is specified, preserve the customer exactly.
- If a region is specified, preserve the region exactly.
- For a month, use a number from 1 to 12.
- If year/month/region/product/customer is not specified, use null.
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

    content = response.json()["choices"][0]["message"]["content"].strip()

    # Handle models that occasionally return ```json ... ```
    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "")
        content = content.strip()

    return json.loads(content)


# ============================================================
# DATE HANDLING
# ============================================================

def get_business_date(order):
    """
    Convert created_at into the business timezone (Asia/Kolkata).
    """

    date = datetime.fromisoformat(
        order["created_at"].replace("Z", "+00:00")
    )

    return date.astimezone(BUSINESS_TZ)


def matches_date(order, year, month):

    if year is None and month is None:
        return True

    date = get_business_date(order)

    if year is not None and date.year != year:
        return False

    if month is not None and date.month != month:
        return False

    return True


# ============================================================
# FILTERING
# ============================================================

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


    # --------------------------------------------------------
    # TOTAL REVENUE
    # --------------------------------------------------------

    if intent == "revenue":

        paid = [
            o for o in filtered
            if o["status"] == "paid"
        ]

        total = sum(
            usd_amount(o)
            for o in paid
        )

        return round(total, 2)


    # --------------------------------------------------------
    # REFUNDS
    # --------------------------------------------------------

    if intent == "refunds":

        refunded = [
            o for o in filtered
            if o["status"] == "refunded"
        ]

        total = sum(
            usd_amount(o)
            for o in refunded
        )

        return round(total, 2)


    # --------------------------------------------------------
    # PRODUCT UNIT SALES
    # --------------------------------------------------------

    if intent == "product_sales":

        return sum(
            o["qty"]
            for o in filtered
            if o["status"] == "paid"
        )


    # --------------------------------------------------------
    # CUSTOMER REVENUE
    # --------------------------------------------------------

    if intent == "customer_sales":

        total = sum(
            usd_amount(o)
            for o in filtered
            if o["status"] == "paid"
        )

        return round(total, 2)


    # --------------------------------------------------------
    # PRODUCT WITH HIGHEST REVENUE
    # --------------------------------------------------------

    if intent == "top_product_revenue":

        paid = [
            o for o in filtered
            if o["status"] == "paid"
        ]

        revenue_by_product = {}

        for order in paid:
            product = order["product"]

            revenue_by_product[product] = (
                revenue_by_product.get(product, 0)
                + usd_amount(order)
            )

        if not revenue_by_product:
            return None

        best_product = max(
            revenue_by_product,
            key=revenue_by_product.get
        )

        return best_product


    # --------------------------------------------------------
    # PRODUCT WITH MOST UNITS SOLD
    # --------------------------------------------------------

    if intent == "top_product_sales":

        paid = [
            o for o in filtered
            if o["status"] == "paid"
        ]

        units_by_product = {}

        for order in paid:
            product = order["product"]

            units_by_product[product] = (
                units_by_product.get(product, 0)
                + order["qty"]
            )

        if not units_by_product:
            return None

        best_product = max(
            units_by_product,
            key=units_by_product.get
        )

        return best_product


    # --------------------------------------------------------
    # DISTINCT CUSTOMERS
    # --------------------------------------------------------

    if intent == "distinct_customers":

        paid = [
            o for o in filtered
            if o["status"] == "paid"
        ]

        customers = {
            o["customer"]
            for o in paid
        }

        return len(customers)


    raise ValueError(f"Unknown intent: {intent}")


# ============================================================
# API ENDPOINT
# ============================================================

@app.post("/")
def answer_question(request: Question):

    parsed = understand_question(request.question)

    print("QUESTION:", request.question)
    print("PARSED:", parsed)

    answer = calculate(parsed)

    print("ANSWER:", answer)

    return {
        "answer": answer
    }
