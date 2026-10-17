import os
import json
import re
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
LLM_MODEL = os.environ.get("LLM_MODEL", "openrouter/free")

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
            timeout=15
        )
        response.raise_for_status()

        data = response.json()
        all_orders.extend(data["orders"])

    return all_orders


def fetch_rates():
    response = requests.get(
        make_url("/rates"),
        timeout=15
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
# DATE
# ============================================================

def business_date(order):
    dt = datetime.fromisoformat(
        order["created_at"].replace("Z", "+00:00")
    )

    return dt.astimezone(BUSINESS_TZ)


def matches_date(order, year=None, month=None):
    if year is None and month is None:
        return True

    dt = business_date(order)

    if year is not None and dt.year != int(year):
        return False

    if month is not None and dt.month != int(month):
        return False

    return True


# ============================================================
# MONEY
# ============================================================

def usd_amount(order):
    return float(order["amount"]) * float(
        rates[order["currency"]]
    )


# ============================================================
# LLM PARSER
# ============================================================

def understand_question(question):

    system_prompt = """
You are a highly accurate query parser for an appliance order ledger.

Return ONLY valid JSON.

Schema:

{
  "metric": "revenue",
  "status": "paid",
  "group_by": null,
  "ranking": null,
  "average": false,
  "distinct": false,
  "region": null,
  "product": null,
  "customer": null,
  "year": null,
  "month": null
}

VALID metric values:

- revenue
- refund
- quantity
- order_count
- customer_count

VALID group_by values:

- null
- product
- customer
- region

VALID ranking values:

- null
- max
- min

Interpret questions carefully.

============================================================
REVENUE
============================================================

Use metric="revenue" for:

- revenue
- sales revenue
- sales
- money earned
- amount earned
- USD revenue
- total revenue
- how much money
- how much did X make
- how much did X generate

Revenue means ONLY paid orders.

============================================================
REFUNDS
============================================================

Use metric="refund" for:

- refunds
- refunded amount
- money refunded
- refund revenue
- how much was refunded

Refund calculations use ONLY orders whose status is "refunded".

============================================================
QUANTITY / UNITS
============================================================

Use metric="quantity" for:

- units sold
- quantity sold
- how many items sold
- how many products sold
- number of units
- total units

Quantity uses ONLY paid orders unless the question explicitly asks otherwise.

============================================================
ORDER COUNT
============================================================

Use metric="order_count" for:

- how many orders
- number of orders
- count of orders
- paid orders
- refunded orders

If the question says "paid orders", status="paid".

If it says "refunded orders", status="refunded".

============================================================
CUSTOMER COUNT
============================================================

Use metric="customer_count" for:

- how many customers
- number of customers
- distinct customers
- unique customers
- how many different customers

When asking how many customers bought/purchased something,
use group_by=null and metric="customer_count".

============================================================
GROUPING
============================================================

Use group_by="product" when asking:

- which product
- what product
- best-selling product
- product with highest revenue
- product with most sales
- product with lowest revenue

Use group_by="customer" when asking:

- which customer
- top customer
- customer who spent the most
- customer with highest revenue

Use group_by="region" when asking:

- which region
- region with highest revenue
- region with most sales

============================================================
RANKING
============================================================

Use ranking="max" for:

- most
- highest
- largest
- greatest
- top
- best
- maximum

Use ranking="min" for:

- least
- lowest
- smallest
- minimum
- worst

If the question asks "which product earned the most revenue",
use:

metric="revenue"
group_by="product"
ranking="max"

If the question asks "which product sold the most units",
use:

metric="quantity"
group_by="product"
ranking="max"

============================================================
AVERAGES
============================================================

Set average=true when the question says:

- average
- on average
- mean
- per order
- average order value
- average amount

Examples:

"On average, how many US dollars is a paid South order worth?"

means:

metric="revenue"
status="paid"
average=true
region="South"

"Average revenue per order for Air Fryer"

means:

metric="revenue"
average=true
product="Air Fryer"

============================================================
DISTINCT
============================================================

Set distinct=true when the question says:

- distinct
- unique
- different

For example:

"How many distinct customers bought an Air Fryer?"

means:

metric="customer_count"
distinct=true
status="paid"
product="Air Fryer"

============================================================
FILTERS
============================================================

Extract:

- region
- product
- customer
- year
- month

Months must be numbers 1-12.

January=1
February=2
March=3
April=4
May=5
June=6
July=7
August=8
September=9
October=10
November=11
December=12

Preserve product, customer and region names exactly as stated.

============================================================
IMPORTANT
============================================================

"Which product earned the most USD revenue?"
=> metric=revenue, group_by=product, ranking=max

"Which product sold the most?"
=> metric=quantity, group_by=product, ranking=max

"Which customer spent the most?"
=> metric=revenue, group_by=customer, ranking=max

"Which region generated the most revenue?"
=> metric=revenue, group_by=region, ranking=max

"How many distinct customers bought an Air Fryer?"
=> metric=customer_count, distinct=true, status=paid, product=Air Fryer

"How much revenue did Air Fryer generate?"
=> metric=revenue, product=Air Fryer

"How many Air Fryers were sold?"
=> metric=quantity, product=Air Fryer, status=paid

"What was the average paid order value?"
=> metric=revenue, average=true, status=paid

"How much was refunded?"
=> metric=refund

Only return JSON.
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

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE
        )
        content = re.sub(
            r"\s*```$",
            "",
            content
        )

    parsed = json.loads(content)

    return normalize_parsed(parsed, question)


# ============================================================
# NORMALIZE PARSER OUTPUT
# ============================================================

def normalize_parsed(parsed, question):

    q = question.lower().strip()

    parsed.setdefault("metric", "revenue")
    parsed.setdefault("status", "paid")
    parsed.setdefault("group_by", None)
    parsed.setdefault("ranking", None)
    parsed.setdefault("average", False)
    parsed.setdefault("distinct", False)
    parsed.setdefault("region", None)
    parsed.setdefault("product", None)
    parsed.setdefault("customer", None)
    parsed.setdefault("year", None)
    parsed.setdefault("month", None)

    # ----------------------------
    # Explicit average detection
    # ----------------------------

    if any(x in q for x in [
        "average",
        "on average",
        "mean",
        "per order",
        "average order value"
    ]):
        parsed["average"] = True
        parsed["metric"] = "revenue"

    # ----------------------------
    # Explicit distinct detection
    # ----------------------------

    if any(x in q for x in [
        "distinct",
        "unique",
        "different customers"
    ]):
        parsed["distinct"] = True

    # ----------------------------
    # Status
    # ----------------------------

    if any(x in q for x in [
        "paid",
        "purchases",
        "purchased",
        "bought",
        "sold"
    ]):
        if "refund" not in q:
            parsed["status"] = "paid"

    if "refunded" in q or "refund" in q:
        if (
            "how much was refunded" in q
            or "refund amount" in q
            or "total refunds" in q
            or "refunded amount" in q
        ):
            parsed["metric"] = "refund"
            parsed["status"] = "refunded"

    # ----------------------------
    # Ranking
    # ----------------------------

    if any(x in q for x in [
        "most",
        "highest",
        "largest",
        "greatest",
        "top",
        "best"
    ]):
        parsed["ranking"] = "max"

    elif any(x in q for x in [
        "least",
        "lowest",
        "smallest",
        "minimum",
        "worst"
    ]):
        parsed["ranking"] = "min"

    # ----------------------------
    # Grouping
    # ----------------------------

    if "customer" in q:
        if any(x in q for x in [
            "which customer",
            "what customer",
            "top customer",
            "customer who",
            "customer with"
        ]):
            parsed["group_by"] = "customer"

    if "product" in q:
        if any(x in q for x in [
            "which product",
            "what product",
            "product that",
            "product with",
            "best-selling product",
            "best selling product"
        ]):
            parsed["group_by"] = "product"

    if "region" in q:
        if any(x in q for x in [
            "which region",
            "what region",
            "region with",
            "region that"
        ]):
            parsed["group_by"] = "region"

    # ----------------------------
    # Revenue vs quantity
    # ----------------------------

    if any(x in q for x in [
        "revenue",
        "usd",
        "dollars",
        "money",
        "earned",
        "generated",
        "spent",
        "worth",
        "amount"
    ]):
        parsed["metric"] = "revenue"

    if any(x in q for x in [
        "units",
        "quantity",
        "items sold",
        "number sold",
        "how many sold"
    ]):
        parsed["metric"] = "quantity"

    # ----------------------------
    # Customer count
    # ----------------------------

    if (
        "customer" in q
        and any(x in q for x in [
            "how many",
            "number of",
            "count",
            "distinct",
            "unique"
        ])
    ):
        parsed["metric"] = "customer_count"
        parsed["group_by"] = None
        parsed["average"] = False

    # ----------------------------
    # Order count
    # ----------------------------

    if (
        "order" in q
        and any(x in q for x in [
            "how many",
            "number of",
            "count"
        ])
    ):
        parsed["metric"] = "order_count"
        parsed["group_by"] = None
        parsed["average"] = False

    return parsed


# ============================================================
# FILTERING
# ============================================================

def filter_orders(parsed):

    result = orders

    region = parsed.get("region")
    product = parsed.get("product")
    customer = parsed.get("customer")
    year = parsed.get("year")
    month = parsed.get("month")

    if region:
        result = [
            o for o in result
            if o["region"].lower() == str(region).lower()
        ]

    if product:
        result = [
            o for o in result
            if o["product"].lower() == str(product).lower()
        ]

    if customer:
        result = [
            o for o in result
            if o["customer"].lower() == str(customer).lower()
        ]

    result = [
        o for o in result
        if matches_date(o, year, month)
    ]

    return result


# ============================================================
# STATUS FILTER
# ============================================================

def status_filtered(parsed):

    filtered = filter_orders(parsed)

    status = parsed.get("status")

    if status:
        filtered = [
            o for o in filtered
            if o["status"] == status
        ]

    return filtered


# ============================================================
# GROUPING
# ============================================================

def group_orders(data, group_by):

    groups = {}

    for order in data:

        if group_by == "product":
            key = order["product"]

        elif group_by == "customer":
            key = order["customer"]

        elif group_by == "region":
            key = order["region"]

        else:
            key = "all"

        groups.setdefault(key, []).append(order)

    return groups


# ============================================================
# CALCULATIONS
# ============================================================

def calculate(parsed):

    metric = parsed.get("metric", "revenue")
    group_by = parsed.get("group_by")
    ranking = parsed.get("ranking")
    average = parsed.get("average", False)
    distinct = parsed.get("distinct", False)

    # ========================================================
    # REFUNDS
    # ========================================================

    if metric == "refund":

        data = filter_orders(parsed)

        data = [
            o for o in data
            if o["status"] == "refunded"
        ]

        if group_by and ranking:

            groups = group_orders(data, group_by)

            values = {
                key: sum(usd_amount(o) for o in group)
                for key, group in groups.items()
            }

            if not values:
                return 0

            if ranking == "max":
                return max(values, key=values.get)

            return min(values, key=values.get)

        return round(
            sum(usd_amount(o) for o in data),
            2
        )


    # ========================================================
    # DISTINCT CUSTOMERS
    # ========================================================

    if metric == "customer_count":

        data = status_filtered(parsed)

        customers = {
            o["customer"]
            for o in data
        }

        return len(customers)


    # ========================================================
    # ORDER COUNT
    # ========================================================

    if metric == "order_count":

        data = status_filtered(parsed)

        if group_by and ranking:

            groups = group_orders(data, group_by)

            values = {
                key: len(group)
                for key, group in groups.items()
            }

            if not values:
                return 0

            if ranking == "max":
                return max(values, key=values.get)

            return min(values, key=values.get)

        return len(data)


    # ========================================================
    # QUANTITY
    # ========================================================

    if metric == "quantity":

        data = status_filtered(parsed)

        if group_by and ranking:

            groups = group_orders(data, group_by)

            values = {
                key: sum(o["qty"] for o in group)
                for key, group in groups.items()
            }

            if not values:
                return 0

            if ranking == "max":
                return max(values, key=values.get)

            return min(values, key=values.get)

        return sum(o["qty"] for o in data)


    # ========================================================
    # REVENUE
    # ========================================================

    if metric == "revenue":

        data = status_filtered(parsed)

        # ----------------------------------------------------
        # GROUPED REVENUE
        # ----------------------------------------------------

        if group_by:

            groups = group_orders(data, group_by)

            values = {
                key: sum(
                    usd_amount(o)
                    for o in group
                )
                for key, group in groups.items()
            }

            if not values:
                return 0

            # ------------------------------------------------
            # Which product/customer/region made the most?
            # ------------------------------------------------

            if ranking == "max":
                return max(values, key=values.get)

            if ranking == "min":
                return min(values, key=values.get)

            # ------------------------------------------------
            # Average revenue for each group
            # ------------------------------------------------

            if average:

                averages = {}

                for key, group in groups.items():

                    if group:
                        averages[key] = (
                            sum(
                                usd_amount(o)
                                for o in group
                            )
                            / len(group)
                        )

                if ranking == "max":
                    return max(
                        averages,
                        key=averages.get
                    )

                if ranking == "min":
                    return min(
                        averages,
                        key=averages.get
                    )

                return {
                    key: round(value, 2)
                    for key, value in averages.items()
                }

            return {
                key: round(value, 2)
                for key, value in values.items()
            }

        # ----------------------------------------------------
        # AVERAGE ORDER VALUE
        # ----------------------------------------------------

        if average:

            if not data:
                return 0

            total = sum(
                usd_amount(o)
                for o in data
            )

            return round(
                total / len(data),
                2
            )

        # ----------------------------------------------------
        # TOTAL REVENUE
        # ----------------------------------------------------

        return round(
            sum(
                usd_amount(o)
                for o in data
            ),
            2
        )


    raise ValueError(
        f"Unsupported metric: {metric}"
    )


# ============================================================
# API
# ============================================================

@app.post("/")
def answer_question(request: Question):

    parsed = understand_question(
        request.question
    )

    print("QUESTION:", request.question)
    print("PARSED:", parsed)

    answer = calculate(parsed)

    print("ANSWER:", answer)

    return {
        "answer": answer
    }
