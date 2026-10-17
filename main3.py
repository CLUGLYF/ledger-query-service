import os
import re
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from zoneinfo import ZoneInfo

import requests
from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# CONFIG
# ============================================================
API_URL = os.environ["LEDGER_API_URL"]
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openrouter/free")
# VERIFY against the API's path=/ description (business timezone)
BUSINESS_TZ = ZoneInfo(os.environ.get("BUSINESS_TZ", "Asia/Kolkata"))

app = FastAPI()


class Question(BaseModel):
    question: str


# ============================================================
# LEDGER API
# ============================================================
def make_url(path, page=None):
    p = urlparse(API_URL)
    params = parse_qs(p.query)
    params["path"] = [path]
    if page is not None:
        params["page"] = [str(page)]
    else:
        params.pop("page", None)
    return urlunparse((p.scheme, p.netloc, p.path, p.params,
                       urlencode(params, doseq=True), p.fragment))


def get_json(path, page=None, tries=3):
    last = None
    for _ in range(tries):
        try:
            r = requests.get(make_url(path, page), timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # retry
            last = e
    raise last


def fetch_all_orders():
    first = get_json("/orders", 1)
    print("PAGE1 KEYS:", {k: (v if not isinstance(v, list) else f"list[{len(v)}]")
                          for k, v in first.items()})
    orders = list(first.get("orders", []))

    total = None
    for k in ("total_pages", "pages", "page_count", "last_page", "num_pages"):
        if isinstance(first.get(k), int):
            total = first[k]
            break

    if total:
        with ThreadPoolExecutor(max_workers=16) as ex:
            pages = list(ex.map(lambda pg: get_json("/orders", pg),
                                range(2, total + 1)))
        for d in pages:
            orders.extend(d.get("orders", []))
    else:
        # walk until an empty page (in batches for speed)
        page = 2
        while page < 1000:
            batch = list(range(page, page + 16))
            with ThreadPoolExecutor(max_workers=16) as ex:
                res = list(ex.map(
                    lambda pg: get_json("/orders", pg), batch))
            done = False
            for d in res:
                o = d.get("orders", [])
                if not o:
                    done = True
                    break
                orders.extend(o)
            if done:
                break
            page += 16
    return orders


def parse_ts(s):
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def dedupe(orders):
    latest = {}
    for o in orders:
        oid = o["id"]
        cur = latest.get(oid)
        if cur is None:
            latest[oid] = o
            continue
        a, b = parse_ts(o.get("updated_at")), parse_ts(cur.get("updated_at"))
        if a is not None and b is not None:
            if a >= b:
                latest[oid] = o
        elif str(o.get("updated_at", "")) >= str(cur.get("updated_at", "")):
            latest[oid] = o
    return list(latest.values())


def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return float(x)


# ============================================================
# LOAD DATA (background, lazy-safe)
# ============================================================
STATE = {"orders": [], "rates": {}, "products": [], "customers": [],
         "regions": [], "statuses": []}
READY = threading.Event()
LOAD_LOCK = threading.Lock()


def load():
    with LOAD_LOCK:
        if READY.is_set():
            return
        raw = dedupe(fetch_all_orders())
        rates = {str(k).upper(): float(v)
                 for k, v in get_json("/rates")["usd_per_unit"].items()}
        rates.setdefault("USD", 1.0)

        orders = []
        for o in raw:
            try:
                dt = parse_ts(o["created_at"]).astimezone(BUSINESS_TZ)
                orders.append({
                    "id": o["id"],
                    "product": str(o["product"]).strip(),
                    "customer": str(o["customer"]).strip(),
                    "region": str(o["region"]).strip(),
                    "status": str(o["status"]).strip().lower(),
                    "qty": to_float(o.get("qty", 1)),
                    "usd": to_float(o["amount"]) * rates[str(o["currency"]).strip().upper()],
                    "year": dt.year,
                    "month": dt.month,
                })
            except Exception:
                print("BAD ORDER:", o)
        STATE["orders"] = orders
        STATE["rates"] = rates
        for key, f in (("products", "product"), ("customers", "customer"),
                       ("regions", "region"), ("statuses", "status")):
            STATE[key] = sorted({o[f] for o in orders}, key=len, reverse=True)
        print(f"Loaded {len(orders)} orders; statuses={STATE['statuses']}; "
              f"regions={STATE['regions']}; rates={rates}")
        READY.set()


@app.on_event("startup")
def startup():
    def run():
        for _ in range(5):
            try:
                load()
                return
            except Exception:
                traceback.print_exc()
    threading.Thread(target=run, daemon=True).start()


@app.get("/")
def health():
    return {"ready": READY.is_set()}


# ============================================================
# PARSING (rules first, LLM fallback)
# ============================================================
MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
MONTH_ABBR = {k[:3]: v for k, v in MONTHS.items()}
MONTH_ABBR["sept"] = 9


def find_entities(q, names):
    """Longest-first, non-overlapping, word-boundary match (plural tolerant)."""
    found, text = [], q
    for n in names:
        pat = r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?:e?s)?(?![a-z0-9])"
        m = re.search(pat, text)
        if m:
            found.append(n)
            text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]
    return found


def has(q, *words):
    return any(re.search(r"(?<![a-z])" + w + r"(?![a-z])", q) for w in words)


def rule_parse(question):
    q = re.sub(r"[^\w\s\-\.]", " ", question.lower())
    q = re.sub(r"\s+", " ", q)
    p = {"metric": "revenue", "status": None, "group_by": None,
         "ranking": None, "average": False, "product": None,
         "customer": None, "region": None, "year": None, "months": None,
         "uncertain": False}

    # entities from the data itself
    prods = find_entities(q, STATE["products"])
    custs = find_entities(q, STATE["customers"])
    regs = find_entities(q, STATE["regions"])
    p["product"] = prods[0] if prods else None
    p["customer"] = custs[0] if custs else None
    p["region"] = regs[0] if regs else None

    # dates
    y = re.search(r"\b(20\d\d)\b", q)
    if y:
        p["year"] = int(y.group(1))
    qm = re.search(r"\bq([1-4])\b", q) or re.search(
        r"\b(first|second|third|fourth) quarter\b", q)
    if qm:
        n = qm.group(1)
        n = {"first": 1, "second": 2, "third": 3, "fourth": 4}.get(n, n)
        n = int(n)
        p["months"] = [3 * (n - 1) + i for i in (1, 2, 3)]
    else:
        ms = []
        for w in re.findall(r"[a-z]+", q):
            if w in MONTHS:
                ms.append(MONTHS[w])
            elif w in MONTH_ABBR and w != "may":
                ms.append(MONTH_ABBR[w])
        if ms:
            p["months"] = sorted(set(ms))

    # status
    if has(q, "refund", "refunds", "refunded"):
        p["status"] = "refunded"
    else:
        for s in STATE["statuses"]:
            if s not in ("paid", "refunded") and re.search(
                    r"(?<![a-z])" + re.escape(s) + r"(?![a-z])", q):
                p["status"] = s
        if p["status"] is None:
            p["status"] = "paid"

    # metric
    cust_count = re.search(
        r"(how many|number of|count of|count|total)\s+(of\s+)?(the\s+)?"
        r"(distinct |unique |different |paying )*customers", q)
    order_count = re.search(
        r"(how many|number of|count of|count|total)\s+(of\s+)?(the\s+)?"
        r"(paid |refunded |total |completed )*(orders|transactions|purchases)", q)
    units = has(q, "units", "unit", "quantity", "quantities", "items", "pieces")

    if cust_count:
        p["metric"] = "customer_count"
    elif order_count or re.search(r"how many orders", q):
        p["metric"] = "order_count"
    elif units or (has(q, "how many") and has(q, "sold", "bought", "purchased", "ordered")):
        p["metric"] = "quantity"
    elif p["status"] == "refunded":
        p["metric"] = "refund"
    elif has(q, "how many") and not has(q, "dollars", "usd", "revenue"):
        p["metric"] = "order_count"
    else:
        p["metric"] = "revenue"

    if p["metric"] == "refund":
        p["status"] = "refunded"

    # average
    if has(q, "average", "avg", "mean", "typical") or "per order" in q:
        p["average"] = True
        if p["metric"] in ("customer_count", "order_count", "refund"):
            p["metric"] = "revenue" if p["metric"] != "refund" else "refund"

    # ranking / grouping
    if has(q, "least", "lowest", "smallest", "fewest", "minimum", "worst",
           "min", "cheapest", "lowest"):
        p["ranking"] = "min"
    elif has(q, "most", "highest", "largest", "greatest", "top", "best",
             "biggest", "maximum", "max", "leading"):
        p["ranking"] = "max"

    if p["metric"] not in ("customer_count",):
        if has(q, "which", "what", "who", "whose") or p["ranking"]:
            if has(q, "customer", "customers", "buyer", "who", "whose") and not p["customer"]:
                p["group_by"] = "customer"
            elif has(q, "product", "products", "item", "appliance") and not p["product"]:
                p["group_by"] = "product"
            elif has(q, "region", "regions", "area", "zone") and not p["region"]:
                p["group_by"] = "region"
        if p["group_by"] and not p["ranking"]:
            p["ranking"] = "max"
        if not p["group_by"]:
            p["ranking"] = None

    # confidence
    p["uncertain"] = not (
        p["product"] or p["customer"] or p["region"] or p["months"] or p["year"]
        or has(q, "revenue", "refund", "refunded", "refunds", "units", "orders",
               "customers", "sales", "sold", "average", "usd", "dollars", "money",
               "earned", "spent", "total"))
    return p


def llm_parse(question):
    if not (LLM_BASE_URL and LLM_API_KEY):
        return None
    sys_prompt = (
        'Return ONLY JSON: {"metric": revenue|refund|quantity|order_count|customer_count, '
        '"status": paid|refunded|null, "group_by": product|customer|region|null, '
        '"ranking": max|min|null, "average": bool, "product": str|null, '
        '"customer": str|null, "region": str|null, "year": int|null, '
        '"months": [1-12 ints]|null}. Revenue = paid orders only.')
    r = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={"model": LLM_MODEL, "temperature": 0,
              "messages": [{"role": "system", "content": sys_prompt},
                           {"role": "user", "content": question}]},
        timeout=7)
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"].strip()
    c = re.sub(r"^```(?:json)?\s*|\s*```$", "", c, flags=re.I)
    p = json.loads(c[c.index("{"): c.rindex("}") + 1])
    p.setdefault("status", "paid")
    p["status"] = p["status"] or "paid"
    p.setdefault("average", False)
    # snap names to real data values
    for key, pool in (("product", "products"), ("customer", "customers"),
                      ("region", "regions")):
        v = p.get(key)
        if v:
            hit = find_entities(str(v).lower(), STATE[pool])
            p[key] = hit[0] if hit else v
    return p


# ============================================================
# CALCULATION
# ============================================================
def select(p):
    out = []
    for o in STATE["orders"]:
        if p.get("region") and o["region"].lower() != str(p["region"]).lower():
            continue
        if p.get("product") and o["product"].lower() != str(p["product"]).lower():
            continue
        if p.get("customer") and o["customer"].lower() != str(p["customer"]).lower():
            continue
        if p.get("year") and o["year"] != int(p["year"]):
            continue
        if p.get("months") and o["month"] not in [int(m) for m in p["months"]]:
            continue
        if p.get("status") and o["status"] != p["status"]:
            continue
        out.append(o)
    return out


def measure(data, metric, average):
    if metric == "customer_count":
        return len({o["customer"] for o in data})
    if metric == "order_count":
        return len(data)
    field = "qty" if metric == "quantity" else "usd"
    total = sum(o[field] for o in data)
    if average:
        return total / len(data) if data else 0
    return total


def calculate(p):
    metric = p["metric"]
    data = select(p)
    gb = p.get("group_by")

    if gb:
        groups = {}
        for o in data:
            groups.setdefault(o[gb], []).append(o)
        if not groups:
            return 0
        vals = {k: measure(g, metric, p["average"]) for k, g in groups.items()}
        if p.get("ranking") == "min":
            return min(sorted(vals), key=lambda k: vals[k])
        return max(sorted(vals), key=lambda k: vals[k])

    v = measure(data, metric, p["average"])
    if metric in ("customer_count", "order_count"):
        return int(v)
    if metric == "quantity" and not p["average"]:
        return int(v) if float(v).is_integer() else v
    return round(v, 2)


# ============================================================
# API
# ============================================================
@app.post("/")
def answer_question(req: Question):
    try:
        if not READY.wait(timeout=11):
            load()
        p = rule_parse(req.question)
        if p["uncertain"]:
            try:
                alt = llm_parse(req.question)
                if alt:
                    p = {**p, **alt}
            except Exception:
                traceback.print_exc()
        print("QUESTION:", req.question, "PARSED:", p)
        ans = calculate(p)
        print("ANSWER:", ans)
        return {"answer": ans}
    except Exception:
        traceback.print_exc()
        return {"answer": 0}
