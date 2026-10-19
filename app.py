#!/usr/bin/env python3
"""
AI-Powered Synthetic Dataset Generator - Flask Web Application
Domain: Online Shopping Order Management

Runs on http://127.0.0.1:5000
"""

import csv
import datetime
import io
import json
import os
import re
import string
import sys
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional

from flask import Flask, redirect, render_template, request, send_file, url_for

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from faker import Faker

    _FAKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FAKER_AVAILABLE = False

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Domain & Schema Configuration
# ---------------------------------------------------------------------------
DOMAIN = "Online Shopping Order Management"
DEFAULT_NUM_RECORDS = 35

PRODUCT_CATEGORIES = [
    "Electronics",
    "Clothing",
    "Home & Kitchen",
    "Books",
    "Beauty",
]

PAYMENT_STATUSES = ["Completed", "Pending", "Refunded", "Failed"]

ORDER_ID_PATTERN = re.compile(r"^ORD-\d{5}$")
ORDER_AMOUNT_MIN = 10.00
ORDER_AMOUNT_MAX = 1500.00
DATE_FORMAT = "%Y-%m-%d"

FIELD_ORDER = [
    "order_id",
    "customer_name",
    "product_category",
    "order_amount",
    "order_date",
    "payment_status",
]

# In-memory store for the latest generated dataset (single-user localhost app)
DATASET: List[Dict[str, Any]] = []

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")


# ---------------------------------------------------------------------------
# Generation Helpers
# ---------------------------------------------------------------------------
def _generate_order_id(existing_ids: set) -> str:
    while True:
        candidate = "ORD-" + "".join(random.choice(string.digits) for _ in range(5))
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate


def _generate_customer_name(faker: Optional[object]) -> str:
    if _FAKER_AVAILABLE and faker is not None:
        return faker.name()
    first_names = [
        "Alex", "Jordan", "Casey", "Riley", "Quinn", "Avery", "Morgan", "Peyton",
        "Reese", "Sydney", "Taylor", "Jamie", "Dakota", "Emery", "Finley",
        "Gray", "Hayden", "Jesse", "Kai", "Lane",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    ]
    return random.choice(first_names) + " " + random.choice(last_names)


def _generate_order_date(start: datetime.date, end: datetime.date) -> str:
    days = (end - start).days
    random_day = start + datetime.timedelta(days=random.randint(0, days))
    return random_day.strftime(DATE_FORMAT)


import random  # noqa: E402  (placed here to avoid reordering noise)


def generate_dataset(num_records: int = DEFAULT_NUM_RECORDS, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)

    faker = Faker() if _FAKER_AVAILABLE else None
    records: List[Dict[str, Any]] = []
    used_order_ids: set = set()

    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)

    for _ in range(num_records):
        record = OrderedDict()
        record["order_id"] = _generate_order_id(used_order_ids)
        record["customer_name"] = _generate_customer_name(faker)
        record["product_category"] = random.choice(PRODUCT_CATEGORIES)
        record["order_amount"] = round(random.uniform(ORDER_AMOUNT_MIN, ORDER_AMOUNT_MAX), 2)
        record["order_date"] = _generate_order_date(start_date, end_date)
        record["payment_status"] = random.choice(PAYMENT_STATUSES)
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------
def _check_missing(records: List[Dict[str, Any]]) -> Dict[str, int]:
    missing: Dict[str, int] = {}
    for field in FIELD_ORDER:
        count = 0
        for record in records:
            value = record.get(field)
            if value is None or value == "":
                count += 1
        missing[field] = count
    return missing


def _check_duplicates(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [record.get("order_id") for record in records]
    counter = Counter(ids)
    dupes = {oid: cnt for oid, cnt in counter.items() if cnt > 1}
    return {
        "total_records": len(ids),
        "unique_order_ids": len(counter),
        "duplicate_count": sum(cnt - 1 for cnt in dupes.values()),
        "duplicates": dupes,
    }


def _check_data_types(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for field in FIELD_ORDER:
        type_counts: Counter = Counter()
        type_errors: List[int] = []
        for idx, record in enumerate(records):
            value = record.get(field)
            if value is None:
                type_counts["NoneType"] += 1
                continue
            type_name = type(value).__name__
            type_counts[type_name] += 1

            if field == "order_amount":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    type_errors.append(idx)
            else:
                if not isinstance(value, str):
                    type_errors.append(idx)

        results[field] = {
            "type_distribution": dict(type_counts),
            "type_error_indices": type_errors,
            "type_error_count": len(type_errors),
        }
    return results


def _check_schema_consistency(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[str] = []
    category_violations: List[int] = []
    amount_violations: List[int] = []
    date_violations: List[int] = []
    order_id_violations: List[int] = []
    payment_violations: List[int] = []

    for idx, record in enumerate(records):
        oid = record.get("order_id")
        if not isinstance(oid, str) or not ORDER_ID_PATTERN.match(oid):
            order_id_violations.append(idx)
            issues.append(f"Row {idx}: invalid order_id format '{oid}'")

        cat = record.get("product_category")
        if cat not in PRODUCT_CATEGORIES:
            category_violations.append(idx)
            issues.append(f"Row {idx}: invalid product_category '{cat}'")

        amt = record.get("order_amount")
        if not isinstance(amt, (int, float)) or isinstance(amt, bool) or not (ORDER_AMOUNT_MIN <= amt <= ORDER_AMOUNT_MAX):
            amount_violations.append(idx)
            issues.append(f"Row {idx}: order_amount out of range '{amt}'")

        odate = record.get("order_date")
        if not isinstance(odate, str):
            date_violations.append(idx)
            issues.append(f"Row {idx}: order_date not a string '{odate}'")
        else:
            try:
                datetime.datetime.strptime(odate, DATE_FORMAT)
            except ValueError:
                date_violations.append(idx)
                issues.append(f"Row {idx}: invalid date format '{odate}'")

        pstatus = record.get("payment_status")
        if pstatus not in PAYMENT_STATUSES:
            payment_violations.append(idx)
            issues.append(f"Row {idx}: invalid payment_status '{pstatus}'")

    return {
        "category_violations": category_violations,
        "amount_violations": amount_violations,
        "date_violations": date_violations,
        "order_id_violations": order_id_violations,
        "payment_violations": payment_violations,
        "total_issues": len(issues),
        "issues": issues,
    }


def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = OrderedDict()
    report["domain"] = DOMAIN
    report["total_records"] = len(records)
    report["fields"] = list(FIELD_ORDER)
    report["missing_values"] = _check_missing(records)
    report["duplicates"] = _check_duplicates(records)
    report["data_types"] = _check_data_types(records)
    report["schema_consistency"] = _check_schema_consistency(records)
    report["overall_pass"] = (
        sum(report["missing_values"].values()) == 0
        and report["duplicates"]["duplicate_count"] == 0
        and all(v["type_error_count"] == 0 for v in report["data_types"].values())
        and report["schema_consistency"]["total_issues"] == 0
    )
    return report


# ---------------------------------------------------------------------------
# Export Helpers
# ---------------------------------------------------------------------------
def _records_to_csv_bytes(records: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELD_ORDER)
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return buf.getvalue().encode("utf-8")


def _records_to_json_bytes(records: List[Dict[str, Any]]) -> bytes:
    return json.dumps(records, indent=2, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    validation_report = validate_dataset(DATASET) if DATASET else None
    return render_template(
        "index.html",
        domain=DOMAIN,
        num_records=len(DATASET),
        records=DATASET,
        field_order=FIELD_ORDER,
        validation_report=validation_report,
        faker_available=_FAKER_AVAILABLE,
        pandas_available=_PANDAS_AVAILABLE,
    )


@app.route("/generate", methods=["POST"])
def generate():
    global DATASET
    try:
        num_records = int(request.form.get("num_records", DEFAULT_NUM_RECORDS))
    except (TypeError, ValueError):
        num_records = DEFAULT_NUM_RECORDS
    if num_records < 1:
        num_records = DEFAULT_NUM_RECORDS

    DATASET = generate_dataset(num_records=num_records, seed=42)
    return redirect(url_for("index"))


@app.route("/download/csv")
def download_csv():
    if not DATASET:
        return redirect(url_for("index"))
    csv_bytes = _records_to_csv_bytes(DATASET)
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="synthetic_orders.csv",
    )


@app.route("/download/json")
def download_json():
    if not DATASET:
        return redirect(url_for("index"))
    json_bytes = _records_to_json_bytes(DATASET)
    return send_file(
        io.BytesIO(json_bytes),
        mimetype="application/json",
        as_attachment=True,
        download_name="synthetic_orders.json",
    )


@app.route("/clear", methods=["POST"])
def clear():
    global DATASET
    DATASET = []
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("AI-Powered Synthetic Dataset Generator (Flask)")
    print(f"Domain: {DOMAIN}")
    print(f"Server: http://127.0.0.1:5000")
    print(f"faker available: {_FAKER_AVAILABLE} | pandas available: {_PANDAS_AVAILABLE}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)
