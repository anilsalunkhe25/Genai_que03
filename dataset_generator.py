#!/usr/bin/env python3
"""
AI-Powered Synthetic Dataset Generator
Domain: Online Shopping Order Management

Generates exactly 35 synthetic order records, exports them to CSV and JSON,
and produces a formatted validation report.
"""

import csv
import datetime
import json
import random
import re
import string
import sys
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from faker import Faker
    _FAKER_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when faker is missing
    _FAKER_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when pandas is missing
    _PANDAS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Domain & Schema Configuration
# ---------------------------------------------------------------------------
DOMAIN = "Online Shopping Order Management"

NUM_RECORDS = 35

PRODUCT_CATEGORIES = [
    "Electronics",
    "Clothing",
    "Home & Kitchen",
    "Books",
    "Beauty",
]

PAYMENT_STATUSES = [
    "Completed",
    "Pending",
    "Refunded",
    "Failed",
]

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

CSV_FILENAME = "synthetic_orders.csv"
JSON_FILENAME = "synthetic_orders.json"

# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

def _generate_order_id(existing_ids: set) -> str:
    """Generate a unique order_id matching ORD-XXXXX."""
    while True:
        candidate = "ORD-" + "".join(random.choice(string.digits) for _ in range(5))
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate


def _generate_customer_name(faker: Optional[object]) -> str:
    """Generate a synthetic (non-PII) customer name."""
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
    """Generate a random date string in YYYY-MM-DD format within [start, end]."""
    days = (end - start).days
    random_day = start + datetime.timedelta(days=random.randint(0, days))
    return random_day.strftime(DATE_FORMAT)


def generate_dataset(num_records: int = NUM_RECORDS, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate a list of synthetic order records."""
    if seed is not None:
        random.seed(seed)

    faker = Faker() if _FAKER_AVAILABLE else None

    records: List[Dict[str, Any]] = []
    used_order_ids: set = set()

    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 12, 31)

    for _ in range(num_records):
        order_amount = round(random.uniform(ORDER_AMOUNT_MIN, ORDER_AMOUNT_MAX), 2)
        record = OrderedDict()
        record["order_id"] = _generate_order_id(used_order_ids)
        record["customer_name"] = _generate_customer_name(faker)
        record["product_category"] = random.choice(PRODUCT_CATEGORIES)
        record["order_amount"] = order_amount
        record["order_date"] = _generate_order_date(start_date, end_date)
        record["payment_status"] = random.choice(PAYMENT_STATUSES)
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# File Export
# ---------------------------------------------------------------------------

def export_csv(records: List[Dict[str, Any]], filename: str = CSV_FILENAME) -> None:
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELD_ORDER)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print(f"[EXPORT] Wrote {len(records)} records to {filename}")


def export_json(records: List[Dict[str, Any]], filename: str = JSON_FILENAME) -> None:
    with open(filename, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    print(f"[EXPORT] Wrote {len(records)} records to {filename}")


# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------

def _check_missing(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count missing/null values per column."""
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
    """Check duplicate order_id values."""
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
    """Verify the data type of each value per column."""
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

            # Expected types per field
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
    """Validate categories, amounts, date formats, and order_id patterns."""
    issues: List[str] = []
    category_violations: List[int] = []
    amount_violations: List[int] = []
    date_violations: List[int] = []
    order_id_violations: List[int] = []
    payment_violations: List[int] = []

    for idx, record in enumerate(records):
        # order_id pattern
        oid = record.get("order_id")
        if not isinstance(oid, str) or not ORDER_ID_PATTERN.match(oid):
            order_id_violations.append(idx)
            issues.append(f"Row {idx}: invalid order_id format '{oid}'")

        # product_category
        cat = record.get("product_category")
        if cat not in PRODUCT_CATEGORIES:
            category_violations.append(idx)
            issues.append(f"Row {idx}: invalid product_category '{cat}'")

        # order_amount range
        amt = record.get("order_amount")
        if not isinstance(amt, (int, float)) or isinstance(amt, bool) or not (ORDER_AMOUNT_MIN <= amt <= ORDER_AMOUNT_MAX):
            amount_violations.append(idx)
            issues.append(f"Row {idx}: order_amount out of range '{amt}'")

        # order_date format
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

        # payment_status
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
    """Run all validation checks and return a combined report."""
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
# Reporting (pandas-based or pure-Python fallback)
# ---------------------------------------------------------------------------

def _format_validation_report_pandas(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("DATASET VALIDATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Domain: {report['domain']}")
    lines.append(f"Total Records: {report['total_records']}")
    lines.append(f"Fields: {', '.join(report['fields'])}")
    lines.append("")

    # Missing values
    lines.append("-" * 60)
    lines.append("1. MISSING / NULL VALUE CHECK")
    lines.append("-" * 60)
    total_missing = 0
    for field in report["fields"]:
        count = report["missing_values"][field]
        total_missing += count
        status = "PASS" if count == 0 else "FAIL"
        lines.append(f"  {field:<22} missing={count:<4} [{status}]")
    lines.append(f"  {'TOTAL':<22} missing={total_missing:<4}")
    lines.append("")

    # Duplicates
    lines.append("-" * 60)
    lines.append("2. DUPLICATE RECORD CHECK (order_id)")
    lines.append("-" * 60)
    dup = report["duplicates"]
    lines.append(f"  Total records:      {dup['total_records']}")
    lines.append(f"  Unique order_ids:   {dup['unique_order_ids']}")
    lines.append(f"  Duplicate count:    {dup['duplicate_count']}")
    lines.append(f"  Status:             {'PASS' if dup['duplicate_count'] == 0 else 'FAIL'}")
    if dup["duplicates"]:
        lines.append(f"  Duplicate IDs:      {dup['duplicates']}")
    lines.append("")

    # Data types
    lines.append("-" * 60)
    lines.append("3. DATA TYPE VERIFICATION")
    lines.append("-" * 60)
    for field in report["fields"]:
        info = report["data_types"][field]
        dist = info["type_distribution"]
        err_count = info["type_error_count"]
        status = "PASS" if err_count == 0 else "FAIL"
        dist_str = ", ".join(f"{k}={v}" for k, v in dist.items())
        lines.append(f"  {field:<22} types=[{dist_str}] errors={err_count:<4} [{status}]")
    lines.append("")

    # Schema consistency
    lines.append("-" * 60)
    lines.append("4. SCHEMA CONSISTENCY CHECK")
    lines.append("-" * 60)
    schema = report["schema_consistency"]
    lines.append(f"  order_id violations:      {len(schema['order_id_violations'])}")
    lines.append(f"  category violations:      {len(schema['category_violations'])}")
    lines.append(f"  amount violations:        {len(schema['amount_violations'])}")
    lines.append(f"  date violations:          {len(schema['date_violations'])}")
    lines.append(f"  payment_status violations:{len(schema['payment_violations'])}")
    lines.append(f"  Total issues:             {schema['total_issues']}")
    lines.append(f"  Status:                   {'PASS' if schema['total_issues'] == 0 else 'FAIL'}")
    if schema["issues"]:
        lines.append("  Issue details:")
        for issue in schema["issues"][:10]:
            lines.append(f"    - {issue}")
        if len(schema["issues"]) > 10:
            lines.append(f"    ... and {len(schema['issues']) - 10} more")
    lines.append("")

    # Overall
    lines.append("-" * 60)
    lines.append("5. OVERALL RESULT")
    lines.append("-" * 60)
    lines.append(f"  Validation: {'PASSED' if report['overall_pass'] else 'FAILED'}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _format_validation_report_python(report: Dict[str, Any]) -> str:
    """Pure-Python formatted report (identical layout to pandas version)."""
    return _format_validation_report_pandas(report)


def display_validation_report(report: Dict[str, Any]) -> str:
    """Build and print the formatted validation report."""
    if _PANDAS_AVAILABLE:
        # Build a small DataFrame summary for the console, then fall back to
        # the text layout for a clean, dependency-free report.
        try:
            summary_rows = []
            for field in report["fields"]:
                missing = report["missing_values"][field]
                dt = report["data_types"][field]
                summary_rows.append({
                    "Field": field,
                    "Missing": missing,
                    "TypeErrors": dt["type_error_count"],
                })
            df = pd.DataFrame(summary_rows)
            print("\n--- Column-Level Summary ---")
            print(df.to_string(index=False))
        except Exception:  # pragma: no cover
            pass

    report_text = _format_validation_report_python(report)
    print(report_text)
    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"AI-Powered Synthetic Dataset Generator")
    print(f"Domain: {DOMAIN}")
    print(f"Generating {NUM_RECORDS} synthetic records...")
    print(f"faker available: {_FAKER_AVAILABLE} | pandas available: {_PANDAS_AVAILABLE}")
    print("-" * 50)

    # 1. Generate
    records = generate_dataset(num_records=NUM_RECORDS, seed=42)

    # 2. Export
    export_csv(records, CSV_FILENAME)
    export_json(records, JSON_FILENAME)

    # 3. Validate
    report = validate_dataset(records)
    report_text = display_validation_report(report)

    # 4. Save report to file
    report_filename = "validation_report.txt"
    with open(report_filename, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"\n[REPORT] Validation report saved to {report_filename}")


if __name__ == "__main__":
    main()
