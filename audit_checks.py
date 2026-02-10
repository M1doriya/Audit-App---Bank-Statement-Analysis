from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class AuditConfig:
    amount_tolerance: float = 0.01
    max_examples: int = 30
    strict_schema: bool = False
    compare_only_common_fields: bool = True


@dataclass
class CheckFinding:
    name: str
    status: str  # "pass" | "fail" | "warn"
    message: str
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AuditResult:
    overall_pass: bool
    checks_run: int
    checks_failed: int
    warnings: int
    failed_checks: List[CheckFinding] = field(default_factory=list)
    warning_checks: List[CheckFinding] = field(default_factory=list)
    recomputed_summary: Optional[Dict[str, Any]] = None
    recomputed_monthly_summary: Optional[List[Dict[str, Any]]] = None
    normalization_notes: List[str] = field(default_factory=list)

    monthly_value_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    monthly_missing_fields: List[Dict[str, Any]] = field(default_factory=list)
    monthly_overview: List[Dict[str, Any]] = field(default_factory=list)


TRANSACTION_FIELD_ALIASES = {
    "date": ["date", "transaction_date", "posting_date", "posted_date", "value_date", "txn_date"],
    "description": ["description", "transaction_description", "narration", "remarks", "details", "memo"],
    "debit": ["debit", "withdrawal", "debit_amount", "money_out", "outflow", "amount_out"],
    "credit": ["credit", "deposit", "credit_amount", "money_in", "inflow", "amount_in"],
    "amount": ["amount", "transaction_amount", "amt"],
    "type": ["type", "transaction_type", "dr_cr", "direction", "entry_type"],
}

MONTHLY_FIELD_ALIASES = {
    "month": ["month", "period", "month_key", "year_month", "statement_month"],
    "transaction_count": ["transaction_count", "count", "txn_count", "no_of_transactions"],
    "total_debit": ["total_debit", "debit_total", "sum_debit", "withdrawal_total", "total_withdrawal"],
    "total_credit": ["total_credit", "credit_total", "sum_credit", "deposit_total", "total_deposit"],
    "net_change": ["net_change", "net", "net_flow", "balance_change", "net_movement"],
}


def _parse_date_safe(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
        if pd.isna(dt):
            dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def _month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _float_or_nan(x: Any) -> float:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return float("nan")
        if isinstance(x, str):
            cleaned = x.replace(",", "").replace("RM", "").replace("$", "").strip()
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = f"-{cleaned[1:-1]}"
            return float(cleaned)
        return float(x)
    except Exception:
        return float("nan")


def _is_num(x: Any) -> bool:
    try:
        f = float(x)
        return math.isfinite(f)
    except Exception:
        return False


def _round2(x: Any) -> Optional[float]:
    try:
        f = float(x)
        if not math.isfinite(f):
            return None
        return round(f, 2)
    except Exception:
        return None


def _first_present(row: Dict[str, Any], aliases: List[str]) -> Any:
    for name in aliases:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _normalize_transaction_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    normalized = {
        "date": _first_present(row, TRANSACTION_FIELD_ALIASES["date"]),
        "description": _first_present(row, TRANSACTION_FIELD_ALIASES["description"]),
        "debit": _first_present(row, TRANSACTION_FIELD_ALIASES["debit"]),
        "credit": _first_present(row, TRANSACTION_FIELD_ALIASES["credit"]),
    }

    amount = _first_present(row, TRANSACTION_FIELD_ALIASES["amount"])
    tx_type = _first_present(row, TRANSACTION_FIELD_ALIASES["type"])

    if normalized["debit"] is None and normalized["credit"] is None and amount is not None:
        amount_val = _float_or_nan(amount)
        direction = str(tx_type).strip().lower() if tx_type is not None else ""
        if math.isfinite(amount_val):
            if direction in {"debit", "dr", "withdrawal", "out", "outflow"}:
                normalized["debit"] = abs(amount_val)
                normalized["credit"] = 0.0
                notes.append("Derived debit/credit from single amount + transaction type.")
            elif direction in {"credit", "cr", "deposit", "in", "inflow"}:
                normalized["debit"] = 0.0
                normalized["credit"] = abs(amount_val)
                notes.append("Derived debit/credit from single amount + transaction type.")
            else:
                if amount_val < 0:
                    normalized["debit"] = abs(amount_val)
                    normalized["credit"] = 0.0
                else:
                    normalized["debit"] = 0.0
                    normalized["credit"] = amount_val
                notes.append("Derived debit/credit from signed single amount.")

    return normalized, notes


def _normalize_transactions(rows: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not isinstance(rows, list):
        return [], ["Input transactions is not a list; treated as empty."]

    out: List[Dict[str, Any]] = []
    notes: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized, row_notes = _normalize_transaction_row(row)
        out.append(normalized)
        notes.extend(row_notes)

    unique_notes = sorted(set(notes))
    return out, unique_notes


def _normalize_month_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    dt = _parse_date_safe(value)
    if dt:
        return _month_key(dt)

    s = str(value).strip()
    if not s:
        return None

    month_formats = ["%Y-%m", "%Y/%m", "%b %Y", "%B %Y", "%Y%m"]
    for fmt in month_formats:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m")
        except ValueError:
            continue

    return s


def _normalize_monthly_rows(rows: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not isinstance(rows, list):
        return [], ["Input monthly_summary is not a list; treated as empty."]

    notes: List[str] = []
    normalized: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        normalized_row: Dict[str, Any] = {}
        for field, aliases in MONTHLY_FIELD_ALIASES.items():
            val = _first_present(row, aliases)
            if field == "month":
                normalized_row[field] = _normalize_month_key(val)
            else:
                normalized_row[field] = val

        if normalized_row.get("month"):
            normalized.append(normalized_row)

        if any(alias in row for alias in ["period", "year_month", "statement_month"]):
            notes.append("Mapped alternate monthly period keys to `month`.")
        if any(alias in row for alias in ["txn_count", "count", "no_of_transactions"]):
            notes.append("Mapped alternate monthly transaction count keys.")

    return normalized, sorted(set(notes))


def _index_by_month(rows: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("month"):
            out[str(r["month"])] = r
    return out


def recompute_summary_and_monthly(transactions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    df = pd.DataFrame(transactions)
    if df.empty:
        return ({"total_transactions": 0, "date_range": {"start": None, "end": None}}, [])

    df["__date"] = df["date"].apply(_parse_date_safe) if "date" in df.columns else None
    df["__debit"] = df["debit"].apply(_float_or_nan) if "debit" in df.columns else float("nan")
    df["__credit"] = df["credit"].apply(_float_or_nan) if "credit" in df.columns else float("nan")

    valid_dates = df["__date"].dropna()
    start = valid_dates.min() if not valid_dates.empty else None
    end = valid_dates.max() if not valid_dates.empty else None

    summary = {
        "total_transactions": int(len(df)),
        "date_range": {
            "start": start.strftime("%Y-%m-%d") if start else None,
            "end": end.strftime("%Y-%m-%d") if end else None,
        },
    }

    df_valid = df.dropna(subset=["__date"]).copy()
    if df_valid.empty:
        return summary, []

    df_valid["__month"] = df_valid["__date"].apply(_month_key)

    rows: List[Dict[str, Any]] = []
    for month, group in df_valid.groupby("__month", sort=True):
        debit = float(pd.Series(group["__debit"]).fillna(0.0).sum())
        credit = float(pd.Series(group["__credit"]).fillna(0.0).sum())
        rows.append(
            {
                "month": month,
                "transaction_count": int(len(group)),
                "total_debit": round(debit, 2),
                "total_credit": round(credit, 2),
                "net_change": round(credit - debit, 2),
            }
        )

    return summary, rows


def compare_monthly_summary(
    provided_rows: Any,
    recomputed_rows: List[Dict[str, Any]],
    cfg: AuditConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    prov = _index_by_month(provided_rows)
    reco = _index_by_month(recomputed_rows)

    months = sorted(set(prov.keys()) | set(reco.keys()))
    computed_fields = ["transaction_count", "total_debit", "total_credit", "net_change"]

    value_mismatches: List[Dict[str, Any]] = []
    missing_fields: List[Dict[str, Any]] = []
    overview: List[Dict[str, Any]] = []

    for month in months:
        pr = prov.get(month) or {}
        rr = reco.get(month) or {}

        if not pr and rr:
            overview.append(
                {
                    "month": month,
                    "status": "MISSING_MONTH_IN_PROVIDED",
                    "summary": "Provided monthly_summary has no row for this month.",
                    "action": "Check monthly grouping in upstream JSON output.",
                }
            )
            continue

        if pr and not rr:
            overview.append(
                {
                    "month": month,
                    "status": "MISSING_MONTH_IN_RECOMPUTED",
                    "summary": "No transactions found for this month in recomputed side.",
                    "action": "Check transaction dates or parsing in source data.",
                }
            )
            continue

        missing = []
        mism = []

        for field in computed_fields:
            pv = pr.get(field, None)
            rv = rr.get(field, None)

            if pv is None:
                missing.append(field)
                missing_fields.append(
                    {
                        "month": month,
                        "field": field,
                        "meaning": "Field absent in provided monthly_summary (schema gap).",
                        "recomputed_value": rv,
                        "action": "Add this field upstream or enable compare_only_common_fields.",
                    }
                )
                continue

            if field == "transaction_count":
                try:
                    if int(pv) != int(rv):
                        mism.append(field)
                        value_mismatches.append(
                            {
                                "month": month,
                                "field": field,
                                "provided": int(pv),
                                "recomputed": int(rv),
                                "delta": int(rv) - int(pv),
                                "meaning": "Transaction count differs.",
                            }
                        )
                except Exception:
                    mism.append(field)
                    value_mismatches.append(
                        {
                            "month": month,
                            "field": field,
                            "provided": pv,
                            "recomputed": rv,
                            "delta": None,
                            "meaning": "Transaction count differs (non-numeric).",
                        }
                    )
            else:
                if _is_num(pv) and _is_num(rv):
                    delta = float(rv) - float(pv)
                    if abs(delta) > cfg.amount_tolerance:
                        mism.append(field)
                        value_mismatches.append(
                            {
                                "month": month,
                                "field": field,
                                "provided": _round2(pv),
                                "recomputed": _round2(rv),
                                "delta": round(delta, 2),
                                "abs_delta": round(abs(delta), 2),
                                "meaning": f"{field} differs beyond tolerance (±{cfg.amount_tolerance}).",
                            }
                        )
                elif str(pv) != str(rv):
                    mism.append(field)
                    value_mismatches.append(
                        {
                            "month": month,
                            "field": field,
                            "provided": pv,
                            "recomputed": rv,
                            "delta": None,
                            "meaning": f"{field} differs (non-numeric).",
                        }
                    )

        if mism:
            overview.append(
                {
                    "month": month,
                    "status": "VALUE_MISMATCH",
                    "summary": f"Real mismatch in: {', '.join(mism)}",
                    "action": "Investigate transaction parsing or month grouping for this period.",
                }
            )
        elif missing and not cfg.compare_only_common_fields:
            overview.append(
                {
                    "month": month,
                    "status": "SCHEMA_GAP",
                    "summary": f"Provided summary missing: {', '.join(missing)}",
                    "action": "Add missing fields upstream or enable compare_only_common_fields.",
                }
            )
        else:
            overview.append(
                {
                    "month": month,
                    "status": "OK",
                    "summary": "OK (all comparable fields match).",
                    "action": "",
                }
            )

    if cfg.compare_only_common_fields:
        missing_fields = []

    return value_mismatches, missing_fields, overview


def check_schema(data: Dict[str, Any], cfg: AuditConfig) -> Optional[CheckFinding]:
    required_top = ["summary", "monthly_summary", "transactions"]
    missing = [key for key in required_top if key not in data]
    if missing and cfg.strict_schema:
        return CheckFinding("schema.top_level", "fail", f"Missing top-level keys: {missing}")
    if missing:
        return CheckFinding("schema.top_level", "warn", f"Missing top-level keys (non-strict): {missing}")

    tx = data.get("transactions", [])
    if not isinstance(tx, list):
        return CheckFinding("schema.transactions_type", "fail", "`transactions` must be a list.")
    return None


def check_summary_consistency(data: Dict[str, Any], recomputed_summary: Dict[str, Any]) -> Optional[CheckFinding]:
    provided = data.get("summary", {})
    if not isinstance(provided, dict):
        return CheckFinding("summary.type", "fail", "`summary` must be a dict/object.")

    provided_total = provided.get("total_transactions")
    recomputed_total = recomputed_summary.get("total_transactions")
    if provided_total is not None and recomputed_total is not None:
        try:
            if int(provided_total) != int(recomputed_total):
                return CheckFinding(
                    "summary.total_transactions",
                    "fail",
                    f"Mismatch: provided={provided_total} recomputed={recomputed_total}",
                )
        except Exception:
            return CheckFinding(
                "summary.total_transactions",
                "warn",
                f"Could not compare totals: {provided_total} vs {recomputed_total}",
            )
    return None


def run_audit(data: Dict[str, Any], cfg: AuditConfig) -> AuditResult:
    checks_run = 0
    failed: List[CheckFinding] = []
    warning_checks: List[CheckFinding] = []

    checks_run += 1
    schema_finding = check_schema(data, cfg)
    if schema_finding:
        (failed if schema_finding.status == "fail" else warning_checks).append(schema_finding)

    normalized_tx, tx_notes = _normalize_transactions(data.get("transactions", []))
    normalized_monthly, monthly_notes = _normalize_monthly_rows(data.get("monthly_summary", []))
    normalization_notes = tx_notes + monthly_notes

    recomputed_summary, recomputed_monthly = recompute_summary_and_monthly(normalized_tx)

    checks_run += 1
    summary_finding = check_summary_consistency(data, recomputed_summary)
    if summary_finding:
        (failed if summary_finding.status == "fail" else warning_checks).append(summary_finding)

    val_mism, miss_fields, overview = compare_monthly_summary(normalized_monthly, recomputed_monthly, cfg)

    if val_mism:
        failed.append(
            CheckFinding(
                "monthly_summary.value_mismatch",
                "fail",
                f"Monthly numeric mismatches detected: {len(val_mism)} item(s).",
                examples=val_mism[: cfg.max_examples],
            )
        )

    if miss_fields and not cfg.compare_only_common_fields:
        warning_checks.append(
            CheckFinding(
                "monthly_summary.schema_gap",
                "warn",
                f"Monthly summary missing computed fields: {len(miss_fields)} item(s).",
                examples=miss_fields[: cfg.max_examples],
            )
        )

    return AuditResult(
        overall_pass=len(failed) == 0,
        checks_run=checks_run,
        checks_failed=len(failed),
        warnings=len(warning_checks),
        failed_checks=failed,
        warning_checks=warning_checks,
        recomputed_summary=recomputed_summary,
        recomputed_monthly_summary=recomputed_monthly,
        normalization_notes=normalization_notes,
        monthly_value_mismatches=val_mism,
        monthly_missing_fields=miss_fields,
        monthly_overview=overview,
    )


def audit_report_json(original: Dict[str, Any], result: AuditResult, cfg: AuditConfig) -> Dict[str, Any]:
    return {
        "audit_meta": {
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": {
                "amount_tolerance": cfg.amount_tolerance,
                "max_examples": cfg.max_examples,
                "strict_schema": cfg.strict_schema,
                "compare_only_common_fields": cfg.compare_only_common_fields,
            },
        },
        "audit_result": {
            "overall_pass": result.overall_pass,
            "checks_run": result.checks_run,
            "checks_failed": result.checks_failed,
            "warnings": result.warnings,
            "normalization_notes": result.normalization_notes,
            "failed_checks": [finding.__dict__ for finding in result.failed_checks],
            "warning_checks": [finding.__dict__ for finding in result.warning_checks],
        },
        "monthly_compare": {
            "overview": result.monthly_overview,
            "value_mismatches": result.monthly_value_mismatches,
            "missing_fields": result.monthly_missing_fields,
        },
        "recomputed": {
            "summary": result.recomputed_summary,
            "monthly_summary": result.recomputed_monthly_summary,
        },
        "input_snapshot": {
            "transactions_count": len(original.get("transactions") or []),
            "monthly_summary_first3": (original.get("monthly_summary") or [])[:3],
        },
    }
