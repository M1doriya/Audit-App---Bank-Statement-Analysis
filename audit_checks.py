from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class AuditConfig:
    amount_tolerance: float = 0.01
    max_examples: int = 30
    strict_schema: bool = False
    compare_only_common_fields: bool = True  # ✅ new


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

    # ✅ new: structured monthly comparison outputs
    monthly_value_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    monthly_missing_fields: List[Dict[str, Any]] = field(default_factory=list)
    monthly_overview: List[Dict[str, Any]] = field(default_factory=list)


# -----------------------------
# Helpers
# -----------------------------
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
            cleaned = re.sub(r"[^0-9.\-]", "", x)
            if cleaned in {"", "-", ".", "-."}:
                return float("nan")
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


def _index_by_month(rows: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if isinstance(r, dict):
            month = _extract_month_value(r)
            if month:
                out[month] = r
    return out


def _extract_first(d: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in d and d.get(key) not in (None, ""):
            return d.get(key)
    return None


def _extract_month_value(row: Dict[str, Any]) -> Optional[str]:
    raw_month = _extract_first(row, ["month", "period", "month_year", "year_month", "statement_month"])
    if raw_month is None:
        return None
    dt = _parse_date_safe(raw_month)
    if dt:
        return _month_key(dt)

    text = str(raw_month).strip()
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{2}/\d{4}", text):
        mm, yyyy = text.split("/")
        return f"{yyyy}-{mm}"
    return text


def _normalize_transaction(tx: Dict[str, Any]) -> Dict[str, Any]:
    date_value = _extract_first(tx, ["date", "transaction_date", "posting_date", "value_date", "txn_date"])
    debit_value = _extract_first(tx, ["debit", "withdrawal", "money_out", "debit_amount"])
    credit_value = _extract_first(tx, ["credit", "deposit", "money_in", "credit_amount"])

    amount_value = _extract_first(tx, ["amount", "transaction_amount", "amt"])
    tx_type = str(_extract_first(tx, ["type", "transaction_type", "dr_cr", "direction"]) or "").strip().lower()

    debit = _float_or_nan(debit_value)
    credit = _float_or_nan(credit_value)

    if not _is_num(debit) and not _is_num(credit) and amount_value is not None:
        amt = _float_or_nan(amount_value)
        if _is_num(amt):
            if tx_type in {"debit", "dr", "withdrawal", "out", "expense"}:
                debit = abs(amt)
                credit = 0.0
            elif tx_type in {"credit", "cr", "deposit", "in", "income"}:
                debit = 0.0
                credit = abs(amt)
            elif amt < 0:
                debit = abs(amt)
                credit = 0.0
            else:
                debit = 0.0
                credit = abs(amt)

    return {
        "date": date_value,
        "debit": None if not _is_num(debit) else debit,
        "credit": None if not _is_num(credit) else credit,
    }


def _extract_monthly_field(row: Dict[str, Any], field_name: str) -> Any:
    aliases = {
        "transaction_count": ["transaction_count", "count", "total_transactions", "transactions"],
        "total_debit": ["total_debit", "debit", "debit_total", "withdrawal_total", "outflow"],
        "total_credit": ["total_credit", "credit", "credit_total", "deposit_total", "inflow"],
        "net_change": ["net_change", "net", "net_amount", "balance_change"],
    }
    return _extract_first(row, aliases.get(field_name, [field_name]))


# -----------------------------
# Recompute monthly from transactions
# -----------------------------
def recompute_summary_and_monthly(transactions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    normalized_tx = [_normalize_transaction(tx) for tx in transactions if isinstance(tx, dict)]
    df = pd.DataFrame(normalized_tx)
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
    for m, g in df_valid.groupby("__month", sort=True):
        d = float(pd.Series(g["__debit"]).fillna(0.0).sum())
        c = float(pd.Series(g["__credit"]).fillna(0.0).sum())
        rows.append(
            {
                "month": m,
                "transaction_count": int(len(g)),
                "total_debit": round(d, 2),
                "total_credit": round(c, 2),
                "net_change": round(c - d, 2),
            }
        )

    return summary, rows


# -----------------------------
# Monthly comparison (clarity-first)
# -----------------------------
def compare_monthly_summary(
    provided_rows: Any,
    recomputed_rows: List[Dict[str, Any]],
    cfg: AuditConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      - value_mismatches: REAL mismatches (numbers differ)
      - missing_fields: schema gaps (field absent in provided)
      - overview: one row per month with a human summary
    """
    prov = _index_by_month(provided_rows)
    reco = _index_by_month(recomputed_rows)

    months = sorted(set(prov.keys()) | set(reco.keys()))
    computed_fields = ["transaction_count", "total_debit", "total_credit", "net_change"]

    value_mismatches: List[Dict[str, Any]] = []
    missing_fields: List[Dict[str, Any]] = []
    overview: List[Dict[str, Any]] = []

    for m in months:
        pr = prov.get(m) or {}
        rr = reco.get(m) or {}

        if not pr and rr:
            overview.append(
                {
                    "month": m,
                    "status": "MISSING_MONTH_IN_PROVIDED",
                    "summary": "Provided monthly_summary has no row for this month.",
                    "action": "Your bank app did not output this month in monthly_summary. Check month grouping logic.",
                }
            )
            continue

        if pr and not rr:
            overview.append(
                {
                    "month": m,
                    "status": "MISSING_MONTH_IN_RECOMPUTED",
                    "summary": "No transactions found for this month (recomputed side missing).",
                    "action": "Either transactions are missing dates or month extraction failed.",
                }
            )
            continue

        # both exist
        missing = []
        mism = []

        for f in computed_fields:
            pv = _extract_monthly_field(pr, f)
            rv = rr.get(f, None)

            if pv is None:
                missing.append(f)
                missing_fields.append(
                    {
                        "month": m,
                        "field": f,
                        "meaning": "This field is not present in provided monthly_summary (schema gap).",
                        "recomputed_value": rv,
                        "action": "Either add this field to monthly_summary output OR enable compare_only_common_fields.",
                    }
                )
                continue

            # Compare if field exists in provided
            if f == "transaction_count":
                try:
                    if int(pv) != int(rv):
                        mism.append(f)
                        value_mismatches.append(
                            {
                                "month": m,
                                "field": f,
                                "provided": int(pv),
                                "recomputed": int(rv),
                                "delta": int(rv) - int(pv),
                                "meaning": "Transaction count differs.",
                            }
                        )
                except Exception:
                    mism.append(f)
                    value_mismatches.append(
                        {
                            "month": m,
                            "field": f,
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
                        mism.append(f)
                        value_mismatches.append(
                            {
                                "month": m,
                                "field": f,
                                "provided": _round2(pv),
                                "recomputed": _round2(rv),
                                "delta": round(delta, 2),
                                "abs_delta": round(abs(delta), 2),
                                "meaning": f"{f} differs beyond tolerance (±{cfg.amount_tolerance}).",
                            }
                        )
                else:
                    if str(pv) != str(rv):
                        mism.append(f)
                        value_mismatches.append(
                            {
                                "month": m,
                                "field": f,
                                "provided": pv,
                                "recomputed": rv,
                                "delta": None,
                                "meaning": f"{f} differs (non-numeric).",
                            }
                        )

        # Month-level summary text (clarity)
        if mism:
            overview.append(
                {
                    "month": m,
                    "status": "VALUE_MISMATCH",
                    "summary": f"Real mismatch in: {', '.join(mism)}",
                    "action": "Investigate month grouping or transaction parsing for this month.",
                }
            )
        elif missing and not cfg.compare_only_common_fields:
            overview.append(
                {
                    "month": m,
                    "status": "SCHEMA_GAP",
                    "summary": f"Provided monthly_summary is missing fields: {', '.join(missing)}",
                    "action": "Add these computed fields to monthly_summary OR turn on compare_only_common_fields.",
                }
            )
        else:
            # If compare_only_common_fields=True, missing fields are not treated as an issue
            overview.append(
                {
                    "month": m,
                    "status": "OK" if not mism else "VALUE_MISMATCH",
                    "summary": "OK (comparable fields match)."
                    if cfg.compare_only_common_fields
                    else "OK.",
                    "action": "",
                }
            )

    # If compare_only_common_fields=True, schema gaps should not clutter the UI
    if cfg.compare_only_common_fields:
        missing_fields = []

    return value_mismatches, missing_fields, overview


# -----------------------------
# Other checks
# -----------------------------
def check_schema(data: Dict[str, Any], cfg: AuditConfig) -> Optional[CheckFinding]:
    required_top = ["summary", "monthly_summary", "transactions"]
    missing = [k for k in required_top if k not in data]
    if missing and cfg.strict_schema:
        return CheckFinding("schema.top_level", "fail", f"Missing top-level keys: {missing}")
    elif missing:
        return CheckFinding("schema.top_level", "warn", f"Missing top-level keys (non-strict): {missing}")

    tx = data.get("transactions", [])
    if not isinstance(tx, list):
        return CheckFinding("schema.transactions_type", "fail", "`transactions` must be a list.")
    return None


def check_summary_consistency(data: Dict[str, Any], recomputed_summary: Dict[str, Any]) -> Optional[CheckFinding]:
    provided = data.get("summary", {})
    if not isinstance(provided, dict):
        return CheckFinding("summary.type", "fail", "`summary` must be a dict/object.")
    pt = provided.get("total_transactions")
    rt = recomputed_summary.get("total_transactions")
    if pt is not None and rt is not None:
        try:
            if int(pt) != int(rt):
                return CheckFinding("summary.total_transactions", "fail", f"Mismatch: provided={pt} recomputed={rt}")
        except Exception:
            return CheckFinding("summary.total_transactions", "warn", f"Could not compare totals: {pt} vs {rt}")
    return None


def run_audit(data: Dict[str, Any], cfg: AuditConfig) -> AuditResult:
    checks_run = 0
    failed: List[CheckFinding] = []
    warnings: List[CheckFinding] = []

    checks_run += 1
    f = check_schema(data, cfg)
    if f:
        (failed if f.status == "fail" else warnings).append(f)

    tx = data.get("transactions", [])
    recomputed_summary, recomputed_monthly = recompute_summary_and_monthly(tx if isinstance(tx, list) else [])

    checks_run += 1
    f = check_summary_consistency(data, recomputed_summary)
    if f:
        (failed if f.status == "fail" else warnings).append(f)

    # Monthly compare (clarity-first)
    val_mism, miss_fields, overview = compare_monthly_summary(data.get("monthly_summary", []), recomputed_monthly, cfg)

    if val_mism:
        warnings.append(
            CheckFinding(
                "monthly_summary.value_mismatch",
                "fail",
                f"Monthly numeric mismatches detected: {len(val_mism)} items.",
                examples=val_mism[: cfg.max_examples],
            )
        )

    if miss_fields and not cfg.compare_only_common_fields:
        warnings.append(
            CheckFinding(
                "monthly_summary.schema_gap",
                "warn",
                f"Monthly summary is missing computed fields (schema gap): {len(miss_fields)} items.",
                examples=miss_fields[: cfg.max_examples],
            )
        )

    # overall = fail only if there are true mismatches or hard failures
    overall_pass = len([x for x in failed]) == 0 and len(val_mism) == 0

    return AuditResult(
        overall_pass=overall_pass,
        checks_run=checks_run,
        checks_failed=len(failed) + (1 if val_mism else 0),
        warnings=len(warnings),
        failed_checks=failed,
        warning_checks=warnings,
        recomputed_summary=recomputed_summary,
        recomputed_monthly_summary=recomputed_monthly,
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
            "failed_checks": [f.__dict__ for f in result.failed_checks],
            "warning_checks": [w.__dict__ for w in result.warning_checks],
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
