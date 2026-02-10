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
    monthly_diffs: Optional[List[Dict[str, Any]]] = None


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

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

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
        return float(x)
    except Exception:
        return float("nan")


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _is_num(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _round2(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
        if not math.isfinite(f):
            return None
        return round(f, 2)
    except Exception:
        return None


# -----------------------------
# Recompute from transactions (bank-agnostic)
# -----------------------------
def recompute_summary_and_monthly(transactions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    df = pd.DataFrame(transactions)
    if df.empty:
        return (
            {"total_transactions": 0, "date_range": {"start": None, "end": None}},
            [],
        )

    df["__date"] = df["date"].apply(_parse_date_safe) if "date" in df.columns else None
    df["__debit"] = df["debit"].apply(_float_or_nan) if "debit" in df.columns else float("nan")
    df["__credit"] = df["credit"].apply(_float_or_nan) if "credit" in df.columns else float("nan")
    df["__balance"] = df["balance"].apply(_float_or_nan) if "balance" in df.columns else float("nan")
    df["__source_file"] = df.get("source_file", pd.Series([""] * len(df))).apply(_safe_str)

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

    debit0 = pd.Series(df_valid["__debit"]).fillna(0.0)
    credit0 = pd.Series(df_valid["__credit"]).fillna(0.0)

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
# Core checks
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
    if tx and not isinstance(tx[0], dict):
        return CheckFinding("schema.transactions_rows", "fail", "Each transaction must be an object/dict.")
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
                return CheckFinding(
                    "summary.total_transactions",
                    "fail",
                    f"`summary.total_transactions` mismatch: provided={pt} recomputed={rt}",
                )
        except Exception:
            return CheckFinding(
                "summary.total_transactions",
                "warn",
                f"Could not compare `total_transactions`: provided={pt} recomputed={rt}",
            )
    return None


# -----------------------------
# Monthly diff engine (professional, explicit)
# -----------------------------
def _index_by_month(rows: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("month"):
            out[str(r["month"])] = r
    return out


def build_monthly_diffs(
    provided_rows: Any,
    recomputed_rows: List[Dict[str, Any]],
    tol: float,
) -> List[Dict[str, Any]]:
    """
    Produces row-level diffs:
      - Missing month on either side
      - Missing field in provided
      - Numeric mismatches with delta & severity
    """
    prov = _index_by_month(provided_rows)
    reco = _index_by_month(recomputed_rows)

    months = sorted(set(prov.keys()) | set(reco.keys()))
    fields = ["transaction_count", "total_debit", "total_credit", "net_change"]

    diffs: List[Dict[str, Any]] = []

    for m in months:
        pr = prov.get(m)
        rr = reco.get(m)

        if pr is None:
            diffs.append(
                {"month": m, "field": "__month__", "status": "MISSING_IN_PROVIDED", "provided": None, "recomputed": rr}
            )
            continue

        if rr is None:
            diffs.append(
                {"month": m, "field": "__month__", "status": "MISSING_IN_RECOMPUTED", "provided": pr, "recomputed": None}
            )
            continue

        for f in fields:
            pv = pr.get(f, None)
            rv = rr.get(f, None)

            # Treat "missing in provided" explicitly (your screenshot shows None)
            if pv is None and rv is not None:
                diffs.append(
                    {
                        "month": m,
                        "field": f,
                        "status": "MISSING_FIELD_IN_PROVIDED",
                        "provided": None,
                        "recomputed": rv,
                        "delta": None,
                        "abs_delta": None,
                    }
                )
                continue

            # If recomputed missing (rare), flag it too
            if rv is None and pv is not None:
                diffs.append(
                    {
                        "month": m,
                        "field": f,
                        "status": "MISSING_FIELD_IN_RECOMPUTED",
                        "provided": pv,
                        "recomputed": None,
                        "delta": None,
                        "abs_delta": None,
                    }
                )
                continue

            # Both missing -> ignore
            if pv is None and rv is None:
                continue

            # transaction_count = exact compare
            if f == "transaction_count":
                try:
                    if int(pv) != int(rv):
                        diffs.append(
                            {
                                "month": m,
                                "field": f,
                                "status": "MISMATCH",
                                "provided": pv,
                                "recomputed": rv,
                                "delta": int(rv) - int(pv),
                                "abs_delta": abs(int(rv) - int(pv)),
                            }
                        )
                except Exception:
                    diffs.append(
                        {
                            "month": m,
                            "field": f,
                            "status": "MISMATCH",
                            "provided": pv,
                            "recomputed": rv,
                            "delta": None,
                            "abs_delta": None,
                        }
                    )
                continue

            # numeric compare with tolerance
            if _is_num(pv) and _is_num(rv):
                delta = float(rv) - float(pv)
                abs_delta = abs(delta)
                if abs_delta > tol:
                    # Severity for UI
                    sev = "HIGH" if abs_delta > max(100.0, 10 * tol) else "LOW"
                    diffs.append(
                        {
                            "month": m,
                            "field": f,
                            "status": "MISMATCH",
                            "severity": sev,
                            "provided": _round2(pv),
                            "recomputed": _round2(rv),
                            "delta": round(delta, 2),
                            "abs_delta": round(abs_delta, 2),
                        }
                    )
            else:
                # non-numeric mismatch
                if str(pv) != str(rv):
                    diffs.append(
                        {
                            "month": m,
                            "field": f,
                            "status": "MISMATCH",
                            "provided": pv,
                            "recomputed": rv,
                            "delta": None,
                            "abs_delta": None,
                        }
                    )

    return diffs


def check_monthly_summary(
    data: Dict[str, Any],
    recomputed_monthly: List[Dict[str, Any]],
    cfg: AuditConfig,
) -> Tuple[Optional[CheckFinding], List[Dict[str, Any]]]:
    provided = data.get("monthly_summary", [])

    diffs = build_monthly_diffs(provided, recomputed_monthly, cfg.amount_tolerance)

    # If the only diffs are "missing fields in provided", make it WARN not FAIL (common in your screenshot)
    hard_mismatches = [d for d in diffs if d.get("status") == "MISMATCH" or d.get("status", "").startswith("MISSING_IN_")]

    if not provided:
        return (CheckFinding("monthly_summary.missing", "warn", "No `monthly_summary` provided; cannot compare."), diffs)

    if diffs:
        # Determine severity: if there are true numeric mismatches -> fail; if only missing fields -> warn
        has_true_mismatch = any(d.get("status") == "MISMATCH" for d in diffs)
        status = "fail" if has_true_mismatch else "warn"

        msg = (
            f"Monthly summary differences found: {len(diffs)} items. "
            f"Tolerance={cfg.amount_tolerance}. "
            f"{'Includes true numeric mismatches.' if has_true_mismatch else 'Mostly missing fields in provided summary.'}"
        )
        return (
            CheckFinding(
                "monthly_summary.diff",
                status,
                msg,
                examples=diffs[: int(cfg.max_examples)],
            ),
            diffs,
        )

    return (None, diffs)


def check_duplicates_and_suspicious(data: Dict[str, Any], cfg: AuditConfig) -> List[CheckFinding]:
    findings: List[CheckFinding] = []
    tx = data.get("transactions", [])
    if not isinstance(tx, list) or not tx:
        return findings

    df = pd.DataFrame(tx)

    core = [c for c in ["date", "description", "debit", "credit", "balance", "source_file"] if c in df.columns]
    if core:
        dup = df[df.duplicated(subset=core, keep=False)]
        if not dup.empty:
            findings.append(
                CheckFinding(
                    "transactions.duplicates",
                    "warn",
                    f"Found {len(dup)} duplicated rows by core fields (showing up to {cfg.max_examples}).",
                    dup[core].head(int(cfg.max_examples)).to_dict(orient="records"),
                )
            )

    if "debit" in df.columns and "credit" in df.columns:
        df["__debit"] = df["debit"].apply(_float_or_nan).fillna(0.0)
        df["__credit"] = df["credit"].apply(_float_or_nan).fillna(0.0)

        both = df[(df["__debit"] > 0) & (df["__credit"] > 0)]
        if not both.empty:
            cols = [c for c in ["date", "description", "debit", "credit", "balance", "source_file", "page"] if c in df.columns]
            findings.append(
                CheckFinding(
                    "transactions.debit_and_credit",
                    "warn",
                    f"{len(both)} rows have both debit and credit > 0 (showing up to {cfg.max_examples}).",
                    both[cols].head(int(cfg.max_examples)).to_dict(orient="records"),
                )
            )

    return findings


# -----------------------------
# Orchestration
# -----------------------------
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

    checks_run += 1
    f, diffs = check_monthly_summary(data, recomputed_monthly, cfg)
    if f:
        (failed if f.status == "fail" else warnings).append(f)

    checks_run += 1
    for f2 in check_duplicates_and_suspicious(data, cfg):
        (failed if f2.status == "fail" else warnings).append(f2)

    return AuditResult(
        overall_pass=(len(failed) == 0),
        checks_run=checks_run,
        checks_failed=len(failed),
        warnings=len(warnings),
        failed_checks=failed,
        warning_checks=warnings,
        recomputed_summary=recomputed_summary,
        recomputed_monthly_summary=recomputed_monthly,
        monthly_diffs=diffs,
    )


def audit_report_json(original: Dict[str, Any], result: AuditResult, cfg: AuditConfig) -> Dict[str, Any]:
    return {
        "audit_meta": {
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": {
                "amount_tolerance": cfg.amount_tolerance,
                "max_examples": cfg.max_examples,
                "strict_schema": cfg.strict_schema,
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
        "recomputed": {
            "summary": result.recomputed_summary,
            "monthly_summary": result.recomputed_monthly_summary,
            "monthly_diffs": result.monthly_diffs,
        },
        "input_snapshot": {
            "summary": original.get("summary"),
            "monthly_summary_first3": (original.get("monthly_summary") or [])[:3],
            "transactions_count": len(original.get("transactions") or []),
        },
    }
