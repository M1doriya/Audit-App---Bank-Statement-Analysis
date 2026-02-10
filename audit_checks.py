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


def _isclose(a: Any, b: Any, tol: float) -> bool:
    try:
        fa = float(a)
        fb = float(b)
        return math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= tol
    except Exception:
        return False


def _parse_date_safe(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    s = str(x).strip()
    if not s:
        return None

    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
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

    df_valid["__page"] = df_valid["page"].apply(_float_or_nan) if "page" in df_valid.columns else float("nan")
    df_valid["__idx"] = range(len(df_valid))
    df_valid = df_valid.sort_values(["__month", "__date", "__page", "__idx"], ascending=True)

    rows: List[Dict[str, Any]] = []
    for m, g in df_valid.groupby("__month", sort=True):
        debit_sum = float(pd.Series(g["__debit"]).fillna(0).sum())
        credit_sum = float(pd.Series(g["__credit"]).fillna(0).sum())

        bal_series = pd.Series(g["__balance"]).dropna()
        ending_balance = float(bal_series.iloc[-1]) if not bal_series.empty else float("nan")
        lowest = float(bal_series.min()) if not bal_series.empty else float("nan")
        highest = float(bal_series.max()) if not bal_series.empty else float("nan")

        files = sorted({f for f in g["__source_file"].tolist() if f})

        rows.append(
            {
                "month": m,
                "transaction_count": int(len(g)),
                "total_debit": round(debit_sum, 2),
                "total_credit": round(credit_sum, 2),
                "net_change": round(credit_sum - debit_sum, 2),
                "ending_balance": None if math.isnan(ending_balance) else round(ending_balance, 2),
                "lowest_balance": None if math.isnan(lowest) else round(lowest, 2),
                "highest_balance": None if math.isnan(highest) else round(highest, 2),
                "source_files": ", ".join(files),
            }
        )

    return summary, rows


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

    if tx and isinstance(tx[0], dict):
        expected = ["date", "description", "debit", "credit", "balance", "page", "bank", "source_file"]
        miss = [f for f in expected if f not in tx[0]]
        if miss and cfg.strict_schema:
            return CheckFinding("schema.transaction_fields", "fail", f"Missing transaction fields: {miss}")
        elif miss:
            return CheckFinding("schema.transaction_fields", "warn", f"Missing transaction fields (non-strict): {miss}")

    return None


def check_summary_consistency(data: Dict[str, Any], recomputed_summary: Dict[str, Any]) -> Optional[CheckFinding]:
    provided = data.get("summary", {})
    if not isinstance(provided, dict):
        return CheckFinding("summary.type", "fail", "`summary` must be a dict/object.")

    pt = provided.get("total_transactions")
    rt = recomputed_summary.get("total_transactions")
    if pt is not None and rt is not None and int(pt) != int(rt):
        return CheckFinding(
            "summary.total_transactions",
            "fail",
            f"`summary.total_transactions` mismatch: provided={pt} recomputed={rt}",
        )
    return None


def check_monthly_summary(
    data: Dict[str, Any],
    recomputed_monthly: List[Dict[str, Any]],
    cfg: AuditConfig,
) -> Tuple[Optional[CheckFinding], List[Dict[str, Any]]]:
    provided = data.get("monthly_summary", [])
    if not provided:
        return (
            CheckFinding("monthly_summary.missing", "warn", "No `monthly_summary` provided; cannot compare."),
            [],
        )

    if not isinstance(provided, list):
        return (CheckFinding("monthly_summary.type", "fail", "`monthly_summary` must be a list."), [])

    p_df = pd.DataFrame(provided)
    r_df = pd.DataFrame(recomputed_monthly)

    diffs: List[Dict[str, Any]] = []
    all_months = sorted(set(p_df.get("month", pd.Series([], dtype=str)).tolist()) | set(r_df.get("month", pd.Series([], dtype=str)).tolist()))

    fields = ["transaction_count", "total_debit", "total_credit", "net_change", "ending_balance", "lowest_balance", "highest_balance"]

    def row_by_month(df: pd.DataFrame, m: str) -> Dict[str, Any]:
        if df.empty or "month" not in df.columns:
            return {}
        sub = df[df["month"] == m]
        if sub.empty:
            return {}
        return sub.iloc[0].to_dict()

    for m in all_months:
        pr = row_by_month(p_df, m)
        rr = row_by_month(r_df, m)
        if not pr or not rr:
            diffs.append({"month": m, "issue": "month_missing", "provided_exists": bool(pr), "recomputed_exists": bool(rr)})
            continue

        for f in fields:
            pv = pr.get(f)
            rv = rr.get(f)
            if f == "transaction_count":
                try:
                    if int(pv) != int(rv):
                        diffs.append({"month": m, "field": f, "provided": pv, "recomputed": rv})
                except Exception:
                    diffs.append({"month": m, "field": f, "provided": pv, "recomputed": rv})
            else:
                if pv is None and rv is None:
                    continue
                if pv is None or rv is None:
                    diffs.append({"month": m, "field": f, "provided": pv, "recomputed": rv})
                elif not _isclose(pv, rv, cfg.amount_tolerance):
                    diffs.append({"month": m, "field": f, "provided": pv, "recomputed": rv})

    if diffs:
        return (
            CheckFinding(
                "monthly_summary.diff",
                "fail",
                f"Monthly summary differs from recomputation in {len(diffs)} places (showing up to {cfg.max_examples}).",
                diffs[: int(cfg.max_examples)],
            ),
            diffs,
        )
    return (None, diffs)


def check_balance_continuity(data: Dict[str, Any], cfg: AuditConfig) -> Optional[CheckFinding]:
    tx = data.get("transactions", [])
    if not isinstance(tx, list) or not tx:
        return None

    df = pd.DataFrame(tx)
    needed = {"debit", "credit", "balance", "source_file"}
    if not needed.issubset(set(df.columns)):
        return CheckFinding(
            "balance_continuity.skipped",
            "warn",
            f"Skipping balance continuity (missing: {sorted(list(needed - set(df.columns)))})",
        )

    df["__date"] = df["date"].apply(_parse_date_safe) if "date" in df.columns else None
    df["__page"] = df["page"].apply(_float_or_nan) if "page" in df.columns else float("nan")
    df["__debit"] = df["debit"].apply(_float_or_nan).fillna(0.0)
    df["__credit"] = df["credit"].apply(_float_or_nan).fillna(0.0)
    df["__balance"] = df["balance"].apply(_float_or_nan)
    df["__source_file"] = df["source_file"].apply(_safe_str)
    df["__bank"] = df["bank"].apply(_safe_str) if "bank" in df.columns else ""

    df["__idx"] = range(len(df))
    df = df.sort_values(["__source_file", "__date", "__page", "__idx"], ascending=True)

    mismatches: List[Dict[str, Any]] = []
    for (sf, bank), g in df.groupby(["__source_file", "__bank"], sort=False):
        g = g.reset_index(drop=True)

        prev_bal = None
        prev_row = None
        for i in range(len(g)):
            bal = g.loc[i, "__balance"]
            if not math.isfinite(bal):
                prev_bal = None
                prev_row = None
                continue

            if prev_bal is not None and prev_row is not None:
                expected = float(prev_bal + g.loc[i, "__credit"] - g.loc[i, "__debit"])
                if abs(expected - bal) > cfg.amount_tolerance:
                    mismatches.append(
                        {
                            "source_file": sf,
                            "bank": bank,
                            "prev_date": _safe_str(prev_row.get("date")),
                            "prev_balance": float(prev_bal),
                            "date": _safe_str(g.loc[i, "date"]),
                            "debit": float(g.loc[i, "__debit"]),
                            "credit": float(g.loc[i, "__credit"]),
                            "expected_balance": round(expected, 2),
                            "actual_balance": round(float(bal), 2),
                            "delta": round(float(bal - expected), 2),
                            "page": g.loc[i, "page"] if "page" in g.columns else None,
                            "description": g.loc[i, "description"] if "description" in g.columns else None,
                        }
                    )
                    if len(mismatches) >= int(cfg.max_examples):
                        break

            prev_bal = float(bal)
            prev_row = g.loc[i].to_dict()

        if len(mismatches) >= int(cfg.max_examples):
            break

    if mismatches:
        return CheckFinding(
            "balance_continuity.mismatch",
            "fail",
            f"Balance continuity mismatches found (showing up to {cfg.max_examples}).",
            mismatches,
        )
    return None


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
            cols = [c for c in ["date", "description", "debit", "credit", "balance", "source_file"] if c in df.columns]
            findings.append(
                CheckFinding(
                    "transactions.debit_and_credit",
                    "warn",
                    f"{len(both)} rows have both debit and credit > 0 (showing up to {cfg.max_examples}).",
                    both[cols].head(int(cfg.max_examples)).to_dict(orient="records"),
                )
            )

        neg = df[(df["__debit"] < 0) | (df["__credit"] < 0)]
        if not neg.empty:
            cols = [c for c in ["date", "description", "debit", "credit", "balance", "source_file"] if c in df.columns]
            findings.append(
                CheckFinding(
                    "transactions.negative_amounts",
                    "warn",
                    f"{len(neg)} rows have negative debit/credit values (showing up to {cfg.max_examples}).",
                    neg[cols].head(int(cfg.max_examples)).to_dict(orient="records"),
                )
            )

    return findings


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
    f = check_balance_continuity(data, cfg)
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
