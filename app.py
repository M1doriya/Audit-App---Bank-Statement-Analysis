from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from audit_checks import (
    AuditConfig,
    AuditResult,
    audit_report_json,
    run_audit,
)


st.set_page_config(page_title="Bank Report Audit", layout="wide")

st.title("Bank Statement Output Audit")
st.caption(
    "Upload a `full_report.json` (output from your bank-statement app) or paste a URL to fetch it, "
    "then this app validates internal consistency (monthly totals, balances, duplicates, schema)."
)

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Choose input", ["Upload JSON", "Fetch by URL"], index=0)

    cfg = AuditConfig(
        amount_tolerance=st.number_input("Amount tolerance (RM)", min_value=0.0, value=0.01, step=0.01),
        max_examples=st.number_input("Max examples per check", min_value=5, value=30, step=5),
        strict_schema=st.checkbox("Strict schema", value=False),
    )

    st.divider()
    st.header("Export")
    st.caption("You can download an audit report JSON after running checks.")


def _load_json_from_upload(uploaded) -> Dict[str, Any]:
    raw = uploaded.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        # Try latin-1 as fallback (rare)
        return json.loads(raw.decode("latin-1"))


def _load_json_from_url(url: str, timeout: int = 25) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    # Accept raw JSON or text/json
    return r.json()


def _parse_input() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if input_mode == "Upload JSON":
        uploaded = st.file_uploader("Upload full_report.json", type=["json"])
        if not uploaded:
            return None, None
        try:
            data = _load_json_from_upload(uploaded)
            return data, f"upload:{uploaded.name}"
        except Exception as e:
            return None, f"Failed to parse uploaded JSON: {e}"
    else:
        url = st.text_input("Paste URL to report JSON (must be publicly accessible)")
        if not url.strip():
            return None, None
        try:
            data = _load_json_from_url(url.strip())
            return data, f"url:{url.strip()}"
        except Exception as e:
            return None, f"Failed to fetch/parse JSON from URL: {e}"


data, err = _parse_input()
if err:
    st.error(err)
    st.stop()

if not data:
    st.info("Provide a JSON input to run the audit.")
    st.stop()

with st.spinner("Running audit checks..."):
    result: AuditResult = run_audit(data, cfg)

# --- Summary row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall", "PASS ✅" if result.overall_pass else "FAIL ❌")
c2.metric("Checks run", str(result.checks_run))
c3.metric("Checks failed", str(result.checks_failed))
c4.metric("Warnings", str(result.warnings))

st.divider()

# --- High-level issues
if result.failed_checks:
    st.subheader("Failed checks")
    for fc in result.failed_checks:
        with st.expander(f"❌ {fc.name}"):
            st.write(fc.message)
            if fc.examples:
                st.dataframe(pd.DataFrame(fc.examples))
else:
    st.success("No failed checks detected.")

if result.warning_checks:
    st.subheader("Warnings")
    for wc in result.warning_checks:
        with st.expander(f"⚠️ {wc.name}"):
            st.write(wc.message)
            if wc.examples:
                st.dataframe(pd.DataFrame(wc.examples))

st.divider()

# --- Comparisons: monthly summary
st.subheader("Monthly summary comparison")
left, right = st.columns(2)

provided_monthly = data.get("monthly_summary", [])
recomputed_monthly = result.recomputed_monthly_summary or []

with left:
    st.caption("Provided `monthly_summary`")
    if provided_monthly:
        st.dataframe(pd.DataFrame(provided_monthly))
    else:
        st.info("No `monthly_summary` present in input JSON.")

with right:
    st.caption("Recomputed monthly summary (from transactions)")
    if recomputed_monthly:
        st.dataframe(pd.DataFrame(recomputed_monthly))
    else:
        st.info("Could not recompute monthly summary (missing or invalid transactions).")

if result.monthly_diffs:
    st.subheader("Monthly diffs (provided vs recomputed)")
    st.dataframe(pd.DataFrame(result.monthly_diffs))

st.divider()

# --- Transactions overview
st.subheader("Transactions overview")
tx = data.get("transactions", [])
if not isinstance(tx, list) or not tx:
    st.warning("No transactions found.")
else:
    df = pd.DataFrame(tx)
    st.caption(f"{len(df)} transactions")
    st.dataframe(df.head(200))

st.divider()

# --- Download audit report
audit_json = audit_report_json(data, result, cfg)
st.download_button(
    "Download audit_report.json",
    data=json.dumps(audit_json, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name="audit_report.json",
    mime="application/json",
)

st.caption("Tip: Fix your parser bugs by using the failed-check examples (page, source_file, balances) to reproduce.")
