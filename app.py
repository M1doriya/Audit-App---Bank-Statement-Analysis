from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

from audit_checks import AuditConfig, audit_report_json, run_audit

st.set_page_config(page_title="Bank Report Audit", layout="wide")
st.title("Bank Statement Output Audit")
st.caption(
    "Upload `full_report.json` (from your bank statement app) or fetch it by URL. "
    "This tool validates internal consistency: monthly totals, balances continuity, duplicates, and schema."
)

# =========================================================
# SIDEBAR (IMPORTANT): uploader is ALWAYS rendered here
# =========================================================
with st.sidebar:
    st.header("Input")

    input_mode = st.radio(
        "Choose input",
        ["Upload JSON", "Fetch by URL"],
        index=0,
        key="input_mode",
    )

    # Always render uploader (never behind st.stop)
    uploaded = st.file_uploader(
        "Upload full_report.json",
        type=["json"],
        key="json_uploader",
        help="If upload does nothing, check Railway HTTP logs. You should see a POST request when this works.",
    )

    url = st.text_input(
        "Fetch by URL (optional)",
        placeholder="https://.../full_report.json",
        key="json_url",
    )

    st.divider()
    st.header("Audit Settings")
    amount_tolerance = st.number_input("Amount tolerance (RM)", min_value=0.0, value=0.01, step=0.01)
    max_examples = st.number_input("Max examples per check (display only)", min_value=5, value=30, step=5)
    strict_schema = st.checkbox("Strict schema", value=False)

    cfg = AuditConfig(
        amount_tolerance=float(amount_tolerance),
        max_examples=int(max_examples),
        strict_schema=bool(strict_schema),
    )

    st.divider()
    st.header("Export")
    st.caption("You can download an audit report JSON after running checks.")

# =========================================================
# LOAD INPUT (no st.stop before uploader)
# =========================================================
data: Optional[Dict[str, Any]] = None
source: Optional[str] = None
load_error: Optional[str] = None

if st.session_state.input_mode == "Upload JSON":
    if uploaded is not None:
        try:
            # confirm immediately so you know it really uploaded
            st.sidebar.success(f"Received: {uploaded.name} ({uploaded.size} bytes)")
            raw = uploaded.read()
            try:
                data = json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                data = json.loads(raw.decode("latin-1"))
            source = f"upload:{uploaded.name}"
        except Exception as e:
            load_error = f"Failed to parse uploaded JSON: {e}"
else:
    u = (st.session_state.json_url or "").strip()
    if u:
        try:
            r = requests.get(u, timeout=30)
            r.raise_for_status()
            data = r.json()
            source = f"url:{u}"
            st.sidebar.success("Fetched JSON from URL.")
        except Exception as e:
            load_error = f"Failed to fetch/parse JSON from URL: {e}"

if load_error:
    st.error(load_error)
    st.stop()

if data is None:
    st.info("Use the sidebar: Upload a JSON or choose Fetch by URL.")
    st.stop()

st.success(f"Loaded report from: {source}")

# =========================================================
# RUN AUDIT
# =========================================================
with st.spinner("Running audit checks..."):
    result = run_audit(data, cfg)

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall", "PASS ✅" if result.overall_pass else "FAIL ❌")
c2.metric("Checks run", str(result.checks_run))
c3.metric("Checks failed", str(result.checks_failed))
c4.metric("Warnings", str(result.warnings))

st.divider()

# Failed checks
if result.failed_checks:
    st.subheader("Failed checks")
    for fc in result.failed_checks:
        with st.expander(f"❌ {fc.name}"):
            st.write(fc.message)
            if fc.examples:
                st.dataframe(pd.DataFrame(fc.examples))
else:
    st.success("No failed checks detected.")

# Warnings
if result.warning_checks:
    st.subheader("Warnings")
    for wc in result.warning_checks:
        with st.expander(f"⚠️ {wc.name}"):
            st.write(wc.message)
            if wc.examples:
                st.dataframe(pd.DataFrame(wc.examples))

st.divider()

# Monthly summary comparison
st.subheader("Monthly summary comparison")
left, right = st.columns(2)

provided_monthly = data.get("monthly_summary", [])
recomputed_monthly = result.recomputed_monthly_summary or []

with left:
    st.caption("Provided `monthly_summary`")
    if isinstance(provided_monthly, list) and provided_monthly:
        st.dataframe(pd.DataFrame(provided_monthly))
    else:
        st.info("No `monthly_summary` present in input.")

with right:
    st.caption("Recomputed monthly summary (from transactions)")
    if recomputed_monthly:
        st.dataframe(pd.DataFrame(recomputed_monthly))
    else:
        st.info("Could not recompute monthly summary (missing/invalid transactions).")

if result.monthly_diffs:
    st.subheader("Monthly diffs (provided vs recomputed)")
    st.dataframe(pd.DataFrame(result.monthly_diffs))

st.divider()

# Transactions preview
st.subheader("Transactions preview")
tx = data.get("transactions", [])
if isinstance(tx, list) and tx:
    st.caption(f"{len(tx)} transactions (showing first 200)")
    st.dataframe(pd.DataFrame(tx).head(200))
else:
    st.warning("No transactions found in the report JSON.")

st.divider()

# Download audit report
out = audit_report_json(data, result, cfg)
st.download_button(
    "Download audit_report.json",
    data=json.dumps(out, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name="audit_report.json",
    mime="application/json",
)
