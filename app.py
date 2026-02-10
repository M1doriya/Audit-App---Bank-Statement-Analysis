from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

from audit_checks import AuditConfig, audit_report_json, run_audit

st.set_page_config(page_title="Statement Audit", layout="wide")

st.markdown("## Statement Audit Console")
st.caption("Clarity-first audit: separates *real mismatches* from *missing fields (schema gaps)*.")

with st.sidebar:
    st.header("Input")
    input_mode = st.radio(
        "Choose input mode",
        ["Fetch by URL (recommended)", "Paste JSON (recommended)"],
        index=0,
        key="input_mode",
    )
    url = st.text_input("JSON URL", placeholder="https://.../full_report.json", key="json_url")

    st.divider()
    st.header("Audit Settings")
    amount_tolerance = st.number_input("Amount tolerance (RM)", min_value=0.0, value=0.01, step=0.01)
    max_examples = st.number_input("Max examples per section", min_value=5, value=30, step=5)
    strict_schema = st.checkbox("Strict schema", value=False)

    compare_only_common_fields = st.checkbox(
        "Compare only common fields (recommended)",
        value=True,
        help="When ON: if provided monthly_summary doesn't include transaction_count/net_change, we won't treat that as an issue.",
    )

    cfg = AuditConfig(
        amount_tolerance=float(amount_tolerance),
        max_examples=int(max_examples),
        strict_schema=bool(strict_schema),
        compare_only_common_fields=bool(compare_only_common_fields),
    )

data: Optional[Dict[str, Any]] = None
source: Optional[str] = None

def _load_url(u: str) -> Dict[str, Any]:
    r = requests.get(u, timeout=30)
    r.raise_for_status()
    return r.json()

try:
    if st.session_state.input_mode.startswith("Fetch"):
        if not url.strip():
            st.info("Paste a JSON URL in the sidebar.")
            st.stop()
        if st.button("Fetch JSON", type="primary"):
            data = _load_url(url.strip())
            source = f"url:{url.strip()}"
        else:
            st.stop()
    else:
        pasted = st.text_area("Paste full JSON", height=260)
        if st.button("Load JSON", type="primary"):
            data = json.loads(pasted)
            source = "pasted-json"
        else:
            st.stop()

except Exception as e:
    st.error("Failed to load JSON.")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

st.success(f"Loaded: {source}")

with st.spinner("Running audit..."):
    result = run_audit(data, cfg)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall", "PASS ✅" if result.overall_pass else "FAIL ❌")
c2.metric("Checks run", result.checks_run)
c3.metric("Failed checks", result.checks_failed)
c4.metric("Warnings", result.warnings)

st.divider()

# -----------------------------
# Monthly clarity section
# -----------------------------
st.markdown("## Monthly Summary: Clear Results")

# Explain meaning once
st.info(
    "**Important:** If you see “Missing field”, it does **not** mean wrong numbers. "
    "It means your provided `monthly_summary` simply **doesn’t include** that field, so it cannot be compared."
)

overview_df = pd.DataFrame(result.monthly_overview or [])
if not overview_df.empty:
    st.markdown("### Month Overview (one line per month)")
    st.dataframe(overview_df, use_container_width=True, height=280)
else:
    st.warning("No monthly overview generated.")

tab1, tab2 = st.tabs(["❌ Real mismatches (action required)", "ℹ️ Missing fields (schema gap)"])

with tab1:
    mism = pd.DataFrame(result.monthly_value_mismatches or [])
    if mism.empty:
        st.success("No real numeric mismatches found.")
    else:
        st.error("These are **real** differences between provided and recomputed values.")
        st.dataframe(mism, use_container_width=True, height=360)

with tab2:
    miss = pd.DataFrame(result.monthly_missing_fields or [])
    if miss.empty:
        st.success("No missing-field items (or they are ignored because 'Compare only common fields' is ON).")
    else:
        st.warning(
            "These are **not errors**. They indicate your provided monthly_summary schema does not include computed fields."
        )
        st.dataframe(miss, use_container_width=True, height=360)

st.divider()

st.markdown("## Monthly: Provided vs Recomputed")
left, right = st.columns(2)
with left:
    st.caption("Provided monthly_summary (as-is from JSON)")
    st.dataframe(pd.DataFrame(data.get("monthly_summary", [])), use_container_width=True, height=260)

with right:
    st.caption("Recomputed monthly summary (from transactions)")
    st.dataframe(pd.DataFrame(result.recomputed_monthly_summary or []), use_container_width=True, height=260)

st.divider()

st.markdown("## Findings")
if result.failed_checks:
    st.subheader("Failed checks")
    for fc in result.failed_checks:
        st.error(f"{fc.name}: {fc.message}")
else:
    st.success("No failed checks.")

if result.warning_checks:
    st.subheader("Warnings")
    for wc in result.warning_checks:
        st.warning(f"{wc.name}: {wc.message}")

st.divider()

st.markdown("## Export")
out = audit_report_json(data, result, cfg)
st.download_button(
    "Download audit_report.json",
    data=json.dumps(out, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name="audit_report.json",
    mime="application/json",
)
