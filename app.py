from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

from audit_checks import AuditConfig, audit_report_json, run_audit

st.set_page_config(page_title="Bank Statement Accuracy Checker", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem;}
    .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Bank Statement Accuracy Checker")
st.caption(
    "Second-layer audit to validate JSON statement outputs across multiple bank formats "
    "while preserving your original checking parameters."
)

with st.sidebar:
    st.header("Data Source")
    input_mode = st.radio(
        "Input mode",
        ["Fetch by URL (recommended)", "Paste JSON"],
        index=0,
        key="input_mode",
    )
    url = st.text_input("JSON URL", placeholder="https://.../full_report.json", key="json_url")

    st.divider()
    st.header("Audit Configuration")
    amount_tolerance = st.number_input("Amount tolerance (RM)", min_value=0.0, value=0.01, step=0.01)
    max_examples = st.number_input("Max examples per section", min_value=5, value=30, step=5)
    strict_schema = st.checkbox("Strict schema", value=False)
    compare_only_common_fields = st.checkbox(
        "Compare only common fields",
        value=True,
        help="Ignore missing monthly fields and compare only overlapping fields.",
    )

    cfg = AuditConfig(
        amount_tolerance=float(amount_tolerance),
        max_examples=int(max_examples),
        strict_schema=bool(strict_schema),
        compare_only_common_fields=bool(compare_only_common_fields),
    )


def _load_url(raw_url: str) -> Dict[str, Any]:
    response = requests.get(raw_url, timeout=30)
    response.raise_for_status()
    return response.json()


data: Optional[Dict[str, Any]] = None
source: Optional[str] = None

try:
    if st.session_state.input_mode.startswith("Fetch"):
        if not url.strip():
            st.info("Enter a JSON URL in the sidebar to begin.")
            st.stop()
        if st.button("Fetch JSON", type="primary"):
            data = _load_url(url.strip())
            source = f"URL: {url.strip()}"
        else:
            st.stop()
    else:
        pasted = st.text_area("Paste full JSON payload", height=260)
        if st.button("Load JSON", type="primary"):
            data = json.loads(pasted)
            source = "Pasted JSON"
        else:
            st.stop()
except Exception as err:
    st.error("Failed to load JSON input.")
    st.code(str(err))
    st.code(traceback.format_exc())
    st.stop()

st.success(f"Loaded source: {source}")

with st.spinner("Running statement audit..."):
    result = run_audit(data, cfg)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Overall Status", "PASS ✅" if result.overall_pass else "FAIL ❌")
k2.metric("Checks Run", result.checks_run)
k3.metric("Failed Checks", result.checks_failed)
k4.metric("Warnings", result.warnings)

if result.normalization_notes:
    st.info("Compatibility layer applied: " + " | ".join(sorted(set(result.normalization_notes))))

st.divider()
st.subheader("Monthly Summary Validation")
st.caption(
    "This section separates true numeric mismatches from schema gaps so your team can focus on accuracy issues first."
)

overview_df = pd.DataFrame(result.monthly_overview or [])
if overview_df.empty:
    st.warning("No monthly overview could be generated.")
else:
    st.dataframe(overview_df, use_container_width=True, height=280)

tab_mismatch, tab_missing = st.tabs(["Real mismatches", "Missing fields (schema gap)"])

with tab_mismatch:
    mismatch_df = pd.DataFrame(result.monthly_value_mismatches or [])
    if mismatch_df.empty:
        st.success("No real numeric mismatches detected.")
    else:
        st.error("These rows indicate real provided-vs-recomputed differences.")
        st.dataframe(mismatch_df, use_container_width=True, height=340)

with tab_missing:
    missing_df = pd.DataFrame(result.monthly_missing_fields or [])
    if missing_df.empty:
        st.success("No schema-gap items detected for current settings.")
    else:
        st.warning("Schema gap means a field is missing in provided monthly_summary, not necessarily wrong values.")
        st.dataframe(missing_df, use_container_width=True, height=340)

st.divider()
st.subheader("Provided vs Recomputed Monthly Summary")
left, right = st.columns(2)
with left:
    st.caption("Provided monthly_summary")
    st.dataframe(pd.DataFrame(data.get("monthly_summary", [])), use_container_width=True, height=250)
with right:
    st.caption("Recomputed from transactions")
    st.dataframe(pd.DataFrame(result.recomputed_monthly_summary or []), use_container_width=True, height=250)

st.divider()
st.subheader("Findings")
if result.failed_checks:
    for finding in result.failed_checks:
        st.error(f"{finding.name}: {finding.message}")
else:
    st.success("No failed checks.")

if result.warning_checks:
    for finding in result.warning_checks:
        st.warning(f"{finding.name}: {finding.message}")

st.divider()
st.subheader("Export")
report = audit_report_json(data, result, cfg)
st.download_button(
    "Download audit_report.json",
    data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name="audit_report.json",
    mime="application/json",
)
