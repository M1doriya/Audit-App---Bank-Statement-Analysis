from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

from audit_checks import AuditConfig, audit_report_json, run_audit

st.set_page_config(page_title="Bank Report Audit", layout="wide")
st.title("Bank Statement Output Audit")
st.caption(
    "If file upload causes 'Reconnecting…' on Railway, use Fetch-by-URL or Paste JSON instead. "
    "Those do not rely on Streamlit's upload transport."
)

# -----------------------------
# Sidebar: input + settings
# -----------------------------
with st.sidebar:
    st.header("Input")

    input_mode = st.radio(
        "Choose input mode",
        ["Fetch by URL (recommended)", "Paste JSON (recommended)", "Upload JSON (may fail on Railway)"],
        index=0,
        key="input_mode",
    )

    url = st.text_input(
        "Report JSON URL",
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

# -----------------------------
# Main input UI
# -----------------------------
data: Optional[Dict[str, Any]] = None
source: Optional[str] = None

def _load_from_url(u: str) -> Dict[str, Any]:
    r = requests.get(u, timeout=30)
    r.raise_for_status()
    return r.json()

def _load_from_text(txt: str) -> Dict[str, Any]:
    return json.loads(txt)

def _load_from_upload(up) -> Dict[str, Any]:
    raw = up.getvalue()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("latin-1"))

try:
    if st.session_state.input_mode.startswith("Fetch"):
        st.subheader("Fetch by URL")
        st.write("This is the most stable method on Railway.")
        if url.strip():
            if st.button("Fetch JSON", type="primary"):
                data = _load_from_url(url.strip())
                source = f"url:{url.strip()}"
        else:
            st.info("Paste a JSON URL in the sidebar, then click **Fetch JSON**.")

    elif st.session_state.input_mode.startswith("Paste"):
        st.subheader("Paste JSON")
        st.write("Paste the full JSON text here (works even when upload reconnects).")
        pasted = st.text_area("full_report.json contents", height=260, placeholder='{"summary": ... }')
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Load pasted JSON", type="primary"):
                if not pasted.strip():
                    st.warning("Paste JSON first.")
                else:
                    data = _load_from_text(pasted.strip())
                    source = "pasted-json"
        with c2:
            st.caption("Tip: You can copy JSON from your bank app download output and paste here.")

    else:
        st.subheader("Upload JSON (may fail on Railway)")
        st.write(
            "If selecting a file triggers **Reconnecting…**, it's a Railway/Streamlit transport issue. "
            "Use Fetch-by-URL or Paste JSON instead."
        )
        uploaded = st.file_uploader("Upload full_report.json", type=["json"], key="json_uploader")
        if uploaded is not None:
            st.success(f"Selected: {uploaded.name} ({uploaded.size} bytes)")
            if st.button("Load uploaded JSON", type="primary"):
                data = _load_from_upload(uploaded)
                source = f"upload:{uploaded.name}"

except Exception as e:
    st.error("Failed to load input.")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

if data is None:
    st.stop()

st.success(f"Loaded report from: {source}")

# -----------------------------
# Run audit
# -----------------------------
try:
    with st.spinner("Running audit checks..."):
        result = run_audit(data, cfg)
except Exception as e:
    st.error("Audit crashed.")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

# -----------------------------
# Output UI
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall", "PASS ✅" if result.overall_pass else "FAIL ❌")
c2.metric("Checks run", str(result.checks_run))
c3.metric("Checks failed", str(result.checks_failed))
c4.metric("Warnings", str(result.warnings))

st.divider()

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

st.subheader("Transactions preview")
tx = data.get("transactions", [])
if isinstance(tx, list) and tx:
    st.caption(f"{len(tx)} transactions (showing first 200)")
    st.dataframe(pd.DataFrame(tx).head(200))
else:
    st.warning("No transactions found in the report JSON.")

st.divider()

out = audit_report_json(data, result, cfg)
st.download_button(
    "Download audit_report.json",
    data=json.dumps(out, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name="audit_report.json",
    mime="application/json",
)
