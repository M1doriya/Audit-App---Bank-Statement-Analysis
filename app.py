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
    "If upload triggers 'Reconnecting…', it usually means the server crashed/restarted. "
    "This build is hardened to prevent upload crashes and to surface errors."
)

# -----------------------------
# Sidebar (uploader ALWAYS rendered)
# -----------------------------
with st.sidebar:
    st.header("Input")

    input_mode = st.radio(
        "Choose input",
        ["Upload JSON", "Fetch by URL"],
        index=0,
        key="input_mode",
    )

    uploaded = st.file_uploader(
        "Upload full_report.json",
        type=["json"],
        key="json_uploader",
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

# -----------------------------
# Debug panel
# -----------------------------
with st.expander("Debug / Diagnostics", expanded=True):
    st.write("If upload causes reconnect, check Railway **Deploy Logs** for a traceback.")
    st.write("This panel shows what the server actually received (if it received anything).")


def load_json_from_upload(up) -> Dict[str, Any]:
    """
    Robust loader:
    - Reads bytes once
    - Tries UTF-8
    - If that fails, tries JSON from bytes via latin-1
    - If still fails, raises with clear context
    """
    raw = up.getvalue()  # safe: does not consume stream multiple times
    if raw is None:
        raise ValueError("No bytes received from uploader.")

    # size sanity (helps confirm upload actually arrived)
    st.write(f"✅ Upload bytes received: {len(raw)} bytes")

    # Try parsing directly from bytes decoding
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("latin-1"))
    except json.JSONDecodeError as e:
        # show first 200 chars for debugging (safe)
        preview = raw[:200]
        try:
            preview_text = preview.decode("utf-8", errors="replace")
        except Exception:
            preview_text = str(preview)
        raise ValueError(f"JSON decode error: {e}. First bytes preview: {preview_text!r}")


def load_json_from_url(u: str) -> Dict[str, Any]:
    r = requests.get(u, timeout=30)
    r.raise_for_status()
    return r.json()


data: Optional[Dict[str, Any]] = None
source: Optional[str] = None

try:
    if st.session_state.input_mode == "Upload JSON":
        if uploaded is not None:
            # confirm immediately
            st.sidebar.success(f"Selected: {uploaded.name} ({uploaded.size} bytes)")
            data = load_json_from_upload(uploaded)
            source = f"upload:{uploaded.name}"
    else:
        u = (st.session_state.json_url or "").strip()
        if u:
            data = load_json_from_url(u)
            source = f"url:{u}"
            st.sidebar.success("Fetched JSON from URL.")
except Exception as e:
    st.error("❌ Failed to load JSON. Details below.")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

if data is None:
    st.info("Use the sidebar to upload a JSON or fetch by URL.")
    st.stop()

st.success(f"Loaded report from: {source}")

# -----------------------------
# Run audit (guarded)
# -----------------------------
try:
    with st.spinner("Running audit checks..."):
        result = run_audit(data, cfg)
except Exception as e:
    st.error("❌ Audit crashed (this should not happen).")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

# -----------------------------
# UI output
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
    st.dataframe(pd.DataFrame(provided_monthly) if isinstance(provided_monthly, list) else pd.DataFrame())

with right:
    st.caption("Recomputed monthly summary (from transactions)")
    st.dataframe(pd.DataFrame(recomputed_monthly) if recomputed_monthly else pd.DataFrame())

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
