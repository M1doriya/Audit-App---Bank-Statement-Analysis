from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

from audit_checks import AuditConfig, audit_report_json, run_audit

# -----------------------------
# Page + styling
# -----------------------------
st.set_page_config(page_title="Statement Audit", layout="wide")

st.markdown(
    """
<style>
/* Subtle professional look */
.block-container { padding-top: 1.6rem; }
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
.badge {
  display:inline-block; padding: 0.20rem 0.55rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.2px;
  border: 1px solid rgba(255,255,255,0.12);
}
.badge-fail { background: rgba(255, 82, 82, 0.18); }
.badge-warn { background: rgba(255, 193, 7, 0.16); }
.badge-pass { background: rgba(76, 175, 80, 0.16); }
.panel {
  padding: 1rem; border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}
hr { border-color: rgba(255,255,255,0.10); }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("## Statement Audit Console")
st.markdown('<div class="small-muted">Audit your generated JSON by recomputing totals and pinpointing monthly mismatches clearly.</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar Input (stable modes)
# -----------------------------
with st.sidebar:
    st.header("Input")

    input_mode = st.radio(
        "Choose input mode",
        ["Fetch by URL (recommended)", "Paste JSON (recommended)", "Upload JSON (may reconnect)"],
        index=0,
        key="input_mode",
    )

    url = st.text_input("JSON URL", placeholder="https://.../full_report.json", key="json_url")

    st.divider()
    st.header("Audit Settings")
    amount_tolerance = st.number_input("Amount tolerance (RM)", min_value=0.0, value=0.01, step=0.01)
    max_examples = st.number_input("Max examples per check", min_value=5, value=30, step=5)
    strict_schema = st.checkbox("Strict schema", value=False)

    cfg = AuditConfig(
        amount_tolerance=float(amount_tolerance),
        max_examples=int(max_examples),
        strict_schema=bool(strict_schema),
    )

    st.divider()
    st.caption("Tip: If upload reconnects on Railway, use Fetch-by-URL or Paste JSON.")

# -----------------------------
# Load input
# -----------------------------
data: Optional[Dict[str, Any]] = None
source: Optional[str] = None

def _load_url(u: str) -> Dict[str, Any]:
    r = requests.get(u, timeout=30)
    r.raise_for_status()
    return r.json()

def _load_text(txt: str) -> Dict[str, Any]:
    return json.loads(txt)

def _load_upload(up) -> Dict[str, Any]:
    raw = up.getvalue()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("latin-1"))

try:
    if st.session_state.input_mode.startswith("Fetch"):
        st.markdown("### Input: Fetch by URL")
        if url.strip():
            colA, colB = st.columns([1, 4])
            with colA:
                go = st.button("Fetch JSON", type="primary")
            with colB:
                st.markdown(f'<div class="small-muted">{url.strip()}</div>', unsafe_allow_html=True)
            if go:
                data = _load_url(url.strip())
                source = f"url:{url.strip()}"
        else:
            st.info("Paste a JSON URL in the sidebar and click **Fetch JSON**.")
            st.stop()

    elif st.session_state.input_mode.startswith("Paste"):
        st.markdown("### Input: Paste JSON")
        pasted = st.text_area("Paste full JSON here", height=260, placeholder='{"summary": ... }')
        if st.button("Load JSON", type="primary"):
            if not pasted.strip():
                st.warning("Paste JSON content first.")
                st.stop()
            data = _load_text(pasted.strip())
            source = "pasted-json"

    else:
        st.markdown("### Input: Upload JSON")
        uploaded = st.file_uploader("Upload full_report.json", type=["json"], key="uploader")
        if uploaded is not None:
            st.success(f"Selected: {uploaded.name} ({uploaded.size} bytes)")
            if st.button("Load uploaded JSON", type="primary"):
                data = _load_upload(uploaded)
                source = f"upload:{uploaded.name}"

except Exception as e:
    st.error("Failed to load input.")
    st.code(str(e))
    st.code(traceback.format_exc())
    st.stop()

if data is None:
    st.stop()

st.markdown(f'<div class="panel"><b>Loaded:</b> {source}</div>', unsafe_allow_html=True)

# -----------------------------
# Run audit
# -----------------------------
with st.spinner("Running checks..."):
    result = run_audit(data, cfg)

# -----------------------------
# KPI Header
# -----------------------------
def _badge(text: str, kind: str) -> str:
    cls = {"pass": "badge-pass", "warn": "badge-warn", "fail": "badge-fail"}.get(kind, "badge-warn")
    return f'<span class="badge {cls}">{text}</span>'

overall = "PASS" if result.overall_pass else "FAIL"
overall_kind = "pass" if result.overall_pass else "fail"

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"### Overall {_badge(overall, overall_kind)}", unsafe_allow_html=True)
c2.metric("Checks run", result.checks_run)
c3.metric("Failed checks", result.checks_failed)
c4.metric("Warnings", result.warnings)

st.markdown("---")

# -----------------------------
# Monthly Comparison (Professional)
# -----------------------------
st.markdown("## Monthly Summary Comparison")

provided_monthly = data.get("monthly_summary", [])
recomputed_monthly = result.recomputed_monthly_summary or []
diffs = result.monthly_diffs or []

left, right = st.columns(2)

with left:
    st.markdown("### Provided (from JSON)")
    if isinstance(provided_monthly, list) and provided_monthly:
        st.dataframe(pd.DataFrame(provided_monthly), use_container_width=True, height=320)
    else:
        st.info("No `monthly_summary` provided.")

with right:
    st.markdown("### Recomputed (from transactions)")
    if recomputed_monthly:
        st.dataframe(pd.DataFrame(recomputed_monthly), use_container_width=True, height=320)
    else:
        st.warning("Unable to recompute monthly summary (missing/invalid transactions).")

st.markdown("---")

# -----------------------------
# Mismatch Dashboard
# -----------------------------
st.markdown("## Mismatch Dashboard")

if not diffs:
    st.success("No monthly diffs detected.")
else:
    df = pd.DataFrame(diffs)

    # Normalize columns for display
    for col in ["severity", "delta", "abs_delta"]:
        if col not in df.columns:
            df[col] = None

    # KPIs
    months_affected = df["month"].nunique() if "month" in df.columns else 0
    hard_mismatches = int((df["status"] == "MISMATCH").sum()) if "status" in df.columns else 0
    missing_fields = int((df["status"].str.contains("MISSING", na=False)).sum()) if "status" in df.columns else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Months affected", months_affected)
    k2.metric("True mismatches", hard_mismatches)
    k3.metric("Missing fields / months", missing_fields)

    st.markdown("### Where exactly is the mismatch?")

    # Styled table
    show_cols = ["month", "field", "status", "severity", "provided", "recomputed", "delta", "abs_delta"]
    show_cols = [c for c in show_cols if c in df.columns]

    def style_row(row):
        status = str(row.get("status", ""))
        sev = str(row.get("severity", ""))
        if status.startswith("MISSING_IN_"):
            return ["background-color: rgba(255,193,7,0.12)"] * len(row)
        if status.startswith("MISSING_FIELD"):
            return ["background-color: rgba(255,193,7,0.08)"] * len(row)
        if status == "MISMATCH":
            if sev == "HIGH":
                return ["background-color: rgba(255,82,82,0.14)"] * len(row)
            return ["background-color: rgba(255,82,82,0.08)"] * len(row)
        return [""] * len(row)

    styled = (
        df[show_cols]
        .copy()
        .style
        .apply(style_row, axis=1)
        .format({"delta": "{:,.2f}", "abs_delta": "{:,.2f}"}, na_rep="")
    )

    st.dataframe(styled, use_container_width=True, height=360)

    st.markdown("---")

    # Drill-down
    st.markdown("## Month Drill-Down")
    months = sorted(df["month"].dropna().unique().tolist())
    chosen = st.selectbox("Select a month to inspect", months)

    prov_map = {r.get("month"): r for r in (provided_monthly if isinstance(provided_monthly, list) else []) if isinstance(r, dict)}
    reco_map = {r.get("month"): r for r in (recomputed_monthly or []) if isinstance(r, dict)}

    pr = prov_map.get(chosen, {})
    rr = reco_map.get(chosen, {})

    colL, colR, colD = st.columns([1, 1, 1])

    with colL:
        st.markdown("### Provided")
        st.json(pr if pr else {"note": "No provided row for this month"})

    with colR:
        st.markdown("### Recomputed")
        st.json(rr if rr else {"note": "No recomputed row for this month"})

    with colD:
        st.markdown("### Delta (Recomputed − Provided)")
        fields = ["transaction_count", "total_debit", "total_credit", "net_change"]
        delta_rows = []
        for f in fields:
            pv = pr.get(f)
            rv = rr.get(f)
            if pv is None and rv is None:
                continue
            if pv is None:
                delta_rows.append({"field": f, "provided": None, "recomputed": rv, "delta": None, "note": "missing in provided"})
                continue
            if rv is None:
                delta_rows.append({"field": f, "provided": pv, "recomputed": None, "delta": None, "note": "missing in recomputed"})
                continue
            try:
                if f == "transaction_count":
                    delta_rows.append({"field": f, "provided": int(pv), "recomputed": int(rv), "delta": int(rv) - int(pv), "note": ""})
                else:
                    delta_rows.append({"field": f, "provided": float(pv), "recomputed": float(rv), "delta": round(float(rv) - float(pv), 2), "note": ""})
            except Exception:
                delta_rows.append({"field": f, "provided": pv, "recomputed": rv, "delta": None, "note": "non-numeric"})

        st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, height=260)

# -----------------------------
# Checks list (professional)
# -----------------------------
st.markdown("---")
st.markdown("## Findings")

if result.failed_checks:
    st.markdown("### ❌ Failed checks")
    for fc in result.failed_checks:
        st.markdown(f'<div class="panel"><b>{fc.name}</b> {_badge("FAIL", "fail")}<br/>{fc.message}</div>', unsafe_allow_html=True)
        if fc.examples:
            st.dataframe(pd.DataFrame(fc.examples), use_container_width=True)
else:
    st.markdown(f'<div class="panel">{_badge("PASS", "pass")} No failed checks.</div>', unsafe_allow_html=True)

if result.warning_checks:
    st.markdown("### ⚠️ Warnings")
    for wc in result.warning_checks:
        st.markdown(f'<div class="panel"><b>{wc.name}</b> {_badge("WARN", "warn")}<br/>{wc.message}</div>', unsafe_allow_html=True)
        if wc.examples:
            st.dataframe(pd.DataFrame(wc.examples), use_container_width=True)

# -----------------------------
# Export
# -----------------------------
st.markdown("---")
st.markdown("## Export")
out = audit_report_json(data, result, cfg)
st.download_button(
    "Download audit_report.json",
    data=json.dumps(out, indent=2, ensure_ascii=False).encode("utf-8"),
    file_name="audit_report.json",
    mime="application/json",
)
