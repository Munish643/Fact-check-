from __future__ import annotations

import os
from datetime import datetime, timezone

import streamlit as st

from claimlens_core import extract_claims, extract_pdf_text, llama_config_from_env, verify_claim


APP_URL = "https://munish.streamlit.app/"


def secret_value(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, os.getenv(name, default))
    except Exception:
        value = os.getenv(name, default)
    return "" if value is None else str(value)


def get_llama_config() -> dict[str, str]:
    env_config = llama_config_from_env()
    return {
        "api_key": secret_value("OPENROUTER_API_KEY", secret_value("LLAMA_API_KEY", env_config["api_key"])),
        "base_url": secret_value("OPENROUTER_API_BASE_URL", secret_value("LLAMA_API_BASE_URL", env_config["base_url"])),
        "model": secret_value("OPENROUTER_MODEL", secret_value("LLAMA_MODEL", env_config["model"])),
    }


st.set_page_config(page_title="ClaimLens", page_icon=":mag:", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; max-width: 1220px; }
      .cl-header { border: 1px solid #d8e1dc; border-radius: 10px; padding: 1.2rem 1.35rem; background: #ffffff; }
      .cl-logo { display:inline-flex; align-items:center; justify-content:center; width:46px; height:46px; border-radius:10px; background:#10201d; color:#c7f9f1; font-weight:800; margin-right:14px; }
      .cl-kicker { color:#0f766e; font-size:.78rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; }
      .cl-title { font-size:2.4rem; font-weight:850; color:#0f172a; margin:0; }
      .cl-card { border: 1px solid #d8e1dc; border-radius: 10px; padding: 1rem; background: #ffffff; }
      .cl-muted { color:#475569; }
      .status-verified { color:#047857; font-weight:800; }
      .status-inaccurate { color:#b45309; font-weight:800; }
      .status-false { color:#be123c; font-weight:800; }
    </style>
    """,
    unsafe_allow_html=True,
)

llama_config = get_llama_config()

with st.sidebar:
    st.subheader("ClaimLens settings")
    st.write("Deployment target")
    st.link_button("Open Streamlit app", APP_URL)
    st.divider()
    st.write("OpenRouter Llama API")
    if llama_config["api_key"]:
        st.success("Configured")
    else:
        st.warning("Missing OPENROUTER_API_KEY")
    st.caption(f"Model: `{llama_config['model']}`")
    st.caption("Set secrets in Streamlit Cloud for production.")

st.markdown(
    f"""
    <div class="cl-header">
      <div style="display:flex; align-items:center;">
        <div class="cl-logo">CL</div>
        <div>
          <div class="cl-kicker">Llama-powered evidence scanner</div>
          <h1 class="cl-title">ClaimLens</h1>
        </div>
      </div>
      <p class="cl-muted" style="margin:.8rem 0 0 0;">
        Upload a PDF, extract measurable claims, verify them against live sources, and classify each claim as
        Verified, Inaccurate, or False.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.38, 0.62], gap="large")

with left:
    st.markdown("### Document workspace")
    uploaded = st.file_uploader("Upload marketing PDF", type=["pdf"])
    pasted_text = st.text_area("Or paste document text", height=180, placeholder="Paste PDF text for a quick trap test.")
    run = st.button("Run verification", type="primary", use_container_width=True)

    st.markdown("### Verification path")
    st.markdown(
        """
        1. Extract PDF text
        2. Ask Llama to identify measurable claims
        3. Search live web and public data sources
        4. Ask Llama to judge evidence and explain the verdict
        """
    )

with right:
    st.markdown("### Claim report")
    if not run:
        st.info("Upload a PDF or paste text, then run verification.")
    else:
        try:
            if pasted_text.strip():
                text = pasted_text.strip()
                file_name = "Pasted text"
            elif uploaded:
                text = extract_pdf_text(uploaded.getvalue())
                file_name = uploaded.name
            else:
                st.error("Please upload a PDF or paste text first.")
                st.stop()

            if len(text) < 80:
                st.error("The extracted text is too short to analyze.")
                st.stop()

            started = datetime.now(timezone.utc).isoformat()
            with st.spinner("Extracting measurable claims..."):
                claims, extractor = extract_claims(text, llama_config)

            if not claims:
                st.warning("No measurable claims were detected.")
                st.stop()

            results = []
            progress = st.progress(0, text="Checking live evidence")
            for index, claim in enumerate(claims):
                with st.spinner(f"Verifying claim {index + 1} of {len(claims)}"):
                    results.append(verify_claim(claim, llama_config))
                progress.progress((index + 1) / len(claims), text="Checking live evidence")
            progress.empty()

            totals = {"Verified": 0, "Inaccurate": 0, "False": 0}
            for item in results:
                totals[item["status"]] += 1

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Claims", len(results))
            c2.metric("Verified", totals["Verified"])
            c3.metric("Inaccurate", totals["Inaccurate"])
            c4.metric("False", totals["False"])

            st.caption(f"File: {file_name} | Checked: {started} | Claim extractor: {extractor}")

            for item in results:
                status_class = f"status-{item['status'].lower()}"
                with st.container(border=True):
                    st.markdown(
                        f"<span class='{status_class}'>{item['status']}</span> "
                        f"<span class='cl-muted'>| {item['category']} | {round(item['confidence'] * 100)}% confidence | {item['verifier']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{item['claim']}**")
                    st.write(item["correction"])
                    st.caption(item["rationale"])

                    if item["sources"]:
                        with st.expander("Sources", expanded=False):
                            for source in item["sources"]:
                                st.markdown(f"**[{source['title']}]({source['url']})** - {source['provider']}")
                                st.caption(source["snippet"])
                    else:
                        st.caption("No source links returned.")

        except Exception as exc:
            st.error(f"Verification failed: {exc}")
