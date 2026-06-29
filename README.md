# ClaimLens

ClaimLens is a Streamlit fact-checking web app for uploaded PDFs. It extracts measurable claims, gathers live evidence, and classifies each claim as `Verified`, `Inaccurate`, or `False`.

Target deployment URL:

https://munish.streamlit.app/

## Streamlit Cloud Setup

Use this repository with:

- Main file: `streamlit_app.py`
- Dependencies: `requirements.txt`
- Python secrets:
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_API_BASE_URL`
  - `OPENROUTER_MODEL`

Example secrets are shown in `.streamlit/secrets.toml.example`.

## What It Does

- Reads uploaded PDFs with `pypdf`.
- Uses OpenRouter Llama API for claim extraction and evidence-grounded verdict reasoning when configured.
- Falls back to deterministic extraction and verification when the API key is missing or unavailable.
- Searches live evidence sources, including Wikipedia, DuckDuckGo Lite, and World Bank population data.
- Includes a professional project deck at `outputs/claimlens-project.pptx`.
