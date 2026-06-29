# ClaimLens Streamlit Deployment

Target URL: https://munish.streamlit.app/

To run this app on Streamlit Cloud, the connected repository should use:

- Main file path: `streamlit_app.py`
- Python dependencies: `requirements.txt`
- Secrets:
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_API_BASE_URL`
  - `OPENROUTER_MODEL`

The app uses OpenRouter to call a Llama model for claim extraction and evidence
judging when `OPENROUTER_API_KEY` is configured. If the key is missing or an API
call fails, the app keeps working with deterministic extraction and verification
fallbacks.
