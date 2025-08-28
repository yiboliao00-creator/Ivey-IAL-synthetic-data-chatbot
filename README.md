# Ivey Synthetic Data App — Streamlit Cloud (Groq-ready)

This repo is ready to push to GitHub and deploy on **Streamlit Community Cloud** with a **free Groq LLM** for the chatbot.

## Files
- `ivey_synthetic_data_app.py` — main app (patched to use Groq in the cloud, falls back to local Ollama)
- `ivey_synthetic_data_app_original.py` — your original code (kept unchanged for reference)
- `requirements.txt` — Python dependencies (adds `langchain-groq` + `groq`)
- `.streamlit/runtime.txt` — Python 3.11 pin for Streamlit Cloud
- `.gitignore` — avoids committing venv & secrets

## Deploy on Streamlit Cloud
1. Create a **new GitHub repo** and push this folder.
2. Visit https://share.streamlit.io → **New app**
3. Repo: select your repo, Branch: `main`
4. **Main file path**: `ivey_synthetic_data_app.py`
5. Deploy.

## Add FREE Groq API key (to enable the chatbot)
Open the app → **Manage app → Settings → Secrets** and paste:
```toml
GROQ_API_KEY = "grq_XXXXXXXXXXXXXXXX"
```
Then **Save** and **Reboot**.

The app will now use Groq's `llama-3.1-8b-instant` model. If no key is set, the chatbot shows a gentle notice instead of failing.

---

### Local run (optional)
```bash
python -m venv .venv && . .venv/Scripts/activate  # on Windows
# or: source .venv/bin/activate                     # on macOS/Linux
pip install -r requirements.txt
streamlit run ivey_synthetic_data_app.py
```

