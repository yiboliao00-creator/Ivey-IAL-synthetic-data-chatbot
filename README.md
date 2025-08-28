# Ivey Synthetic Data App — Streamlit Cloud (Groq-ready)

This repo is ready to deploy on **Streamlit Community Cloud** with a **free Groq LLM** for the chatbot.
It uses Groq in the cloud (if a key is provided) and falls back to local **Ollama** only for local development.

## Files
- `ivey_synthetic_data_app.py` — main app (patched for Groq; local fallback to Ollama)
- `ivey_synthetic_data_app_original.py` — original code (unchanged)
- `requirements.txt` — Python dependencies (includes `langchain-groq` + `groq`)
- `.streamlit/runtime.txt` — Python 3.11 pin for Streamlit Cloud
- `.gitignore`

---

## Get a FREE API Key (Groq)

1. Go to **https://console.groq.com/** and create/sign in to your account.
2. In the left sidebar, click **API Keys** → **Create API Key**.
3. Copy the key that looks like **`grq_XXXXXXXXXXXXXXXX`** and keep it safe.

> Groq offers a *free tier* suitable for coursework. Quotas and availability can change—check their console/pricing for current limits.

### Add the key to Streamlit Cloud (Secrets)
1. Open your deployed app → click **Manage app** (bottom-left).
2. Go to **Settings → Secrets**.
3. Paste the following **TOML** and click **Save**:
   ```toml
   GROQ_API_KEY = "grq_XXXXXXXXXXXXXXXX"
   ```
4. Click **Reboot** to apply the secret.

### (Optional) Use the key locally
Create a file `.streamlit/secrets.toml` in your project with:
```toml
GROQ_API_KEY = "grq_XXXXXXXXXXXXXXXX"
```
Then run:
```bash
streamlit run ivey_synthetic_data_app.py
```

---

## Deploy on Streamlit Cloud
1. Push this folder to a **new GitHub repo**.
2. Visit **https://share.streamlit.io** → **New app**.
3. Select your repo, branch **`main`**.
4. **Main file path**: `ivey_synthetic_data_app.py`
5. Click **Deploy**.
6. Add the `GROQ_API_KEY` in Secrets (steps above) to enable the chatbot.

## Local run (optional)
```bash
python -m venv .venv && . .venv/Scripts/activate  # Windows
# or: python3 -m venv .venv && source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run ivey_synthetic_data_app.py
```

## Troubleshooting
- **ModuleNotFoundError (langchain_groq / groq):** ensure `requirements.txt` includes both packages and redeploy.
- **No Groq key found:** add `GROQ_API_KEY` in **Settings → Secrets** (TOML format).
- **Rate limits / 429:** keep prompts concise; try again later.
- **Prefer local Ollama:** on your laptop, run an Ollama server; the app will automatically use it if no Groq key is set.

---

### Notes
- Keep API keys private—never commit them to GitHub.
- The chatbot uses Groq's `llama-3.1-8b-instant` model (`temperature=0.2`, `max_tokens=512`) by default.
- Your original file is preserved as `ivey_synthetic_data_app_original.py` for reference.
