# Ivey Synthetic Data App — Streamlit Cloud (Free Groq Chatbot)

This repo deploys a Streamlit app with:
- **Tab 1:** AI Education Assistant (uses **Groq’s free tier** first; falls back to local **Ollama** for dev)
- Data generator & Healthcare simulator tabs

## Repo layout
- `ivey_synthetic_data_app.py` – main app (Groq-first, Ollama fallback)
- `requirements.txt` – includes `langchain-groq` and `groq`
- `.streamlit/runtime.txt` – pins Python 3.11 on Streamlit Cloud
- `.gitignore`

---

## Quick start (Streamlit Community Cloud)

1. Push this folder to a **new GitHub repo**.
2. Go to **https://share.streamlit.io** → **New app**.
3. Select your repo, branch **`main`**.
4. **Main file path:** `ivey_synthetic_data_app.py`.
5. Click **Deploy**.
6. Add a Groq key in **Manage app → Settings → Secrets** (see below), then **Reboot**.

---

## Get a **FREE** API key (Groq)

1. Open **https://console.groq.com/** and sign in / create an account.  
2. In the left sidebar, go to **API Keys** → **Create API Key**.  
3. Copy the key that looks like `grq_...` (or `gsk_...`).

### Add the key to Streamlit (Secrets)

In your deployed app: **Manage app → Settings → Secrets** → paste TOML:

```toml
GROQ_API_KEY = "grq_XXXXXXXXXXXXXXXX"
```

Click **Save**, then **Reboot** the app.

> Tip: If the app still shows an “AI Assistant Setup” message, use the app menu (⋮) → **Clear cache** and refresh.

### (Optional) Use the key locally
Create `.streamlit/secrets.toml` in the project:

```toml
GROQ_API_KEY = "grq_XXXXXXXXXXXXXXXX"
```

Run locally:
```bash
python -m venv .venv
# Windows:   .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run ivey_synthetic_data_app.py
```

---

## Requirements

`requirements.txt` must include these (plus your other libs):

```
streamlit>=1.36
pandas
numpy
plotly>=5.20
scipy
requests
langchain-core
langchain-community
langchain-groq
groq
```

The chatbot uses Groq’s **`llama-3.1-8b-instant`** with `temperature=0.2`, `max_tokens=512` by default.

---

## Troubleshooting

- **“AI Assistant Setup” card won’t go away**  
  Add `GROQ_API_KEY` in **Secrets**, then **Reboot** (or **Clear cache**).  
- **`ModuleNotFoundError: langchain_groq / groq`**  
  Add both packages to `requirements.txt`, push, and redeploy.
- **401/403**  
  Key missing/invalid → paste the correct key in **Secrets**.
- **429 (rate limit)**  
  Keep questions short; try again later.
- **App tries `localhost:11434`**  
  That’s the Ollama fallback. Ensure `GROQ_API_KEY` is set so the app uses Groq on the cloud.

---

## (Optional) Local Ollama for development

If you run Ollama on your laptop, the app will automatically use it **only when no Groq key is set** and the server is reachable:

- Default host: `http://localhost:11434`  
- To point elsewhere, set `OLLAMA_HOST` in your environment or in Streamlit **Secrets**:

```toml
OLLAMA_HOST = "http://127.0.0.1:11434"
```

---

## Notes

- **Never commit API keys** to Git—keep them in **Secrets**.
- Your original behavior is preserved: Groq in the cloud, Ollama as a local fallback.
- Python version is pinned via `.streamlit/runtime.txt` for reproducible builds.
