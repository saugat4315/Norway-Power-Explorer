# ⚡ Norway Power Explorer

Streamlit app: day-ahead power price (bold line) compared with weather, reservoirs, rivers,
consumption/production and cross-border flows. All settings are in the left sidebar.

## Put it online
1. github.com → **New repository** (e.g. `norway-power-explorer`).
2. **Add file → Upload files**: `app.py`, `data_sources.py`, `requirements.txt`, `README.md`, `.gitignore`.
   Then **Add file → Create new file**, name it `.streamlit/config.toml` and paste its contents.
   Never upload keys.
3. share.streamlit.io → **Create app** → choose the repository, branch `main`, file `app.py`.
4. **Advanced settings → Secrets**: paste ENTSOE_TOKEN, FROST_ID, HYDAPI_KEY lines. **Deploy**.

Change the app later by editing or re-uploading files on GitHub; Streamlit updates within a minute.
