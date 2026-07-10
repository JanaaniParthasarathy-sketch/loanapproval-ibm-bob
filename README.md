# 🏦 LoanAdvisor AI — IBM watsonx.ai + Streamlit

A loan approval prediction app powered by a Gradient Boosting ML model and
IBM Granite LLM (via watsonx.ai) for intelligent chat-based financial guidance.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loanapproval-ibm-bob.streamlit.app)

---

## Features
- **ML Prediction** — GradientBoostingClassifier trained on 614 synthetic loan records
- **Feature Impact** — visualises which factors drove the decision
- **AI Chat** — IBM Granite-13b-chat-v2 via watsonx.ai (smart fallback if not configured)
- **Prediction History** — tracks last 10 predictions in session
- **Dataset Stats** — live charts in sidebar

## Local Setup

```bash
git clone https://github.com/JanaaniParthasarathy-sketch/loanapproval-ibm-bob.git
cd loanapproval-ibm-bob
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud Deployment

1. Push this repo to GitHub (all files at root level)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** = `streamlit_app.py`
4. Under **Advanced settings → Secrets**, add:

```toml
IBM_API_KEY = "your_ibm_cloud_api_key"
WATSONX_URL = "https://au-syd.ml.cloud.ibm.com"
WATSONX_PROJECT_ID = "your_project_id"
GRANITE_MODEL_ID = "ibm/granite-13b-chat-v2"
```

5. Click **Deploy** — live in ~2 minutes!

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `IBM_API_KEY` | IBM Cloud API key | Optional (fallback mode if absent) |
| `WATSONX_URL` | watsonx.ai region URL | Optional |
| `WATSONX_PROJECT_ID` | watsonx.ai project ID | Optional |
| `GRANITE_MODEL_ID` | Granite model ID | Optional |

## Project Structure

```
├── streamlit_app.py      ← Main Streamlit app (entry point)
├── loan_model.py         ← ML model training & prediction
├── agent_config.py       ← IBM Granite agent instructions
├── generate_dataset.py   ← Synthetic dataset generator
├── requirements.txt      ← Python dependencies
├── .streamlit/
│   └── config.toml       ← Streamlit theme & server config
└── README.md
```
