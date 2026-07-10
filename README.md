# 🏦 LoanAdvisor AI — Loan Approval Prediction
### Powered by IBM watsonx.ai + Granite Models | Built with IBM BOB

A production-ready web application that uses Machine Learning and IBM Granite AI to predict
loan approval probability, explain key decision factors, and provide personalized financial guidance.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **IBM Granite AI** | IBM watsonx.ai Granite-13B chat model for intelligent loan explanations |
| 📊 **ML Prediction** | Gradient Boosting model trained on loan approval dataset (614+ records) |
| 💬 **AI Chat Assistant** | Ask questions about loan eligibility, financial planning, credit scores |
| 📈 **Analytics Dashboard** | Interactive charts showing dataset distributions and approval rates |
| 🕐 **Prediction History** | Session-based history of all predictions made |
| 🌙 **Dark Mode** | Toggle between light and dark themes |
| 📱 **Mobile Responsive** | Full Bootstrap 5 responsive design |
| 🔒 **Secure Config** | IBM API keys stored in `.env` (never in code) |
| 🛠️ **AGENT_INSTRUCTIONS** | Easily customize AI behavior, tone, and banking policies |

---

## 📁 Project Structure

```
loan_approval_app/
├── app.py                  # Flask backend + IBM watsonx.ai routes
├── loan_model.py           # ML model training + prediction logic
├── agent_config.py         # ⭐ AGENT_INSTRUCTIONS — customize AI behavior here
├── generate_dataset.py     # Generates synthetic loan approval dataset
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .env                    # Your actual credentials (DO NOT commit)
├── templates/
│   └── index.html          # Full frontend (Bootstrap 5 + Charts.js)
├── models/                 # Auto-created when model is trained
│   ├── loan_model.joblib
│   ├── label_encoders.joblib
│   └── scaler.joblib
└── data/                   # Auto-created dataset
    └── loan_approval.csv
```

---

## 🚀 Quick Start

### Step 1 — Clone & Install

```bash
# Navigate to the app folder
cd loan_approval_app

# Create a virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Configure IBM watsonx.ai Credentials

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` with your credentials:

```env
IBM_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-13b-chat-v2
FLASK_SECRET_KEY=change_this_to_a_random_string
FLASK_DEBUG=False
```

> **Get IBM credentials:**
> 1. Sign up at [IBM Cloud](https://cloud.ibm.com)
> 2. Create a **watsonx.ai** service instance
> 3. Go to **Manage → Access (IAM)** → Create API Key
> 4. Copy your **Project ID** from watsonx.ai Studio

### Step 3 — Upload Your CSV Dataset (Optional)

If you have the Loan Approval Prediction CSV dataset, copy it:

```bash
mkdir data
cp /path/to/your/loan_approval.csv data/loan_approval.csv
```

Expected columns:
```
Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status
```

If no dataset is provided, the app **auto-generates** a realistic 614-record dataset on first run.

### Step 4 — Train the Model & Run

```bash
# Train model on the dataset
python loan_model.py

# Start the Flask server
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🎛️ Customizing the AI Agent

Edit [`agent_config.py`](agent_config.py) to customize:

```python
AGENT_INSTRUCTIONS = """
## RESPONSE TONE
- Change how formal or casual the AI sounds

## LOAN EVALUATION CRITERIA
- Adjust credit score thresholds
- Change income requirements
- Modify DTI ratio limits

## SAFETY RULES
- Add compliance rules (GDPR, RBI, etc.)
- Block specific query types

## BANKING POLICIES
- Set maximum loan amounts
- Change tenure limits
- Define insurance requirements
"""
```

No restart needed — changes take effect on the next API call.

---

## 📊 Machine Learning Model Details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Training Split | 80% train / 20% test |
| Features | 11 (income, loan amount, credit history, education, etc.) |
| Target | Loan_Status (Y/N → Approved/Rejected) |
| Typical Accuracy | 80–85% |
| Missing Value Strategy | Mode (categorical) / Median (numerical) |
| Scaling | StandardScaler |

**Feature Importance Order:**
1. Credit History (most impactful)
2. Applicant Income
3. Loan Amount
4. Debt-to-Income Ratio
5. Co-Applicant Income
6. Property Area
7. Education Level

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Serve the web application |
| `GET /api/status` | GET | Model + watsonx.ai status |
| `POST /api/predict` | POST | Run loan prediction |
| `POST /api/chat` | POST | AI chat (Granite or fallback) |
| `GET /api/history` | GET | Session prediction history |
| `GET /api/dataset-stats` | GET | Dataset statistics for charts |
| `POST /api/train` | POST | Retrain model (admin) |

### Sample Prediction Request

```json
POST /api/predict
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 6000,
  "CoapplicantIncome": 2000,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

### Sample Prediction Response

```json
{
  "approved": true,
  "confidence": 87.4,
  "approval_probability": 87.4,
  "rejection_probability": 12.6,
  "dti_ratio": 5.21,
  "monthly_emi": 416.67,
  "total_income": 8000,
  "feature_impact": [
    { "feature": "Credit History", "importance": 34.2, "value": "1" },
    { "feature": "Applicant Income", "importance": 22.1, "value": "6000" }
  ]
}
```

---

## ☁️ Deployment

### Deploy to IBM Code Engine

```bash
# Build Docker image
docker build -t loan-advisor-ai .

# Tag and push to IBM Container Registry
ibmcloud cr login
ibmcloud cr build --file Dockerfile --registry-secret mysecret \
  --tag us.icr.io/mynamespace/loan-advisor-ai:latest .

# Deploy to Code Engine
ibmcloud ce application create \
  --name loan-advisor-ai \
  --image us.icr.io/mynamespace/loan-advisor-ai:latest \
  --env-from-secret loan-advisor-secrets \
  --port 5000
```

### Deploy with Docker Compose

```bash
docker-compose up -d
```

### Deploy to Heroku

```bash
heroku create loan-advisor-ai
heroku config:set IBM_API_KEY=xxx WATSONX_PROJECT_ID=yyy
git push heroku main
```

### Deploy to Azure App Service / AWS Elastic Beanstalk

Use `gunicorn` as the production WSGI server (included in requirements.txt):

```bash
gunicorn --workers 4 --bind 0.0.0.0:$PORT app:app
```

---

## 🔒 Security Best Practices

- ✅ API keys stored in `.env` (never in source code)
- ✅ `.env` added to `.gitignore`
- ✅ Session-based history (no database, no PII persistence)
- ✅ CORS enabled via flask-cors
- ✅ Input validation on all endpoints
- ✅ No sensitive data logged

---

## 🛠️ Development Notes

**Fallback Mode:** If `IBM_API_KEY` is not set, the app runs in intelligent fallback mode with
rule-based responses for all common loan questions. The ML prediction still works fully.

**Model Retraining:** To retrain with new data:
```bash
# Replace data/loan_approval.csv with new dataset
curl -X POST http://localhost:5000/api/train
```

**Custom Dataset:** Ensure your CSV has these columns (column names are case-sensitive):
`Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome,`
`LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status`

---

## 📄 License

MIT License — Free to use for educational and commercial projects.

---

*Built with ❤️ using IBM BOB, IBM watsonx.ai, and Flask*
