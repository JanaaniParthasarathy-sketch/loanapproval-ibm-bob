"""
Loan Approval Prediction — Flask Backend
IBM watsonx.ai + Granite Models Integration
"""
import os
import json
import datetime
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv

from loan_model import predict_loan, train_model, load_model
from agent_config import AGENT_INSTRUCTIONS

# ─── Load environment ────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "loan-advisor-ai-secret-2024")
CORS(app)

# ─── IBM watsonx.ai client ───────────────────────────────────────────────────
watsonx_client = None
WATSONX_AVAILABLE = False


def init_watsonx():
    global watsonx_client, WATSONX_AVAILABLE
    try:
        from ibm_watsonx_ai import APIClient, Credentials
        api_key = os.getenv("IBM_API_KEY", "")
        url = os.getenv("WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")
        project_id = os.getenv("WATSONX_PROJECT_ID", "")

        if not api_key or api_key == "your_ibm_cloud_api_key_here":
            print("⚠  IBM API Key not configured — AI chat will use fallback mode.")
            return

        credentials = Credentials(url=url, api_key=api_key)
        watsonx_client = APIClient(credentials=credentials, project_id=project_id)
        WATSONX_AVAILABLE = True
        print("✅ IBM watsonx.ai connected successfully.")
    except Exception as e:
        print(f"⚠  watsonx.ai init failed: {e}. Using fallback mode.")


def call_granite(prompt: str, max_tokens: int = 600) -> str:
    """Call IBM Granite model via watsonx.ai SDK."""
    if not WATSONX_AVAILABLE or watsonx_client is None:
        return None

    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        model_id = os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-chat-v2")
        project_id = os.getenv("WATSONX_PROJECT_ID", "")

        parameters = {
            GenParams.MAX_NEW_TOKENS: max_tokens,
            GenParams.MIN_NEW_TOKENS: 30,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_P: 0.95,
            GenParams.REPETITION_PENALTY: 1.1,
        }

        model = ModelInference(
            model_id=model_id,
            params=parameters,
            credentials=watsonx_client.credentials,
            project_id=project_id
        )
        response = model.generate_text(prompt=prompt)
        return response.strip()
    except Exception as e:
        print(f"Granite call error: {e}")
        return None


def build_chat_prompt(user_message: str, loan_context: dict = None, history: list = None) -> str:
    """Build the full prompt with system instructions + context + history."""
    system_block = f"<|system|>\n{AGENT_INSTRUCTIONS}\n<|end|>\n"

    context_block = ""
    if loan_context:
        context_block = (
            "\n<|context|>\n"
            f"Current Applicant Data:\n{json.dumps(loan_context, indent=2)}\n"
            "<|end|>\n"
        )

    history_block = ""
    if history:
        for msg in history[-4:]:  # last 4 turns
            role = "user" if msg["role"] == "user" else "assistant"
            history_block += f"<|{role}|>\n{msg['content']}\n<|end|>\n"

    return f"{system_block}{context_block}{history_block}<|user|>\n{user_message}\n<|end|>\n<|assistant|>\n"


def get_fallback_response(user_message: str, loan_context: dict = None) -> str:
    """Smart rule-based fallback when watsonx.ai is unavailable."""
    msg = user_message.lower()

    if loan_context and loan_context.get("prediction"):
        pred = loan_context["prediction"]
        approved = pred.get("approved", False)
        conf = pred.get("confidence", 0)
        dti = pred.get("dti_ratio", 0)

        if any(w in msg for w in ["why", "reason", "factor", "explain", "analysis"]):
            if approved:
                return (
                    f"✅ **Loan Approved ({conf:.0f}% confidence)**\n\n"
                    "**Key Positive Factors:**\n"
                    "• Strong credit history indicating responsible repayment behavior\n"
                    f"• Debt-to-income ratio of {dti:.1f}% is within acceptable limits\n"
                    "• Income level supports the requested loan amount\n"
                    "• Employment profile meets lender requirements\n\n"
                    "**Next Steps:** Proceed with formal application, gather KYC documents, "
                    "and contact your preferred bank for interest rate negotiation."
                )
            else:
                return (
                    f"❌ **Loan Not Approved ({conf:.0f}% confidence)**\n\n"
                    "**Key Factors Affecting Decision:**\n"
                    "• Credit history score needs improvement\n"
                    f"• Debt-to-income ratio of {dti:.1f}% may be high\n"
                    "• Income-to-loan ratio requires adjustment\n\n"
                    "**Improvement Suggestions:**\n"
                    "1. Repay existing debts to improve credit score\n"
                    "2. Add a co-applicant with stable income\n"
                    "3. Reduce loan amount or extend repayment term\n"
                    "4. Revisit application in 6-12 months after improving finances"
                )

    if any(w in msg for w in ["credit", "cibil", "score"]):
        return ("A credit score (CIBIL) of **750+** is ideal for loan approval. "
                "Pay all EMIs on time, avoid multiple loan inquiries, and keep credit utilization below 30%. "
                "Your score typically improves within 6-12 months of consistent repayment.")

    if any(w in msg for w in ["dti", "debt", "income ratio", "emi"]):
        return ("The Debt-to-Income (DTI) ratio compares your monthly EMI obligations to your income. "
                "A DTI **below 40%** is considered healthy by most lenders. "
                "For example, if your income is ₹50,000/month, your total EMIs should not exceed ₹20,000.")

    if any(w in msg for w in ["document", "kyc", "require"]):
        return ("**Standard Loan Documents Required:**\n"
                "• Identity Proof: Aadhaar/PAN/Passport\n"
                "• Address Proof: Utility bill/Aadhaar\n"
                "• Income Proof: 3 months salary slips or ITR (2 years for self-employed)\n"
                "• Bank Statements: Last 6 months\n"
                "• Property Documents: For secured loans")

    if any(w in msg for w in ["interest", "rate", "roi"]):
        return ("Loan interest rates vary by type and credit profile:\n"
                "• **Home Loans:** 8.5% – 10.5% p.a.\n"
                "• **Personal Loans:** 10.5% – 18% p.a.\n"
                "• **Business Loans:** 12% – 20% p.a.\n"
                "A higher CIBIL score (750+) and good income profile qualify you for lower rates.")

    if any(w in msg for w in ["hello", "hi", "hey", "help"]):
        return ("Hello! 👋 I'm **LoanAdvisor AI**, your intelligent banking assistant.\n\n"
                "I can help you:\n"
                "• **Analyze** your loan application\n"
                "• **Explain** approval/rejection reasons\n"
                "• **Suggest** ways to improve eligibility\n"
                "• **Answer** questions about loans and financial planning\n\n"
                "Fill in your applicant details and click **Predict** to get started!")

    return ("I'm here to help with your loan application questions. "
            "You can ask me about loan eligibility, credit scores, required documents, "
            "EMI calculations, or ways to improve your loan approval chances. "
            "For a detailed analysis, please fill in your applicant profile and run a prediction.")


# ─── Initialize model on startup ────────────────────────────────────────────
def ensure_model_ready():
    model, _, _ = load_model()
    if model is None:
        print("📊 No trained model found. Generating dataset and training...")
        import subprocess, sys
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)
        train_model()
        print("✅ Model trained and ready.")
    else:
        print("✅ Trained model loaded.")


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    model, _, _ = load_model()
    return jsonify({
        "model_ready": model is not None,
        "watsonx_available": WATSONX_AVAILABLE,
        "granite_model": os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-chat-v2"),
        "version": "1.0.0"
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400

    result = predict_loan(data)
    if "error" in result:
        return jsonify(result), 500

    # Store prediction in session history
    if "history" not in session:
        session["history"] = []

    history_entry = {
        "id": len(session["history"]) + 1,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "applicant": {
            "income": data.get("ApplicantIncome", 0),
            "loan_amount": data.get("LoanAmount", 0),
            "credit_history": data.get("Credit_History", 0),
            "education": data.get("Education", ""),
            "property_area": data.get("Property_Area", "")
        },
        "approved": result["approved"],
        "confidence": result["confidence"]
    }
    session["history"] = session.get("history", [])[-9:] + [history_entry]
    session.modified = True

    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    loan_context = data.get("loan_context", None)
    chat_history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Try IBM Granite first
    ai_response = None
    if WATSONX_AVAILABLE:
        prompt = build_chat_prompt(user_message, loan_context, chat_history)
        ai_response = call_granite(prompt)

    # Fallback to rule-based response
    if not ai_response:
        ai_response = get_fallback_response(user_message, loan_context)

    return jsonify({
        "response": ai_response,
        "model": "ibm/granite-13b-chat-v2" if WATSONX_AVAILABLE else "fallback",
        "timestamp": datetime.datetime.now().strftime("%H:%M")
    })


@app.route("/api/history")
def get_history():
    return jsonify(session.get("history", []))


@app.route("/api/train", methods=["POST"])
def retrain():
    """Endpoint to retrain the model (admin use)."""
    try:
        import os
        csv_path = "data/loan_approval.csv"
        if not os.path.exists(csv_path):
            import subprocess, sys
            subprocess.run([sys.executable, "generate_dataset.py"], check=True)
        model, _, _, accuracy = train_model(csv_path)
        return jsonify({"success": True, "accuracy": round(accuracy * 100, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset-stats")
def dataset_stats():
    """Return dataset statistics for dashboard charts."""
    try:
        import pandas as pd
        df = pd.read_csv("data/loan_approval.csv")
        stats = {
            "total_records": len(df),
            "approval_rate": round((df["Loan_Status"] == "Y").mean() * 100, 1),
            "avg_income": int(df["ApplicantIncome"].mean()),
            "avg_loan": int(df["LoanAmount"].mean()),
            "education_dist": df["Education"].value_counts().to_dict(),
            "property_dist": df["Property_Area"].value_counts().to_dict(),
            "credit_dist": {
                "Good (1.0)": int((df["Credit_History"] == 1.0).sum()),
                "Poor (0.0)": int((df["Credit_History"] == 0.0).sum())
            },
            "income_buckets": {
                "< 2500": int((df["ApplicantIncome"] < 2500).sum()),
                "2500–5000": int(((df["ApplicantIncome"] >= 2500) & (df["ApplicantIncome"] < 5000)).sum()),
                "5000–10000": int(((df["ApplicantIncome"] >= 5000) & (df["ApplicantIncome"] < 10000)).sum()),
                "> 10000": int((df["ApplicantIncome"] >= 10000).sum()),
            }
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── App Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_watsonx()
    ensure_model_ready()
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)
