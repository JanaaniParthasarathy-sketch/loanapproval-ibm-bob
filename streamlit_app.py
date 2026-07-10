"""
Loan Approval Prediction — Streamlit App
IBM watsonx.ai + Granite Models Integration
"""
import os
import sys
import json
import datetime

import streamlit as st
from dotenv import load_dotenv

# ── ensure imports work regardless of cwd ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from loan_model import predict_loan, train_model, load_model
from agent_config import AGENT_INSTRUCTIONS

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Streamlit Cloud secrets → os.environ (no-op locally) ───────────────────
try:
    for _k, _v in st.secrets.items():
        if _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:
    pass

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanAdvisor AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "applicant_data" not in st.session_state:
    st.session_state.applicant_data = {}
if "history" not in st.session_state:
    st.session_state.history = []

# ────────────────────────────────────────────────────────────────────────────
# IBM watsonx.ai initialisation (cached)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_watsonx():
    try:
        from ibm_watsonx_ai import APIClient, Credentials
        api_key   = os.getenv("IBM_API_KEY", "")
        url       = os.getenv("WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")
        project_id = os.getenv("WATSONX_PROJECT_ID", "")
        if not api_key or api_key == "your_ibm_cloud_api_key_here":
            return None
        credentials = Credentials(url=url, api_key=api_key)
        client = APIClient(credentials=credentials, project_id=project_id)
        return client
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_model():
    """Load or train the ML model (cached across reruns)."""
    model, le, sc = load_model()
    if model is None:
        import subprocess
        dataset = os.path.join(os.path.dirname(__file__), "data", "loan_approval.csv")
        if not os.path.exists(dataset):
            subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(__file__), "generate_dataset.py")],
                check=True,
                cwd=os.path.dirname(__file__),
            )
        train_model(os.path.join(os.path.dirname(__file__), "data", "loan_approval.csv"))
        model, le, sc = load_model()
    return model


# ────────────────────────────────────────────────────────────────────────────
# Granite / fallback chat helpers
# ────────────────────────────────────────────────────────────────────────────
def call_granite(client, prompt: str, max_tokens: int = 600) -> str | None:
    if client is None:
        return None
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        model_id   = os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-chat-v2")
        project_id = os.getenv("WATSONX_PROJECT_ID", "")
        params = {
            GenParams.MAX_NEW_TOKENS: max_tokens,
            GenParams.MIN_NEW_TOKENS: 30,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_P: 0.95,
            GenParams.REPETITION_PENALTY: 1.1,
        }
        m = ModelInference(
            model_id=model_id, params=params,
            credentials=client.credentials, project_id=project_id,
        )
        return m.generate_text(prompt=prompt).strip()
    except Exception:
        return None


def build_chat_prompt(user_message: str, loan_context: dict = None, history: list = None) -> str:
    system_block  = f"<|system|>\n{AGENT_INSTRUCTIONS}\n<|end|>\n"
    context_block = ""
    if loan_context:
        context_block = (
            "\n<|context|>\n"
            f"Current Applicant Data:\n{json.dumps(loan_context, indent=2)}\n"
            "<|end|>\n"
        )
    history_block = ""
    if history:
        for msg in history[-4:]:
            role = "user" if msg["role"] == "user" else "assistant"
            history_block += f"<|{role}|>\n{msg['content']}\n<|end|>\n"
    return f"{system_block}{context_block}{history_block}<|user|>\n{user_message}\n<|end|>\n<|assistant|>\n"


def get_fallback_response(user_message: str, loan_context: dict = None) -> str:
    msg = user_message.lower()
    if loan_context and loan_context.get("prediction"):
        pred     = loan_context["prediction"]
        approved = pred.get("approved", False)
        conf     = pred.get("confidence", 0)
        dti      = pred.get("dti_ratio", 0)
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
                "Fill in your applicant details on the left and click **Predict** to get started!")
    return ("I'm here to help with your loan application questions. "
            "You can ask me about loan eligibility, credit scores, required documents, "
            "EMI calculations, or ways to improve your loan approval chances. "
            "For a detailed analysis, fill in your applicant profile and run a prediction.")


# ────────────────────────────────────────────────────────────────────────────
# App UI
# ────────────────────────────────────────────────────────────────────────────
watsonx_client = init_watsonx()
WATSONX_AVAILABLE = watsonx_client is not None

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🏦 LoanAdvisor AI")
st.caption(
    f"IBM Granite-powered loan prediction · "
    f"{'✅ watsonx.ai connected' if WATSONX_AVAILABLE else '⚠️ Running in fallback mode'}"
)
st.divider()

# ── Layout: left = form, right = result + chat ───────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

# ────────────────────────────────────────────────────────────────────────────
# LEFT — Applicant Form
# ────────────────────────────────────────────────────────────────────────────
with left_col:
    st.subheader("📋 Applicant Profile")

    with st.form("loan_form"):
        c1, c2 = st.columns(2)
        with c1:
            gender        = st.selectbox("Gender",        ["Male", "Female"])
            married       = st.selectbox("Married",       ["Yes", "No"])
            dependents    = st.selectbox("Dependents",    ["0", "1", "2", "3+"])
            education     = st.selectbox("Education",     ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with c2:
            applicant_income    = st.number_input("Applicant Income (₹/mo)",   min_value=0, value=5000,  step=500)
            coapplicant_income  = st.number_input("Co-Applicant Income (₹/mo)", min_value=0, value=0,     step=500)
            loan_amount         = st.number_input("Loan Amount (₹ thousands)",  min_value=1, value=150,   step=10)
            loan_amount_term    = st.selectbox("Loan Term (months)",
                                               [120, 180, 240, 300, 360, 480], index=4)
            credit_history      = st.selectbox("Credit History",
                                               [1.0, 0.0],
                                               format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor (0.0)")

        submitted = st.form_submit_button("🔍 Predict Loan Eligibility", use_container_width=True)

    if submitted:
        # Ensure model is ready
        with st.spinner("Loading model…"):
            get_model()

        applicant_data = {
            "Gender":             gender,
            "Married":            married,
            "Dependents":         dependents,
            "Education":          education,
            "Self_Employed":      self_employed,
            "ApplicantIncome":    applicant_income,
            "CoapplicantIncome":  coapplicant_income,
            "LoanAmount":         loan_amount,
            "Loan_Amount_Term":   loan_amount_term,
            "Credit_History":     credit_history,
            "Property_Area":      property_area,
        }

        with st.spinner("Running prediction…"):
            result = predict_loan(applicant_data)

        if "error" in result:
            st.error(f"Prediction error: {result['error']}")
        else:
            st.session_state.prediction_result = result
            st.session_state.applicant_data    = applicant_data
            # Save to history
            entry = {
                "id":        len(st.session_state.history) + 1,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "applicant": {"income": applicant_income, "loan_amount": loan_amount,
                              "credit_history": credit_history, "education": education,
                              "property_area": property_area},
                "approved":   result["approved"],
                "confidence": result["confidence"],
            }
            st.session_state.history = st.session_state.history[-9:] + [entry]
            st.success("Prediction complete! See results →")

    # ── Prediction History ────────────────────────────────────────────────
    if st.session_state.history:
        st.subheader("📜 Prediction History")
        for h in reversed(st.session_state.history[-5:]):
            badge = "✅" if h["approved"] else "❌"
            st.markdown(
                f"{badge} **#{h['id']}** · {h['timestamp']} · "
                f"Income ₹{h['applicant']['income']:,} · "
                f"Loan ₹{h['applicant']['loan_amount']}K · "
                f"Confidence {h['confidence']}%"
            )

# ────────────────────────────────────────────────────────────────────────────
# RIGHT — Prediction Result + Chat
# ────────────────────────────────────────────────────────────────────────────
with right_col:

    # ── Prediction Result ─────────────────────────────────────────────────
    result = st.session_state.prediction_result
    if result:
        approved = result["approved"]
        conf     = result["confidence"]

        if approved:
            st.success(f"✅ **LOAN APPROVED** — Confidence: {conf:.1f}%")
        else:
            st.error(f"❌ **LOAN NOT APPROVED** — Confidence: {conf:.1f}%")

        m1, m2, m3 = st.columns(3)
        m1.metric("Approval Probability", f"{result['approval_probability']:.1f}%")
        m2.metric("DTI Ratio",            f"{result['dti_ratio']:.1f}%",
                  delta="Good" if result['dti_ratio'] < 40 else "High",
                  delta_color="normal" if result['dti_ratio'] < 40 else "inverse")
        m3.metric("Monthly EMI",          f"₹{result['monthly_emi']:,.0f}")

        st.subheader("📊 Feature Impact")
        for fi in result["feature_impact"]:
            st.progress(
                min(int(fi["importance"]), 100),
                text=f"{fi['feature']} — {fi['importance']:.1f}%  (value: {fi['value']})",
            )

        st.divider()

    # ── AI Chat ────────────────────────────────────────────────────────────
    st.subheader("💬 LoanAdvisor AI Chat")

    # Display history
    chat_container = st.container(height=340)
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Ask me anything about loans, credit scores, or your prediction result!")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about your loan or financial planning…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        loan_context = None
        if st.session_state.prediction_result:
            loan_context = {
                "prediction": st.session_state.prediction_result,
                "applicant":  st.session_state.applicant_data,
            }

        with st.spinner("Thinking…"):
            ai_response = None
            if WATSONX_AVAILABLE:
                prompt      = build_chat_prompt(user_input, loan_context, st.session_state.chat_history)
                ai_response = call_granite(watsonx_client, prompt)
            if not ai_response:
                ai_response = get_fallback_response(user_input, loan_context)

        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# Sidebar — dataset stats
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📈 Dataset Stats")
    csv_path = os.path.join(os.path.dirname(__file__), "data", "loan_approval.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        st.metric("Total Records",  len(df))
        st.metric("Approval Rate",  f"{(df['Loan_Status']=='Y').mean()*100:.1f}%")
        st.metric("Avg Income",     f"₹{int(df['ApplicantIncome'].mean()):,}")
        st.metric("Avg Loan",       f"₹{int(df['LoanAmount'].mean())}K")

        st.subheader("Property Area")
        st.bar_chart(df["Property_Area"].value_counts())

        st.subheader("Education")
        st.bar_chart(df["Education"].value_counts())
    else:
        st.info("Run a prediction to generate the dataset.")

    st.divider()
    st.caption("LoanAdvisor AI v1.0 · IBM watsonx.ai")
    if st.button("🔄 Retrain Model", use_container_width=True):
        with st.spinner("Training…"):
            try:
                get_model.clear()
                if not os.path.exists(csv_path):
                    import subprocess
                    subprocess.run(
                        [sys.executable, os.path.join(os.path.dirname(__file__), "generate_dataset.py")],
                        check=True, cwd=os.path.dirname(__file__),
                    )
                acc = train_model(csv_path)[3]
                st.success(f"Retrained! Accuracy: {acc*100:.1f}%")
            except Exception as e:
                st.error(str(e))
