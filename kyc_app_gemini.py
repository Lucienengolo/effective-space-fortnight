"""
kyc_app.py — E-KYC Identity Verification System (Streamlit + Gemini)
──────────────────────────────────────────────────────────────────────
Run:
    streamlit run kyc_app.py

Set your Gemini API key in .streamlit/secrets.toml:
    GEMINI_API_KEY = "AIza..."
or via environment variable:
    export GEMINI_API_KEY=AIza...

Get a free Gemini API key at: https://aistudio.google.com/app/apikey
"""

import os, base64, json, time
from io import BytesIO

import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-KYC Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: linear-gradient(135deg, #020817 0%, #0a0f1e 50%, #020817 100%);
    min-height: 100vh;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

/* ── Custom header bar ── */
.kyc-header {
    background: linear-gradient(90deg, #0d1b2e 0%, #0a1628 100%);
    border: 1px solid rgba(0,200,255,0.15);
    border-radius: 14px;
    padding: 18px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}
.kyc-header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #00c8ff;
    letter-spacing: 0.05em;
}
.kyc-header-badge {
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    color: #00c8ff;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Step indicator ── */
.step-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin-bottom: 28px;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.step-circle {
    width: 40px; height: 40px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem;
    border: 2px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.3);
    transition: all 0.3s;
}
.step-circle.active {
    background: rgba(0,200,255,0.15);
    border-color: #00c8ff;
    color: #00c8ff;
    box-shadow: 0 0 18px rgba(0,200,255,0.4);
}
.step-circle.done {
    background: rgba(0,229,160,0.15);
    border-color: #00e5a0;
    color: #00e5a0;
}
.step-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.3);
    font-weight: 600;
}
.step-label.active { color: #00c8ff; }
.step-label.done   { color: #00e5a0; }
.step-line {
    flex: 1;
    height: 2px;
    background: rgba(255,255,255,0.07);
    margin: 0 8px;
    margin-bottom: 20px;
    min-width: 60px;
    transition: background 0.4s;
}
.step-line.done { background: #00e5a0; }

/* ── Cards ── */
.kyc-card {
    background: rgba(13,27,46,0.8);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 16px;
}
.kyc-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 4px;
}
.kyc-card-sub {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.38);
    margin-bottom: 16px;
}

/* ── Upload area ── */
.stFileUploader > div > div {
    background: rgba(10,22,40,0.8) !important;
    border: 2px dashed rgba(0,200,255,0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s !important;
}
.stFileUploader > div > div:hover {
    border-color: rgba(0,200,255,0.55) !important;
}

/* ── Camera / snapshot ── */
div[data-testid="stCameraInput"] > div {
    background: #0a0f1e !important;
    border: 2px solid rgba(0,200,255,0.2) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Fields ── */
.field-block {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: all 0.3s;
}
.field-block.filled {
    background: rgba(0,229,160,0.06);
    border-color: rgba(0,229,160,0.25);
}
.field-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.3);
    margin-bottom: 4px;
    font-weight: 600;
}
.field-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #fff;
    font-family: 'Space Grotesk', sans-serif;
}
.field-value.filled { color: #00e5a0; }

/* ── Verdict banner ── */
.verdict-verified {
    background: linear-gradient(135deg, rgba(0,229,160,0.12), rgba(0,200,255,0.06));
    border: 1px solid rgba(0,229,160,0.35);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    margin-bottom: 20px;
}
.verdict-failed {
    background: linear-gradient(135deg, rgba(255,80,80,0.12), rgba(255,120,60,0.06));
    border: 1px solid rgba(255,80,80,0.35);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    margin-bottom: 20px;
}
.verdict-icon { font-size: 3rem; margin-bottom: 8px; }
.verdict-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.verdict-title.ok   { color: #00e5a0; }
.verdict-title.fail { color: #ff5050; }
.verdict-reason {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
}
.conf-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 8px 4px 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.conf-high   { background: rgba(0,229,160,0.15); color: #00e5a0; border: 1px solid rgba(0,229,160,0.3); }
.conf-medium { background: rgba(255,190,50,0.15); color: #ffbe32; border: 1px solid rgba(255,190,50,0.3); }
.conf-low    { background: rgba(255,80,80,0.15);  color: #ff5050; border: 1px solid rgba(255,80,80,0.3); }

/* ── Comparison photos ── */
.photo-compare {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 16px 0;
}
.photo-box {
    text-align: center;
}
.photo-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.3);
    margin-bottom: 6px;
}
.compare-sym {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 16px;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
}

/* ── Progress / spinner ── */
.stSpinner > div { border-top-color: #00c8ff !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Autofill badge ── */
.autofill-badge {
    display: inline-block;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.65rem;
    color: #00c8ff;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    vertical-align: middle;
    margin-left: 8px;
}

/* ── Timeline log ── */
.log-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
}
.log-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = dict(
        step=1,
        doc_pil=None,
        doc_b64=None,
        doc_mime="image/jpeg",
        live_b64=None,
        live_pil=None,
        extracted={},
        face_match=None,
        validated=False,
        log=[],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
S = st.session_state


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_client():
    key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not key:
        st.error("Missing Gemini API Key")
        st.stop()

    return genai.Client(api_key=key)


# ─────────────────────────────────────────────────────────────────────────────
# AI FUNCTIONS  (powered by Google Gemini Vision)
# ─────────────────────────────────────────────────────────────────────────────

# Gemini model used for both tasks.
# gemini-2.0-flash is fast, cheap, and has strong vision / OCR capabilities.
GEMINI_MODEL = "gemini-2.0-flash"


def _image_part(b64: str, mime: str) -> types.Part:
    """
    Build a Gemini multimodal Part from a base64-encoded image.

    Gemini's SDK accepts images as inline_data blobs — raw bytes with a MIME
    type. We decode our stored base64 string back to bytes here.
    """
    return types.Part.from_bytes(
        data=base64.b64decode(b64),
        mime_type=mime,
    )


def extract_document_info(client: genai.Client, doc_b64: str, mime: str) -> dict:
    """
    Send the identity document image to Gemini Vision for OCR extraction.

    Gemini receives:
      - Part 1: the document image (inline bytes)
      - Part 2: the instruction text with the exact JSON schema

    The system instruction tells the model to act as an OCR engine and
    return ONLY raw JSON — no markdown fences, no explanation.
    We strip any accidental backticks before parsing.
    """
    prompt_text = (
        "Extract every visible identity field from this document.\n"
        "Return ONLY this JSON object (use null for missing fields, no markdown):\n"
        '{"full_name":null,"date_of_birth":null,"id_number":null,'
        '"document_type":null,"nationality":null,"gender":null,'
        '"issue_date":null,"expiry_date":null,"address":null,'
        '"place_of_birth":null,"mrz":null}'
    )

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            _image_part(doc_b64, mime),   # Image block — the document
            types.Part.from_text(text=prompt_text),  # Text block — the instruction
        ],
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a precision OCR and document intelligence engine. "
                "Read identity documents and extract fields as structured JSON. "
                "Return ONLY raw JSON — no markdown fences, no explanation."
            ),
            temperature=0,        # Deterministic output — no creative variation
            max_output_tokens=1000,
        ),
    )

    raw = resp.text.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        # Graceful fallback if the model returned unexpected text
        return {"full_name": raw[:60], "document_type": "ID Document"}


def compare_faces(client: genai.Client, doc_b64: str, live_b64: str, mime: str) -> dict:
    """
    Send both images to Gemini Vision and ask it to compare the faces.

    Gemini receives three parts:
      - Part 1: identity document image  (the reference face)
      - Part 2: live webcam selfie       (the face to verify)
      - Part 3: comparison instruction + JSON schema

    Gemini reasons about facial features, lighting, angle, and similarity
    and returns a structured JSON verdict with a match boolean, confidence
    level, numeric similarity score, and a human-readable reason.
    """
    prompt_text = (
        "Image 1 is an identity document. Image 2 is a live webcam selfie.\n"
        "Carefully compare the face in Image 1 with the face in Image 2.\n"
        "Return ONLY this JSON (no markdown, no explanation):\n"
        '{"match":true,"confidence":"High","similarity_score":85,'
        '"reason":"one concise sentence explaining the verdict",'
        '"liveness_note":"observation about the live photo quality"}'
    )

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            _image_part(doc_b64, mime),               # Document image
            _image_part(live_b64, "image/jpeg"),       # Live selfie
            types.Part.from_text(text=prompt_text),   # Instruction
        ],
        config=types.GenerateContentConfig(
            system_instruction=(
                "This project uses Gemini Vision for document understanding and visual similarity analysis between the ID photo and a live selfie. It demonstrates an AI-assisted e-KYC workflow."
                "Analyze two images and determine if they show the same person. "
                "Return ONLY raw JSON — no markdown."
            ),
            temperature=0,
            max_output_tokens=400,
        ),
    )

    raw = resp.text.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "match": False,
            "confidence": "Low",
            "similarity_score": 0,
            "reason": raw[:120],
            "liveness_note": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="kyc-header">
        <div class="kyc-header-title">⬡ &nbsp; E-KYC IDENTITY VERIFICATION SYSTEM</div>
        <div class="kyc-header-badge">🔒 &nbsp; Powered by Gemini AI</div>
    </div>
    """, unsafe_allow_html=True)


def render_stepper(current: int):
    def cls(n):
        if n < current:  return "done"
        if n == current: return "active"
        return ""

    def line_cls(n):
        return "done" if n < current else ""

    steps = [(1, "Document"), (2, "Face Scan"), (3, "Validate")]
    html = '<div class="step-bar">'
    for i, (n, label) in enumerate(steps):
        c = cls(n)
        icon = "✓" if c == "done" else str(n)
        html += f"""
        <div class="step-item">
            <div class="step-circle {c}">{icon}</div>
            <div class="step-label {c}">{label}</div>
        </div>"""
        if i < len(steps) - 1:
            html += f'<div class="step-line {line_cls(n)}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_field(label: str, value, filled=False):
    val_html = str(value) if value else "—"
    filled_cls = "filled" if (value and filled) else ""
    return f"""
    <div class="field-block {'filled' if value and filled else ''}">
        <div class="field-label">{label}</div>
        <div class="field-value {filled_cls}">{val_html if value else '<span style="color:rgba(255,255,255,0.15);font-style:italic">Not found</span>'}</div>
    </div>"""


def render_log():
    if not S.log:
        return
    html = ""
    for item in S.log:
        html += f"""
        <div class="log-item">
            <div class="log-dot" style="background:{item['color']}"></div>
            <div><span style="color:rgba(255,255,255,0.25);margin-right:8px">{item['ts']}</span>{item['msg']}</div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Document Upload
# ─────────────────────────────────────────────────────────────────────────────
def render_step1():
    col_main, col_info = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-title">📄 &nbsp; Upload Identity Document</div>', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-sub">Accepted: National ID card · Passport · Driver\'s license (JPG, PNG, BMP)</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop your document here",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            mime = "image/jpeg" if uploaded.type in ("image/jpeg", "image/jpg") else "image/png"
            S.doc_pil  = img
            S.doc_b64  = pil_to_b64(img, "JPEG")
            S.doc_mime = mime

            st.image(img, caption="✓  Document loaded", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if S.doc_b64:
            if st.button("Continue to Face Scan →", type="primary", use_container_width=True):
                S.step = 2
                st.rerun()
        else:
            st.button("Continue to Face Scan →", disabled=True, use_container_width=True)

    with col_info:
        st.markdown("""
        <div class="kyc-card">
            <div class="kyc-card-title">How It Works</div>
            <div style="margin-top:14px">
                <div style="display:flex;gap:10px;margin-bottom:14px;align-items:flex-start">
                    <div style="width:26px;height:26px;border-radius:50%;background:rgba(0,200,255,0.2);border:1px solid #00c8ff;display:flex;align-items:center;justify-content:center;color:#00c8ff;font-weight:700;font-size:0.8rem;flex-shrink:0">1</div>
                    <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);padding-top:4px">Upload your government-issued identity document.</div>
                </div>
                <div style="display:flex;gap:10px;margin-bottom:14px;align-items:flex-start">
                    <div style="width:26px;height:26px;border-radius:50%;background:rgba(0,200,255,0.2);border:1px solid #00c8ff;display:flex;align-items:center;justify-content:center;color:#00c8ff;font-weight:700;font-size:0.8rem;flex-shrink:0">2</div>
                    <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);padding-top:4px">Take a live selfie via the webcam capture.</div>
                </div>
                <div style="display:flex;gap:10px;align-items:flex-start">
                    <div style="width:26px;height:26px;border-radius:50%;background:rgba(0,200,255,0.2);border:1px solid #00c8ff;display:flex;align-items:center;justify-content:center;color:#00c8ff;font-weight:700;font-size:0.8rem;flex-shrink:0">3</div>
                    <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);padding-top:4px">Gemini AI reads your document, compares faces & auto-fills all fields.</div>
                </div>
            </div>
        </div>
        <div class="kyc-card" style="margin-top:0">
            <div class="kyc-card-title" style="font-size:0.85rem">🔐 Privacy Notice</div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.35);margin-top:8px;line-height:1.6">
                Your document images are processed in real-time and are never stored on any server. All AI analysis happens transiently via the Google Gemini API.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Face Capture
# ─────────────────────────────────────────────────────────────────────────────
def render_step2():
    col_main, col_preview = st.columns([3, 2], gap="large")

    with col_main:
        st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-title">📷 &nbsp; Live Face Capture</div>', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-sub">Allow camera access · Center your face · Take a clear selfie</div>', unsafe_allow_html=True)

        photo = st.camera_input("Take a selfie", label_visibility="collapsed")

        if photo:
            img = Image.open(photo).convert("RGB")
            S.live_pil = img
            S.live_b64 = pil_to_b64(img, "JPEG")

        st.markdown('</div>', unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("← Back to Document", use_container_width=True):
                S.step = 1
                st.rerun()
        with btn_col2:
            if S.live_b64:
                if st.button("Verify Identity →", type="primary", use_container_width=True):
                    S.step = 3
                    st.rerun()
            else:
                st.button("Verify Identity →", disabled=True, use_container_width=True)

    with col_preview:
        st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-title">Document on File</div>', unsafe_allow_html=True)
        if S.doc_pil:
            st.image(S.doc_pil, caption="Uploaded document", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if S.live_pil:
            st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
            st.markdown('<div class="kyc-card-title" style="color:#00e5a0">✓ &nbsp; Photo Captured</div>', unsafe_allow_html=True)
            st.image(S.live_pil, caption="Live selfie", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kyc-card" style="text-align:center;padding:32px">
                <div style="font-size:2.5rem;margin-bottom:10px">🤳</div>
                <div style="color:rgba(255,255,255,0.3);font-size:0.82rem">
                    Awaiting selfie capture…<br>
                    <span style="font-size:0.73rem">Make sure your face is well-lit and clearly visible</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Verification & Results
# ─────────────────────────────────────────────────────────────────────────────
def render_step3():
    # ── Run AI if not yet done ───────────────────────────────────────────────
    if not S.validated:
        client = get_client()

        progress_placeholder = st.empty()
        log_placeholder      = st.empty()

        with progress_placeholder.container():
            st.markdown("""
            <div class="kyc-card" style="text-align:center;padding:40px">
                <div style="font-size:2.5rem;margin-bottom:12px">🧠</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:600;color:#00c8ff;margin-bottom:6px">
                    AI Processing Your Identity
                </div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.82rem">Please wait while Gemini analyzes your documents…</div>
            </div>
            """, unsafe_allow_html=True)

        # Step A — Extract document info
        add_log("📄 Reading identity document with Gemini Vision…", "#00c8ff")
        with log_placeholder.container():
            render_log()

        with st.spinner("Extracting document information…"):
            info = extract_document_info(client, S.doc_b64, S.doc_mime)
            S.extracted = info

        add_log(f"✓ Document type detected: {info.get('document_type','ID Document')}", "#00e5a0")
        add_log(f"✓ Name extracted: {info.get('full_name','—')}", "#00e5a0")
        with log_placeholder.container():
            render_log()

        # Step B — Face comparison
        add_log("🔍 Comparing document face vs live capture…", "#00c8ff")
        with log_placeholder.container():
            render_log()

        with st.spinner("Running biometric face comparison…"):
            match = compare_faces(client, S.doc_b64, S.live_b64, S.doc_mime)
            S.face_match = match

        verdict_log = "✓ Face match confirmed" if match.get("match") else "✗ Face mismatch detected"
        verdict_col = "#00e5a0" if match.get("match") else "#ff5050"
        add_log(f"{verdict_log} — Confidence: {match.get('confidence','?')} · Score: {match.get('similarity_score','?')}/100", verdict_col)
        add_log("✓ KYC validation complete", "#00e5a0" if match.get("match") else "#ff5050")
        with log_placeholder.container():
            render_log()

        S.validated = True
        progress_placeholder.empty()
        log_placeholder.empty()
        st.rerun()

    # ── Show results ─────────────────────────────────────────────────────────
    match    = S.face_match or {}
    info     = S.extracted  or {}
    is_match = match.get("match", False)
    conf     = match.get("confidence", "Low").lower()
    score    = match.get("similarity_score", 0)
    reason   = match.get("reason", "")
    liveness = match.get("liveness_note", "")

    # ── Verdict ──────────────────────────────────────────────────────────────
    if is_match:
        st.markdown(f"""
        <div class="verdict-verified">
            <div class="verdict-icon">✅</div>
            <div class="verdict-title ok">IDENTITY VERIFIED</div>
            <div class="verdict-reason">{reason}</div>
            <div>
                <span class="conf-badge conf-{conf}">Confidence: {match.get('confidence','?')}</span>
                <span class="conf-badge conf-{conf}">Match score: {score}/100</span>
            </div>
            {f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:8px">{liveness}</div>' if liveness else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-failed">
            <div class="verdict-icon">❌</div>
            <div class="verdict-title fail">VERIFICATION FAILED</div>
            <div class="verdict-reason">{reason}</div>
            <div>
                <span class="conf-badge conf-{conf}">Confidence: {match.get('confidence','?')}</span>
                <span class="conf-badge conf-low">Match score: {score}/100</span>
            </div>
            {f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:8px">{liveness}</div>' if liveness else ''}
        </div>
        """, unsafe_allow_html=True)

    # ── Photo comparison + Extracted fields side by side ──────────────────
    col_left, col_right = st.columns([2, 3], gap="large")

    with col_left:
        # Photo comparison
        st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-title">Biometric Comparison</div>', unsafe_allow_html=True)

        p1, sym, p2 = st.columns([5, 1, 5])
        with p1:
            st.markdown('<div style="text-align:center;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:rgba(255,255,255,0.3);margin-bottom:6px">Document Photo</div>', unsafe_allow_html=True)
            if S.doc_pil:
                st.image(S.doc_pil, use_container_width=True)
        with sym:
            sym_char = "≈" if is_match else "≠"
            sym_color = "#00e5a0" if is_match else "#ff5050"
            st.markdown(f'<div style="text-align:center;font-size:1.8rem;color:{sym_color};padding-top:40px">{sym_char}</div>', unsafe_allow_html=True)
        with p2:
            st.markdown('<div style="text-align:center;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:rgba(255,255,255,0.3);margin-bottom:6px">Live Capture</div>', unsafe_allow_html=True)
            if S.live_pil:
                st.image(S.live_pil, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Audit log
        st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-title" style="margin-bottom:12px">Audit Log</div>', unsafe_allow_html=True)
        render_log()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Extracted fields
        st.markdown(f"""
        <div class="kyc-card">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
                <div class="kyc-card-title">Extracted Document Information</div>
                <span class="autofill-badge">AUTO-FILLED</span>
            </div>
        """, unsafe_allow_html=True)

        fields = [
            ("full_name",      "Full Name"),
            ("document_type",  "Document Type"),
            ("date_of_birth",  "Date of Birth"),
            ("id_number",      "ID / Document Number"),
            ("nationality",    "Nationality"),
            ("gender",         "Gender"),
            ("place_of_birth", "Place of Birth"),
            ("issue_date",     "Issue Date"),
            ("expiry_date",    "Expiry Date"),
            ("address",        "Address"),
            ("mrz",            "MRZ Line"),
        ]

        fields_html = ""
        for key, label in fields:
            val = info.get(key)
            if val:
                fields_html += render_field(label, val, filled=True)

        st.markdown(fields_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # KYC status card
        if is_match:
            st.markdown("""
            <div class="kyc-card" style="background:rgba(0,229,160,0.07);border-color:rgba(0,229,160,0.2)">
                <div style="display:flex;align-items:center;gap:14px">
                    <div style="font-size:2rem">🛡️</div>
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#00e5a0;margin-bottom:3px">KYC Validation Complete</div>
                        <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);line-height:1.5">
                            Identity successfully verified. Document data extracted and confirmed against live biometric capture. This session is compliant.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kyc-card" style="background:rgba(255,80,80,0.07);border-color:rgba(255,80,80,0.2)">
                <div style="display:flex;align-items:center;gap:14px">
                    <div style="font-size:2rem">⛔</div>
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#ff5050;margin-bottom:3px">Verification Unsuccessful</div>
                        <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);line-height:1.5">
                            The live capture did not match the identity document. Please retake the selfie in good lighting, or re-upload a clearer document image.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Reset button ─────────────────────────────────────────────────────────
    st.divider()
    if st.button("↩  Start New Verification", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER LOOP
# ─────────────────────────────────────────────────────────────────────────────
render_header()
render_stepper(S.step)

if S.step == 1:
    render_step1()
elif S.step == 2:
    render_step2()
elif S.step == 3:
    render_step3()
