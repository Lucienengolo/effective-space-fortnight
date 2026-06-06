"""
main.py — E-KYC Identity Verification System
─────────────────────────────────────────────
AI Stack:
  • 🧠 Hugging Face (TrOCR)  → OCR / document text extraction
  • 👤 DeepFace              → local biometric face verification
  • 🎨 Streamlit             → UI

Run:
    streamlit run main.py

Install dependencies:
    pip install streamlit pillow deepface transformers torch pytesseract
"""

import streamlit as st

from css import get_styles
from functions import (
    pil_to_b64,
    add_log,
    extract_document_info,
    compare_faces,
    render_header,
    render_stepper,
    render_field,
    render_log,
    render_verdict,
)
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-KYC Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(get_styles(), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        step=1,
        doc_pil=None,
        doc_b64=None,
        live_pil=None,
        live_b64=None,
        extracted={},
        face_match=None,
        validated=False,
        log=[],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
S = st.session_state


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Document Upload
# ─────────────────────────────────────────────────────────────────────────────
def render_step1():
    col_main, col_info = st.columns([3, 1], gap="large")

    with col_main:
        st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-title">📄 &nbsp; Upload Identity Document</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="kyc-card-sub">Accepted: National ID · Passport · Driver\'s License (JPG, PNG, BMP)</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Drop your document here",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            img       = Image.open(uploaded).convert("RGB")
            S.doc_pil = img
            S.doc_b64 = pil_to_b64(img, "JPEG")
            st.image(img, caption="✓  Document loaded", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

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
                    <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);padding-top:4px">HuggingFace reads your document, DeepFace verifies your identity locally.</div>
                </div>
            </div>
        </div>
        <div class="kyc-card" style="margin-top:0">
            <div class="kyc-card-title" style="font-size:0.85rem">🔐 Privacy Notice</div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.35);margin-top:8px;line-height:1.6">
                All processing is <strong style="color:rgba(255,255,255,0.6)">100% local</strong>.
                No images are sent to any external server. DeepFace and TrOCR run entirely on your machine.
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
        st.markdown(
            '<div class="kyc-card-sub">Allow camera access · Center your face · Take a clear selfie</div>',
            unsafe_allow_html=True,
        )

        photo = st.camera_input("Take a selfie", label_visibility="collapsed")
        if photo:
            img        = Image.open(photo).convert("RGB")
            S.live_pil = img
            S.live_b64 = pil_to_b64(img, "JPEG")

        st.markdown("</div>", unsafe_allow_html=True)

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("← Back to Document", use_container_width=True):
                S.step = 1
                st.rerun()
        with btn2:
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
        st.markdown("</div>", unsafe_allow_html=True)

        if S.live_pil:
            st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
            st.markdown('<div class="kyc-card-title" style="color:#00e5a0">✓ &nbsp; Photo Captured</div>', unsafe_allow_html=True)
            st.image(S.live_pil, caption="Live selfie", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
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
    # ── Run AI pipeline if not yet done ──────────────────────────────────────
    if not S.validated:
        progress_ph = st.empty()
        log_ph      = st.empty()

        with progress_ph.container():
            st.markdown("""
            <div class="kyc-card" style="text-align:center;padding:40px">
                <div style="font-size:2.5rem;margin-bottom:12px">🧠</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:600;color:#00c8ff;margin-bottom:6px">
                    AI Processing Your Identity
                </div>
                <div style="color:rgba(255,255,255,0.4);font-size:0.82rem">
                    HuggingFace TrOCR is reading your document…
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Stage A: OCR ──────────────────────────────────────────────────
        add_log(S, "📄 Running HuggingFace TrOCR on identity document…", "#00c8ff")
        with log_ph.container():
            render_log(S)

        try:
            with st.spinner("Extracting document information…"):
                info      = extract_document_info(S.doc_pil)
                S.extracted = info
            add_log(S, f"✓ Document type: {info.get('document_type', 'ID Document')}", "#00e5a0")
            add_log(S, f"✓ Name extracted: {info.get('full_name', '—')}", "#00e5a0")
        except RuntimeError as e:
            add_log(S, f"⚠️ OCR error: {e}", "#ffbe32")
            S.extracted = {"document_type": "Unknown", "full_name": None}

        with log_ph.container():
            render_log(S)

        # ── Stage B: Face Verification ────────────────────────────────────
        add_log(S, "👤 Running DeepFace biometric comparison (local)…", "#00c8ff")
        with log_ph.container():
            render_log(S)

        try:
            with st.spinner("Running biometric face comparison…"):
                match       = compare_faces(S.doc_pil, S.live_pil)
                S.face_match = match
            verdict_msg   = "✓ Face match confirmed" if match.get("match") else "✗ Face mismatch detected"
            verdict_color = "#00e5a0" if match.get("match") else "#ff5050"
            add_log(
                S,
                f"{verdict_msg} — Confidence: {match.get('confidence','?')} · Score: {match.get('similarity_score','?')}/100",
                verdict_color,
            )
        except RuntimeError as e:
            add_log(S, f"⚠️ Face verification error: {e}", "#ff5050")
            S.face_match = {
                "match": False, "confidence": "Low",
                "similarity_score": 0,
                "reason": str(e), "liveness_note": "",
            }

        add_log(S, "✓ KYC pipeline complete", "#00e5a0" if S.face_match.get("match") else "#ff5050")
        with log_ph.container():
            render_log(S)

        S.validated = True
        progress_ph.empty()
        log_ph.empty()
        st.rerun()

    # ── Render Results ────────────────────────────────────────────────────────
    match    = S.face_match or {}
    info     = S.extracted  or {}
    is_match = match.get("match", False)

    render_verdict(match)

    col_left, col_right = st.columns([2, 3], gap="large")

    with col_left:
        # Biometric comparison photos
        st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-title">Biometric Comparison</div>', unsafe_allow_html=True)

        p1, sym_col, p2 = st.columns([5, 1, 5])
        with p1:
            st.markdown(
                '<div style="text-align:center;font-size:0.68rem;text-transform:uppercase;'
                'letter-spacing:0.1em;color:rgba(255,255,255,0.3);margin-bottom:6px">Document</div>',
                unsafe_allow_html=True,
            )
            if S.doc_pil:
                st.image(S.doc_pil, use_container_width=True)
        with sym_col:
            sym   = "≈" if is_match else "≠"
            color = "#00e5a0" if is_match else "#ff5050"
            st.markdown(
                f'<div style="text-align:center;font-size:1.8rem;color:{color};padding-top:40px">{sym}</div>',
                unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                '<div style="text-align:center;font-size:0.68rem;text-transform:uppercase;'
                'letter-spacing:0.1em;color:rgba(255,255,255,0.3);margin-bottom:6px">Live</div>',
                unsafe_allow_html=True,
            )
            if S.live_pil:
                st.image(S.live_pil, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Audit log
        st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-title" style="margin-bottom:12px">Audit Log</div>', unsafe_allow_html=True)
        render_log(S)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # Extracted document fields
        st.markdown("""
        <div class="kyc-card">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
                <div class="kyc-card-title">Extracted Document Information</div>
                <span class="autofill-badge">AUTO-FILLED</span>
            </div>
        """, unsafe_allow_html=True)

        FIELD_MAP = [
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

        fields_html = "".join(
            render_field(label, info.get(key), filled=bool(info.get(key)))
            for key, label in FIELD_MAP
        )
        st.markdown(fields_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # KYC status summary card
        if is_match:
            st.markdown("""
            <div class="kyc-card" style="background:rgba(0,229,160,0.07);border-color:rgba(0,229,160,0.2)">
                <div style="display:flex;align-items:center;gap:14px">
                    <div style="font-size:2rem">🛡️</div>
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#00e5a0;margin-bottom:3px">
                            KYC Validation Complete
                        </div>
                        <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);line-height:1.5">
                            Identity verified locally using DeepFace. Document fields auto-extracted via TrOCR. No data left this device.
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
                        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#ff5050;margin-bottom:3px">
                            Verification Unsuccessful
                        </div>
                        <div style="font-size:0.78rem;color:rgba(255,255,255,0.4);line-height:1.5">
                            The live capture did not match the document face. Try retaking in better lighting, or re-upload a clearer document.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Reset ─────────────────────────────────────────────────────────────────
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

if   S.step == 1: render_step1()
elif S.step == 2: render_step2()
elif S.step == 3: render_step3()
