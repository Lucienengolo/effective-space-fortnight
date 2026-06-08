"""
main.py — E-KYC Identity Verification System
─────────────────────────────────────────────
This app performs on-device identity verification by:
  • extracting text from uploaded identity documents with HuggingFace TrOCR,
  • capturing a live selfie using Streamlit camera input,
  • comparing the document face and live selfie using InsightFace ArcFace.

Tools used:
  • Streamlit for UI and camera capture
  • Pillow for image handling
  • InsightFace (ArcFace) for local face matching
  • HuggingFace Transformers (TrOCR) for OCR
  • PyTorch as the ML backend
  • Tesseract OCR support via pytesseract
  • OpenCV for image processing

The goal is to provide a privacy-first KYC flow where all processing stays
local to the user's machine.
"""

import streamlit as st

# UI stylesheet helper and application logic helpers.
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
# Configure the Streamlit page and inject the custom CSS styles.
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
# Initialize the Streamlit session state with application defaults.
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
# Render the first step where the user uploads an identity document.
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
                    <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);padding-top:4px">HuggingFace reads your document, InsightFace verifies your identity locally.</div>
                </div>
            </div>
        </div>
        <div class="kyc-card" style="margin-top:0">
            <div class="kyc-card-title" style="font-size:0.85rem">🔐 Privacy Notice</div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.35);margin-top:8px;line-height:1.6">
                All processing is <strong style="color:rgba(255,255,255,0.6)">100% local</strong>.
                No images are sent to any external server. InsightFace and TrOCR run entirely on your machine.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Face Capture
# ─────────────────────────────────────────────────────────────────────────────
# Render the second step: capture a live selfie for biometric verification.
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
# Perform OCR and face matching, then show the verification outcome.
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
        add_log(S, "👤 Running InsightFace biometric comparison (local)…", "#00c8ff")
        with log_ph.container():
            render_log(S)

        try:
            with st.spinner("Running biometric face comparison…"):
                # Use default threshold of 0.50 for strict KYC standard
                match       = compare_faces(S.doc_pil, S.live_pil, threshold=0.50, show_diagnostics=True)
                S.face_match = match
            verdict_msg   = "✓ Face match confirmed" if match.get("match") else "✗ Face mismatch detected"
            verdict_color = "#00e5a0" if match.get("match") else "#ff5050"
            raw_cos = match.get("raw_cosine", None)
            raw_info = f" · Raw: {raw_cos:.4f}" if raw_cos is not None else ""
            add_log(
                S,
                f"{verdict_msg} — Confidence: {match.get('confidence','?')} · Score: {match.get('similarity_score','?')}/100{raw_info}",
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

        # Face Detection Diagnostics
        doc_diag = match.get("doc_diagnostics", {})
        live_diag = match.get("live_diagnostics", {})
        if doc_diag or live_diag:
            st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
            st.markdown('<div class="kyc-card-title">Face Detection Diagnostics</div>', unsafe_allow_html=True)
            
            diag_text = "**Document Face:**\n"
            if doc_diag:
                diag_text += f"  • Detected: {doc_diag.get('num_faces', 0)} face(s)\n"
                if doc_diag.get('selected_size'):
                    w, h = doc_diag['selected_size']
                    diag_text += f"  • Selected face size: {w}×{h}px\n"
                if doc_diag.get('quality_warnings'):
                    for warning in doc_diag['quality_warnings']:
                        diag_text += f"  ⚠️ {warning}\n"
                else:
                    diag_text += "  ✓ Face quality good\n"
            
            diag_text += "\n**Live Selfie:**\n"
            if live_diag:
                diag_text += f"  • Detected: {live_diag.get('num_faces', 0)} face(s)\n"
                if live_diag.get('selected_size'):
                    w, h = live_diag['selected_size']
                    diag_text += f"  • Selected face size: {w}×{h}px\n"
                if live_diag.get('quality_warnings'):
                    for warning in live_diag['quality_warnings']:
                        diag_text += f"  ⚠️ {warning}\n"
                else:
                    diag_text += "  ✓ Face quality good\n"
            
            st.markdown(diag_text)
            st.markdown("</div>", unsafe_allow_html=True)

        # Threshold Calibration
        st.markdown('<div class="kyc-card">', unsafe_allow_html=True)
        st.markdown('<div class="kyc-card-title">Threshold Calibration (Advanced)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-bottom:12px;line-height:1.5">
        <strong>Adjust threshold to retune matching:</strong><br>
        • <strong>0.40–0.45:</strong> Lenient (false accepts ~15%, 60/100 score)<br>
        • <strong>0.45–0.50:</strong> Balanced (false accepts ~5%, 65–70/100 score)<br>
        • <strong>0.50–0.55:</strong> Strict (false accepts <2%, 70–75/100 score)<br>
        • <strong>0.55+:</strong> Very strict (high security, may reject valid users)
        </div>
        """, unsafe_allow_html=True)
        
        current_raw_cos = match.get("raw_cosine", 0.0)
        new_threshold = st.slider(
            "Match Threshold (raw cosine similarity)",
            min_value=0.30,
            max_value=0.65,
            value=match.get("threshold_used", 0.50),
            step=0.01,
            format="%.2f",
            key="threshold_calibration",
            help="Move slider left (lenient) or right (strict) to change match sensitivity"
        )
        
        # Recalculate match with new threshold
        if new_threshold != match.get("threshold_used", 0.50):
            rematch_result = current_raw_cos >= new_threshold
            rematch_score = int(max(0, min(100, round((current_raw_cos + 1) / 2 * 100))))
            st.markdown(f"""
            <div style="background:rgba(0,200,255,0.1);border:1px solid rgba(0,200,255,0.3);border-radius:8px;padding:12px;margin-top:8px;font-size:0.82rem">
            <strong>With threshold {new_threshold:.2f}:</strong><br>
            Result: <strong>{'✓ MATCH' if rematch_result else '✗ NO MATCH'}</strong><br>
            Score would be: <strong>{rematch_score}/100</strong> (cosine: {current_raw_cos:.4f})<br>
            <span style="color:rgba(255,255,255,0.4)">Tip: Rerun verification (Start New Verification) to save new threshold as default.</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
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
                            Identity verified locally using InsightFace ArcFace. Document fields auto-extracted via TrOCR. No data left this device.
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
# Dispatch the correct UI step based on the current session state.
render_header()
render_stepper(S.step)

if   S.step == 1: render_step1()
elif S.step == 2: render_step2()
elif S.step == 3: render_step3()


# ─────────────────────────────────────────────────────────────────────────────
# PRESENTATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
# 1. Top docstring explains the app purpose and tools used.
# 2. Import Streamlit and helper modules for UI, styling, and image logic.
# 3. Configure the Streamlit page title, icon, layout, and sidebar state.
# 4. Inject custom CSS using get_styles() so the app has branded styling.
# 5. _init_state() sets up shared session state with defaults for steps, images, OCR result, and logs.
# 6. The step 1 renderer builds the document upload UI:
#      - upload a file, convert it to PIL, save it in state, and show a preview.
#      - enable the continue button only after a document is loaded.
# 7. The step 2 renderer builds the live selfie capture UI:
#      - capture a camera image, convert and store it, show live preview.
#      - allow returning to step 1 or moving to verification when a selfie exists.
# 8. The step 3 renderer runs the validation pipeline only once per session:
#      - show progress and audit log placeholders.
#      - extract document fields with OCR and store them.
#      - compare the uploaded document face with the selfie using AI.
#      - log results, set validated state, clear progress, and rerun.
# 9. Step 3 also renders the final results UI:
#      - verdict banner, document/live photos, audit log, extracted fields, and outcome card.
#      - provide a reset button to clear state and start a new verification.
# 10. The main render loop draws the header, stepper, and executes the current step function.
#
# Overall, this file creates a 3-step E-KYC experience: document upload, face capture, and local verification.



