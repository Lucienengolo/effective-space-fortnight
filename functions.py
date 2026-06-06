"""
functions.py — Core logic and UI render helpers for the E-KYC app.

AI Stack:
  • OCR / Text Extraction  → Hugging Face  (microsoft/trocr-large-printed  or
                                             tesseract fallback via pytesseract)
  • Face Verification      → DeepFace      (local, no API key required)
  • UI Framework           → Streamlit
"""

import time
import re
import tempfile
import os
from io import BytesIO
import base64

import streamlit as st
from PIL import Image

# ── Optional heavy imports (lazy-loaded to avoid import errors if not installed) ──
try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_OK = True
except ImportError:
    TROCR_OK = False

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Convert a PIL image to a base64-encoded string."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    """Convert a base64 string back to a PIL image."""
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data)).convert("RGB")


def add_log(session, msg: str, color: str = "#00c8ff"):
    """Append a timestamped entry to the session audit log."""
    session.log.append({"msg": msg, "color": color, "ts": time.strftime("%H:%M:%S")})


def save_pil_to_tmp(img: Image.Image, suffix: str = ".jpg") -> str:
    """Save a PIL image to a temp file and return the path (for DeepFace)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    img.save(tmp.name, format="JPEG")
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# AI FUNCTION 1 — OCR / Document Extraction  (Hugging Face TrOCR)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_trocr():
    """Load TrOCR model once and cache it across sessions."""
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
    return processor, model


def _ocr_with_trocr(img: Image.Image) -> str:
    """Run Hugging Face TrOCR on a PIL image and return extracted text."""
    processor, model = _load_trocr()
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixel_values)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def _ocr_with_tesseract(img: Image.Image) -> str:
    """Fallback OCR using pytesseract (must be installed + Tesseract binary)."""
    return pytesseract.image_to_string(img)


def _ocr_raw_text(img: Image.Image) -> str:
    """
    Try TrOCR first (best quality), fall back to Tesseract, then give up
    gracefully with an error message.
    """
    if TROCR_OK:
        return _ocr_with_trocr(img)
    if TESSERACT_OK:
        return _ocr_with_tesseract(img)
    raise RuntimeError(
        "No OCR backend found. Install one:\n"
        "  pip install transformers torch   (for TrOCR)\n"
        "  pip install pytesseract          (for Tesseract)"
    )


def _parse_fields_from_text(raw: str) -> dict:
    """
    Heuristic field parser: scan raw OCR text for common ID document patterns.
    Returns a dict matching the expected KYC schema.
    """
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    text  = raw.upper()

    def find_after(keyword: str, raw_block: str) -> str | None:
        """Return the token(s) that follow 'keyword:' on the same line."""
        pattern = rf"{re.escape(keyword)}[:\s]+([^\n]+)"
        m = re.search(pattern, raw_block, re.IGNORECASE)
        return m.group(1).strip() if m else None

    # Date-like pattern: DD/MM/YYYY or DD-MM-YYYY or YYYY-MM-DD
    date_rx = re.compile(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{2}[/\-\.]\d{2})\b")
    dates = date_rx.findall(raw)

    # ID number: 8–20 alphanumeric chars (covers CNI, passport, driving licence)
    id_rx = re.compile(r"\b([A-Z0-9]{8,20})\b")
    id_candidates = id_rx.findall(text)

    # MRZ line: lines with <<
    mrz_lines = [l for l in lines if "<<" in l]

    # Gender
    gender = None
    if re.search(r"\bM\b|\bMALE\b|\bMASCULIN\b",    text): gender = "Male"
    if re.search(r"\bF\b|\bFEMALE\b|\bFEMININ\b",   text): gender = "Female"

    # Nationality: 3-letter ISO code near NATIONALITY keyword
    nat_m = re.search(r"NATIONAL[^\n]{0,10}([A-Z]{3})", text)
    nationality = nat_m.group(1) if nat_m else None

    return {
        "full_name":      find_after("NOM", raw) or find_after("NAME", raw) or find_after("SURNAME", raw),
        "date_of_birth":  dates[0] if dates else None,
        "id_number":      id_candidates[0] if id_candidates else None,
        "document_type":  _guess_doc_type(text),
        "nationality":    nationality,
        "gender":         gender,
        "issue_date":     dates[1] if len(dates) > 1 else None,
        "expiry_date":    dates[2] if len(dates) > 2 else None,
        "address":        find_after("ADDRESS", raw) or find_after("ADRESSE", raw),
        "place_of_birth": find_after("PLACE OF BIRTH", raw) or find_after("LIEU DE NAISSANCE", raw),
        "mrz":            " | ".join(mrz_lines) if mrz_lines else None,
    }


def _guess_doc_type(text_upper: str) -> str:
    if "PASSPORT"  in text_upper: return "Passport"
    if "PERMIS"    in text_upper: return "Driver's License"
    if "DRIVING"   in text_upper: return "Driver's License"
    if "NATIONAL"  in text_upper: return "National ID Card"
    if "CARTE"     in text_upper: return "ID Card"
    return "Identity Document"


def extract_document_info(img: Image.Image) -> dict:
    """
    Public API: run OCR on the document image and return a structured dict.
    Raises RuntimeError if no OCR backend is available.
    """
    raw    = _ocr_raw_text(img)
    fields = _parse_fields_from_text(raw)
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# AI FUNCTION 2 — Face Verification  (DeepFace)
# ─────────────────────────────────────────────────────────────────────────────

def compare_faces(doc_img: Image.Image, live_img: Image.Image) -> dict:
    """
    Use DeepFace to compare the face in the identity document with the live
    selfie.  Returns a verdict dict compatible with the original Gemini schema
    so the rest of the app needs no changes.

    DeepFace.verify() returns:
      {
        "verified": bool,
        "distance": float,          # lower = more similar
        "threshold": float,
        "model": str,
        "detector_backend": str,
        "similarity_metric": str,
      }
    """
    if not DEEPFACE_OK:
        raise RuntimeError(
            "DeepFace is not installed. Run:\n"
            "  pip install deepface"
        )

    doc_path  = save_pil_to_tmp(doc_img)
    live_path = save_pil_to_tmp(live_img)

    try:
        result = DeepFace.verify(
            img1_path        = doc_path,
            img2_path        = live_path,
            model_name       = "VGG-Face",    # fast & reliable for KYC
            detector_backend = "retinaface",  # good at angled / small faces
            enforce_detection= False,         # graceful if no face detected
        )

        distance  = result.get("distance",  1.0)
        threshold = result.get("threshold", 0.4)
        verified  = result.get("verified",  False)

        # Map raw distance to a 0–100 similarity score
        similarity = max(0, round((1 - distance / max(threshold, 0.001)) * 100))
        similarity = min(similarity, 100)

        # Confidence tier
        if similarity >= 75:   confidence = "High"
        elif similarity >= 50: confidence = "Medium"
        else:                  confidence = "Low"

        reason = (
            f"Facial geometry is consistent across both images (score {similarity}/100)."
            if verified else
            f"Significant facial differences detected (score {similarity}/100)."
        )

        return {
            "match":            verified,
            "confidence":       confidence,
            "similarity_score": similarity,
            "reason":           reason,
            "liveness_note":    f"DeepFace model: VGG-Face · distance: {distance:.3f} · threshold: {threshold:.3f}",
        }

    finally:
        # Always clean up temp files
        for p in (doc_path, live_path):
            try:
                os.remove(p)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="kyc-header">
        <div class="kyc-header-title">⬡ &nbsp; E-KYC IDENTITY VERIFICATION SYSTEM</div>
        <div class="kyc-header-badge">🤖 &nbsp; HuggingFace · DeepFace</div>
    </div>
    """, unsafe_allow_html=True)


def render_stepper(current: int):
    """Render the 3-step progress indicator."""

    def circle_cls(n: int) -> str:
        if n < current:  return "done"
        if n == current: return "active"
        return ""

    def line_cls(n: int) -> str:
        return "done" if n < current else ""

    steps = [(1, "Document"), (2, "Face Scan"), (3, "Validate")]
    parts = ['<div class="step-bar">']

    for i, (n, label) in enumerate(steps):
        c    = circle_cls(n)
        icon = "✓" if c == "done" else str(n)
        parts.append(f"""
        <div class="step-item">
            <div class="step-circle {c}">{icon}</div>
            <div class="step-label {c}">{label}</div>
        </div>""")
        if i < len(steps) - 1:
            parts.append(f'<div class="step-line {line_cls(n)}"></div>')

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_field(label: str, value, filled: bool = False) -> str:
    """Return HTML for a single extracted field block."""
    val_str      = str(value) if value else ""
    filled_class = "filled" if (value and filled) else ""
    empty_html   = '<span style="color:rgba(255,255,255,0.15);font-style:italic">Not found</span>'

    return f"""
    <div class="field-block {filled_class}">
        <div class="field-label">{label}</div>
        <div class="field-value {filled_class}">{val_str if value else empty_html}</div>
    </div>"""


def render_log(session):
    """Render the audit log timeline."""
    if not session.log:
        return
    items = []
    for item in session.log:
        items.append(f"""
        <div class="log-item">
            <div class="log-dot" style="background:{item['color']}"></div>
            <div>
                <span style="color:rgba(255,255,255,0.25);margin-right:8px">{item['ts']}</span>
                {item['msg']}
            </div>
        </div>""")
    st.markdown("".join(items), unsafe_allow_html=True)


def render_verdict(match: dict):
    """Render the big verdict banner (verified / failed)."""
    is_match   = match.get("match", False)
    conf       = match.get("confidence", "Low").lower()
    score      = match.get("similarity_score", 0)
    reason     = match.get("reason", "")
    liveness   = match.get("liveness_note", "")
    conf_label = match.get("confidence", "?")

    liveness_html = (
        f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:8px">{liveness}</div>'
        if liveness else ""
    )

    if is_match:
        st.markdown(f"""
        <div class="verdict-verified">
            <div class="verdict-icon">✅</div>
            <div class="verdict-title ok">IDENTITY VERIFIED</div>
            <div class="verdict-reason">{reason}</div>
            <div>
                <span class="conf-badge conf-{conf}">Confidence: {conf_label}</span>
                <span class="conf-badge conf-{conf}">Match score: {score}/100</span>
            </div>
            {liveness_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-failed">
            <div class="verdict-icon">❌</div>
            <div class="verdict-title fail">VERIFICATION FAILED</div>
            <div class="verdict-reason">{reason}</div>
            <div>
                <span class="conf-badge conf-{conf}">Confidence: {conf_label}</span>
                <span class="conf-badge conf-low">Match score: {score}/100</span>
            </div>
            {liveness_html}
        </div>
        """, unsafe_allow_html=True)
