"""
functions.py — Core logic and UI render helpers for the E-KYC app.

AI Stack:
  • OCR / Text Extraction  → Hugging Face TrOCR  (primary)
                             pytesseract           (fallback)
  • Face Verification      → InsightFace (ArcFace) via ONNX — pre-built wheels,
                             no TensorFlow, no cmake, works on Python 3.14
  • UI Framework           → Streamlit
"""

# Standard library imports
import time
import re
from io import BytesIO
import base64

# Third-party libraries
import streamlit as st
from PIL import Image
import numpy as np

# OpenCV is optional; only required for face embedding extraction.
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

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
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_OK = True
except ImportError:
    INSIGHTFACE_OK = False


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
    # Normalize the OCR output and keep meaningful lines for parsing.
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
# AI FUNCTION 2 — Face Verification  (InsightFace ArcFace via ONNX)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_face_app():
    """Load InsightFace ArcFace model once and cache it across sessions."""
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app


def _get_embedding(app, pil_img: Image.Image, source: str = "unknown"):
    """
    Extract the ArcFace embedding for the largest detected face.
    Returns a tuple: (embedding_vector, diagnostics_dict) or (None, diagnostics_dict) if no face found.
    
    Diagnostics include:
      - num_faces: count of faces detected
      - face_sizes: list of bounding box dimensions
      - selected_bbox: (x1, y1, x2, y2) of chosen face
      - selected_size: (width, height) of chosen face
      - quality_warnings: list of potential issues
    """
    if not CV2_OK:
        raise RuntimeError("opencv-python-headless is not installed.")
    
    diagnostics = {
        "source": source,
        "num_faces": 0,
        "face_sizes": [],
        "selected_bbox": None,
        "selected_size": None,
        "quality_warnings": [],
    }
    
    # InsightFace expects BGR numpy array (OpenCV convention).
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_height, img_width = bgr.shape[:2]
    faces = app.get(bgr)
    
    diagnostics["num_faces"] = len(faces)
    if not faces:
        diagnostics["quality_warnings"].append("No face detected")
        return None, diagnostics
    
    # Log all detected faces
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox
        w, h = x2 - x1, y2 - y1
        diagnostics["face_sizes"].append((int(w), int(h)))
    
    # Select the biggest detected face to avoid picking small background faces.
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = largest.bbox
    face_w, face_h = int(x2 - x1), int(y2 - y1)
    
    diagnostics["selected_bbox"] = (int(x1), int(y1), int(x2), int(y2))
    diagnostics["selected_size"] = (face_w, face_h)
    
    # Quality checks
    # Warning: face too small (less than 80x80)
    if face_w < 80 or face_h < 80:
        diagnostics["quality_warnings"].append(
            f"Face too small: {face_w}x{face_h}px (ideal ≥80x80). Consider retaking image."
        )
    
    # Warning: face occupies very little of image
    face_area_ratio = (face_w * face_h) / (img_width * img_height)
    if face_area_ratio < 0.05:
        diagnostics["quality_warnings"].append(
            f"Face occupies only {face_area_ratio*100:.1f}% of image. Zoom in or move closer."
        )
    
    return largest.embedding, diagnostics


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [−1, 1]; 1 = identical direction."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _normalize_similarity_to_score(cos_sim: float) -> int:
    """
    Convert raw cosine similarity [-1, 1] to a user-friendly 0-100 score.
    
    Mapping:
      cos_sim = -1.0 → score = 0    (opposite directions, very different)
      cos_sim =  0.0 → score = 50   (orthogonal, no relation)
      cos_sim =  1.0 → score = 100  (identical direction, perfect match)
    """
    return int(max(0, min(100, round((cos_sim + 1) / 2 * 100))))


def compare_faces(
    doc_img: Image.Image, live_img: Image.Image,
    threshold: float = 0.50,
    show_diagnostics: bool = True
) -> dict:
    """
    Use InsightFace (ArcFace model, ONNX runtime) to compare the face in the
    identity document with the live selfie.

    Args:
      threshold: cosine similarity threshold [0.0, 1.0]. Tuning guide:
        - 0.40: lenient (may accept different people, ~60/100 score)
        - 0.45: balanced (commercial KYC standard, ~65/100 score)
        - 0.50: strict (recommended for high security, ~70/100 score)
        - 0.55+: very strict (~75+/100 score)
      show_diagnostics: include face detection details in output

    Returns:
      dict with keys: match, confidence, similarity_score, reason, liveness_note,
      doc_diagnostics, live_diagnostics (if show_diagnostics=True)

    Pre-built ONNX wheels: no TensorFlow, no cmake, no dlib compilation.
    Works on Python 3.14 and Streamlit Cloud out of the box.
    """
    if not INSIGHTFACE_OK:
        raise RuntimeError(
            "insightface is not installed. Run:\n"
            "  pip install insightface onnxruntime"
        )

    face_app = _load_face_app()

    doc_emb, doc_diag  = _get_embedding(face_app, doc_img, source="document")
    live_emb, live_diag = _get_embedding(face_app, live_img, source="live_selfie")

    result = {
        "match": False, "confidence": "Low", "similarity_score": 0,
        "raw_cosine": None,
        "threshold_used": threshold,
        "reason": "",
        "liveness_note": "InsightFace ArcFace (buffalo_sc · ONNX CPU)",
    }
    
    if show_diagnostics:
        result["doc_diagnostics"] = doc_diag
        result["live_diagnostics"] = live_diag

    if doc_emb is None:
        result["reason"] = (
            "No face detected in the identity document image.\n" +
            "\n".join(f"  • {w}" for w in doc_diag.get("quality_warnings", []))
        )
        return result
    
    if live_emb is None:
        result["reason"] = (
            "No face detected in the live selfie.\n" +
            "\n".join(f"  • {w}" for w in live_diag.get("quality_warnings", []))
        )
        return result

    # Cosine similarity range is [-1, 1]. A higher value means closer face embeddings.
    cos_sim = _cosine_similarity(doc_emb, live_emb)
    result["raw_cosine"] = round(cos_sim, 4)
    verified = cos_sim >= threshold

    # Normalize cosine similarity (−1…1) → score (0…100) for UI display.
    similarity = _normalize_similarity_to_score(cos_sim)

    # Confidence levels based on normalized score
    if similarity >= 75:   confidence = "High"
    elif similarity >= 60: confidence = "Medium"
    else:                  confidence = "Low"

    # Collect quality warnings from both images
    all_warnings = doc_diag.get("quality_warnings", []) + live_diag.get("quality_warnings", [])
    warnings_text = "".join(f"  ⚠️  {w}\n" for w in all_warnings) if all_warnings else ""

    reason = (
        f"✓ Facial features match across both images.\n"
        f"  Similarity: {similarity}/100 (raw cosine: {cos_sim:.4f})"
        if verified else
        f"✗ Facial differences detected. Likely a different person.\n"
        f"  Similarity: {similarity}/100 (raw cosine: {cos_sim:.4f})"
    )
    
    if warnings_text:
        reason = f"{reason}\n\n{warnings_text}Recommendations:\n  1. Ensure good lighting on both images\n  2. Face should occupy 10-30% of image\n  3. Minimize angles and expressions\n  4. Use clear, high-contrast document photo"

    result.update({
        "match": verified,
        "confidence": confidence,
        "similarity_score": similarity,
        "reason": reason,
        "liveness_note": f"InsightFace ArcFace · cosine: {cos_sim:.4f} · threshold: {threshold}",
    })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="kyc-header">
        <div class="kyc-header-title">⬡ &nbsp; E-KYC IDENTITY VERIFICATION SYSTEM</div>
        <div class="kyc-header-badge">🤖 &nbsp; HuggingFace · InsightFace</div>
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
    raw_cos    = match.get("raw_cosine", None)
    reason     = match.get("reason", "").replace("\n", "<br>")
    liveness   = match.get("liveness_note", "")
    conf_label = match.get("confidence", "?")

    # Format raw cosine display
    raw_cos_html = f"<br><span style=\"font-size:0.85rem;color:rgba(255,255,255,0.4)\">Raw cosine: {raw_cos:.4f}</span>" if raw_cos is not None else ""
    
    liveness_html = (
        f'<div style=\"font-size:0.75rem;color:rgba(255,255,255,0.3);margin-top:8px\">{liveness}</div>'
        if liveness else ""
    )

    if is_match:
        st.markdown(f"""
        <div class="verdict-verified">
            <div class="verdict-icon">✅</div>
            <div class="verdict-title ok">IDENTITY VERIFIED</div>
            <div class="verdict-reason" style="white-space: pre-wrap; word-wrap: break-word;">{reason}</div>
            <div>
                <span class="conf-badge conf-{conf}">Confidence: {conf_label}</span>
                <span class="conf-badge conf-{conf}">Match score: {score}/100</span>
                {raw_cos_html}
            </div>
            {liveness_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-failed">
            <div class="verdict-icon">❌</div>
            <div class="verdict-title fail">VERIFICATION FAILED</div>
            <div class="verdict-reason" style="white-space: pre-wrap; word-wrap: break-word;">{reason}</div>
            <div>
                <span class="conf-badge conf-{conf}">Confidence: {conf_label}</span>
                <span class="conf-badge conf-low">Match score: {score}/100</span>
                {raw_cos_html}
            </div>
            {liveness_html}
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PRESENTATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
# This file is structured into three main sections:
#
# 1) Utilities
# #   - Image conversion helpers for PIL/base64 interchange.
# #   - A session audit logger for timestamped KYC events.
#
# 2) AI pipeline functionality
# #   - OCR: Use HuggingFace TrOCR first, fall back to pytesseract if needed.
# #   - Document parsing: extract fields such as name, document type, dates, and MRZ.
# #   - Face verification: load InsightFace ArcFace, compute embeddings, compare faces.
#
# 3) UI render helpers
# #   - Build Streamlit components for the page header, stepper, field cards, log timeline, and verdict banners.
#
# Global flow summary:
#   Upload a document image -> extract text fields -> capture a selfie -> compare face embeddings -> display verification results.
#
# The entire module supports a privacy-first local KYC experience by keeping all AI processing on-device.


