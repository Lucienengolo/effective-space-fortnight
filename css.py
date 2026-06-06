"""
css.py — All global CSS styles for the E-KYC Verification app.
Inject with: st.markdown(get_styles(), unsafe_allow_html=True)
"""
# Return the full page stylesheet as a single HTML string for Streamlit

def get_styles() -> str:
    return """
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

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

/* ── Header bar ── */
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

/* ── Camera input ── */
div[data-testid="stCameraInput"] > div {
    background: #0a0f1e !important;
    border: 2px solid rgba(0,200,255,0.2) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Extracted field blocks ── */
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

/* ── Verdict banners ── */
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
.verdict-reason { font-size: 0.85rem; color: rgba(255,255,255,0.5); }

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

/* ── Audit log ── */
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

/* ── Auto-fill badge ── */
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

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00c8ff !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
"""
