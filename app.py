mport base64
import hashlib
import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

st.set_page_config(
    page_title="AgriVision Maharashtra",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "agrivision_model.pth"
NUM_CLASSES = 3
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
LOW_CONFIDENCE = 60.0

DISEASES = {
    0: {
        "crop": "Cotton",
        "name": "Cotton - Healthy",
        "marathi": "कापूस - निरोगी",
        "status": "Healthy",
        "tone": "healthy",
        "chemical": "None",
        "organic": "Maintain soil moisture and compost.",
    },
    1: {
        "crop": "Cotton",
        "name": "Cotton - Bacterial Blight",
        "marathi": "कापूस - जीवाणूजन्य करपा",
        "status": "Disease detected",
        "tone": "alert",
        "chemical": "Copper Oxychloride 50% WP @ 25g/10L.",
        "organic": "Neem seed kernel extract (5%) foliar spray.",
    },
    2: {
        "crop": "Sugarcane",
        "name": "Sugarcane - Red Rot",
        "marathi": "ऊस - तांबेरा / लाल कूज",
        "status": "Disease detected",
        "tone": "alert",
        "chemical": "Uproot and burn infected clumps.",
        "organic": "Soil application of Trichoderma viride.",
    },
}

MODEL_ERRORS = {
    "runtime": "PyTorch and torchvision are not installed in this environment. Diagnostics are unavailable.",
    "missing": "Model weights not found. Place agrivision_model.pth alongside app.py to enable diagnostics.",
    "incompatible": "The checkpoint does not match the MobileNetV2 three-class architecture. Verify the export and redeploy.",
    "corrupt": "The model file could not be read. Re-export agrivision_model.pth and redeploy.",
}

SCAN_PHASES = [
    ("Analyzing cellular structure...", 0.15),
    ("Isolating lesion boundaries...", 0.35),
    ("Normalizing color channels...", 0.55),
    ("Running MobileNetV2 inference...", 0.80),
    ("Scoring class probabilities...", 0.95),
]

PIPELINE = [
    ("01", "Capture", "Leaf photograph from a phone or field camera. JPG, JPEG or PNG up to 10 MB."),
    ("02", "Validate", "Pillow decodes the file and forces RGB, flattening alpha and palette modes."),
    ("03", "Preprocess", "Resize to 224 × 224, convert to tensor, normalize with ImageNet mean and std."),
    ("04", "Infer", "MobileNetV2 backbone with a 3-class head, evaluated on CPU in inference mode."),
    ("05", "Score", "Softmax over logits. Top class and full probability distribution are reported."),
    ("06", "Advise", "Class-mapped chemical and organic remedies from the agronomy lookup table."),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

:root {
  --av-green-900: #0B3B2C;
  --av-green-700: #1F6B4F;
  --av-green-50: #EEF7F1;
  --av-orange-50: #FFF4E8;
  --av-ink: #1C2421;
  --av-muted: #5F6B66;
  --av-line: #E4E8E3;
  --av-shadow: 0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06);
}

html, body, [class*="css"], .stApp {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  color: var(--av-ink);
}
.stApp { background: #F7F8F5; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
.devanagari { font-family: 'Noto Sans Devanagari', 'Inter', sans-serif; }

[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--av-line); }

.av-hero {
  background: linear-gradient(120deg, #0B3B2C 0%, #16523D 55%, #1F6B4F 100%);
  border-radius: 12px; padding: 28px 32px; color: #FFFFFF;
  box-shadow: 0 10px 30px rgba(11,59,44,.18); margin-bottom: 1.5rem;
  display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;
}
.av-hero h1 { color: #FFFFFF; font-size: 1.85rem; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.01em; }
.av-hero p { margin: 0; color: rgba(255,255,255,.78); font-size: .95rem; }
.av-chip {
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); color: #FFFFFF;
  border-radius: 999px; padding: 6px 12px; font-size: .78rem; font-weight: 500; white-space: nowrap;
}

.av-card {
  background: #FFFFFF; border-radius: 12px; padding: 20px 22px; border: 1px solid var(--av-line);
  box-shadow: var(--av-shadow); margin-bottom: 16px;
}
.av-card h4 {
  margin: 0 0 10px 0; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em;
  color: var(--av-muted); font-weight: 600;
}
.av-section-title { font-size: 1.05rem; font-weight: 600; margin: 0 0 6px 0; }
.av-muted { color: var(--av-muted); font-size: .9rem; line-height: 1.55; margin: 0; }
.av-list { margin: 8px 0 0 0; padding-left: 18px; color: var(--av-muted); font-size: .88rem; line-height: 1.7; }

.av-empty {
  border: 1.5px dashed #B9C7BE; border-radius: 12px; background: #FFFFFF; padding: 64px 28px;
  text-align: center; box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.av-empty-title { font-weight: 600; font-size: 1.1rem; margin-top: 14px; }
.av-empty-text { color: var(--av-muted); font-size: .9rem; margin: 6px auto 0 auto; max-width: 380px; line-height: 1.55; }

.av-preview { padding: 10px; }
.av-preview img, .scan-wrap img { display: block; width: 100%; border-radius: 12px; }
.scan-wrap {
  position: relative; overflow: hidden; border-radius: 12px; box-shadow: var(--av-shadow);
  background: #FFFFFF; padding: 10px;
}
.scan-overlay { position: absolute; inset: 10px; border-radius: 12px; background: linear-gradient(180deg, rgba(11,59,44,0) 0%, rgba(11,59,44,.28) 100%); }
.scan-line {
  position: absolute; left: 10px; right: 10px; height: 3px; border-radius: 3px;
  background: linear-gradient(90deg, transparent, #7CFFB2, transparent);
  box-shadow: 0 0 14px #7CFFB2; animation: av-scan 1.4s ease-in-out infinite;
}
@keyframes av-scan {
  0% { top: 10px; }
  50% { top: calc(100% - 13px); }
  100% { top: 10px; }
}

.av-diag-crop { font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; color: var(--av-muted); font-weight: 600; }
.av-diag-name { font-size: 1.4rem; font-weight: 700; margin: 4px 0 2px 0; letter-spacing: -0.01em; }
.av-diag-mr { font-size: 1.05rem; color: #3A4541; margin-bottom: 12px; }
.av-badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; }
.badge-healthy { background: #E3F4EA; color: #1E6B3F; }
.badge-alert { background: #FDECEA; color: #A32020; }

.av-conf-value { font-size: 2.6rem; font-weight: 700; color: var(--av-green-900); line-height: 1; }
.av-conf-value span { font-size: 1.1rem; color: var(--av-muted); margin-left: 2px; }
.av-conf-note { font-size: .85rem; color: var(--av-muted); margin: 8px 0 14px 0; line-height: 1.5; }
.bar-track { height: 8px; background: #EDF0EC; border-radius: 999px; overflow: hidden; }
.bar-track.lg { height: 10px; margin-bottom: 4px; }
.bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2E7D4F, #55A877); }
.bar-fill.muted { background: #B8C4BD; }
.av-prob { margin-top: 12px; }
.av-prob-row { display: flex; justify-content: space-between; font-size: .82rem; color: #3A4541; margin-bottom: 6px; }

.av-remedy { border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; border: 1px solid; box-shadow: var(--av-shadow); }
.av-remedy.organic { background: var(--av-green-50); border-color: #CFE7D7; border-left: 4px solid #2E7D4F; }
.av-remedy.chemical { background: var(--av-orange-50); border-color: #F7D9B5; border-left: 4px solid #D9730D; }
.av-remedy-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.av-remedy.organic .av-remedy-label { color: #2E7D4F; }
.av-remedy.chemical .av-remedy-label { color: #B35A05; }
.av-remedy-body { font-size: .98rem; font-weight: 500; margin-top: 6px; color: var(--av-ink); line-height: 1.45; }

.av-pipe { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 16px; }
.av-step { background: #FFFFFF; border: 1px solid var(--av-line); border-radius: 12px; padding: 16px; box-shadow: var(--av-shadow); }
.av-step .idx { font-size: .72rem; color: #2E7D4F; font-weight: 700; letter-spacing: .08em; }
.av-step .t { font-weight: 600; margin: 4px 0 6px 0; }
.av-step .d { font-size: .84rem; color: var(--av-muted); line-height: 1.45; }

[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid var(--av-line); border-radius: 12px;
  padding: 16px 18px; box-shadow: var(--av-shadow);
}
[data-testid="stMetricValue"] { color: var(--av-green-900); }

[data-testid="stFileUploader"] section {
  border-radius: 12px; border: 1.5px dashed #B9C7BE; background: #FBFCFA;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid var(--av-line); }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--av-green-700) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--av-green-700) !important; }

.av-side-brand { font-size: 1.25rem; font-weight: 700; color: var(--av-green-900); }
.av-side-sub { font-size: .8rem; color: var(--av-muted); margin-bottom: 8px; }
.av-side-h {
  font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: #2E7D4F;
  font-weight: 700; margin: 20px 0 6px 0;
}
.av-side-item {
  display: flex; justify-content: space-between; gap: 8px; padding: 9px 0;
  border-bottom: 1px solid var(--av-line); font-size: .84rem;
}
.av-side-item span { color: var(--av-muted); }
.av-side-item strong { color: var(--av-ink); font-weight: 600; text-align: right; }
</style>
"""

HERO = """
<div class="av-hero">
  <div>
    <h1>AgriVision Maharashtra</h1>
    <p>Leaf-level disease classification for cotton and sugarcane, with class-mapped treatment guidance.</p>
  </div>
  <div class="av-chip">MobileNetV2 · On-device CPU inference</div>
</div>
"""

EMPTY_STATE = """
<div class="av-empty">
  <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#1F6B4F" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
    <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
  </svg>
  <div class="av-empty-title">No leaf sample loaded</div>
  <div class="av-empty-text">Upload a close-up of a single cotton or sugarcane leaf. Diagnosis, confidence score and treatment guidance will appear here.</div>
</div>
"""

GUIDE_CARD = """
<div class="av-card">
  <h4>Capture guidelines</h4>
  <ul class="av-list">
    <li>One leaf per image, filled frame</li>
    <li>Natural daylight, no flash or heavy shadow</li>
    <li>Avoid motion blur and compression artifacts</li>
  </ul>
</div>
"""


def render(markup):
    st.markdown(
        "\n".join(line.strip() for line in markup.splitlines() if line.strip()),
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading MobileNetV2 weights...")
def load_model():
    if not TORCH_AVAILABLE:
        return None, "runtime"
    if not MODEL_PATH.is_file():
        return None, "missing"
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model, None
    except FileNotFoundError:
        return None, "missing"
    except (RuntimeError, KeyError, AttributeError, TypeError):
        return None, "incompatible"
    except Exception:
        return None, "corrupt"


def read_image(data):
    if len(data) > MAX_UPLOAD_BYTES:
        return None, f"File exceeds the {MAX_UPLOAD_MB} MB limit. Compress the image and upload again."
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.convert("RGB"), None
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        return None, "The file could not be read as a JPG or PNG image. Upload an unmodified photograph."


def to_data_uri(image, max_side=720):
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def predict(model, image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(image).unsqueeze(0)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
    latency_ms = (time.perf_counter() - started) * 1000
    probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    return probs, latency_ms


def preview_markup(uri):
    return f'<div class="av-card av-preview"><img src="{uri}" alt="Uploaded leaf sample"/></div>'


def scanner_markup(uri):
    return (
        f'<div class="scan-wrap"><img src="{uri}" alt="Leaf sample under scan"/>'
        '<div class="scan-overlay"></div><div class="scan-line"></div></div>'
    )


def run_scan(model, image, uri, scan_slot):
    scan_slot.markdown(scanner_markup(uri), unsafe_allow_html=True)
    status = st.empty()
    bar = status.progress(0, text=SCAN_PHASES[0][0])
    for label, value in SCAN_PHASES:
        bar.progress(value, text=label)
        time.sleep(0.3)
    try:
        probs, latency = predict(model, image)
    except RuntimeError:
        status.empty()
        return None
    bar.progress(1.0, text="Diagnosis compiled")
    time.sleep(0.3)
    status.empty()
    top = max(range(len(probs)), key=probs.__getitem__)
    return {"probs": probs, "latency": latency, "top": top, "confidence": probs[top] * 100}


def remedy_card(kind, title, body):
    return (
        f'<div class="av-remedy {kind}"><div class="av-remedy-label">{title}</div>'
        f'<div class="av-remedy-body">{body}</div></div>'
    )


def render_results(result):
    probs, top, confidence = result["probs"], result["top"], result["confidence"]
    info = DISEASES[top]

    render(f"""
        <div class="av-card">
          <h4>Diagnosis</h4>
          <div class="av-diag-crop">{info["crop"]}</div>
          <div class="av-diag-name">{info["name"]}</div>
          <div class="av-diag-mr devanagari">{info["marathi"]}</div>
          <span class="av-badge badge-{info["tone"]}">{info["status"]}</span>
        </div>
    """)

    if confidence >= LOW_CONFIDENCE:
        note = f"Above the {LOW_CONFIDENCE:.0f}% decision threshold."
    else:
        note = f"Below the {LOW_CONFIDENCE:.0f}% decision threshold. Re-capture the sample in daylight and rescan before acting."

    prob_rows = "".join(
        f'<div class="av-prob"><div class="av-prob-row"><span>{DISEASES[i]["name"]}</span>'
        f'<span>{p * 100:.1f}%</span></div>'
        f'<div class="bar-track"><div class="bar-fill{"" if i == top else " muted"}" style="width:{p * 100:.1f}%"></div></div></div>'
        for i, p in enumerate(probs)
    )
    render(f"""
        <div class="av-card">
          <h4>Confidence</h4>
          <div class="av-conf-value">{confidence:.1f}<span>%</span></div>
          <div class="av-conf-note">{note}</div>
          <div class="bar-track lg"><div class="bar-fill" style="width:{confidence:.1f}%"></div></div>
          {prob_rows}
        </div>
    """)

    render(
        remedy_card("organic", "Organic control", info["organic"])
        + remedy_card("chemical", "Chemical control", info["chemical"])
    )


model, model_status = load_model()
st.session_state.setdefault("history", [])

render(CSS)
render(HERO)

with st.sidebar:
    render("""
        <div class="av-side-brand">AgriVision</div>
        <div class="av-side-sub">Maharashtra crop diagnostics</div>
        <div class="av-side-h">Project</div>
        <div class="av-side-item"><span>Challenge</span><strong>SIH Problem 61</strong></div>
        <div class="av-side-item"><span>Goal</span><strong>SDG 2: Zero Hunger</strong></div>
        <div class="av-side-item"><span>Stage</span><strong>Milestone 2 Build</strong></div>
    """)
    status_text, status_color = ("Ready", "#1E6B3F") if model is not None else ("Unavailable", "#A32020")
    render(f"""
        <div class="av-side-h">Runtime</div>
        <div class="av-side-item"><span>Model</span><strong>MobileNetV2 · 3 classes</strong></div>
        <div class="av-side-item"><span>Status</span><strong style="color:{status_color}">{status_text}</strong></div>
        <div class="av-side-item"><span>Execution</span><strong>CPU, edge</strong></div>
        <div class="av-side-item"><span>Input</span><strong>224 × 224 RGB</strong></div>
    """)

tab_diag, tab_impact, tab_arch = st.tabs(["Diagnostic Engine", "Impact Metrics", "Architecture"])

with tab_diag:
    left, right = st.columns([5, 6], gap="large")

    with left:
        render("""
            <div class="av-card">
              <div class="av-section-title">Sample intake</div>
              <p class="av-muted">Upload a single leaf photograph for classification. JPG, JPEG or PNG, up to 10 MB.</p>
            </div>
        """)
        uploaded = st.file_uploader(
            "Leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        render(GUIDE_CARD)

    with right:
        if uploaded is None:
            render(EMPTY_STATE)
        else:
            data = uploaded.getvalue()
            image, error = read_image(data)
            if error:
                st.error(error)
            else:
                uri = to_data_uri(image)
                key = hashlib.sha256(data).hexdigest()
                scan_slot = st.empty()
                result = None

                if model is not None:
                    if st.session_state.get("scan_key") == key:
                        result = st.session_state.scan
                    else:
                        result = run_scan(model, image, uri, scan_slot)
                        if result is not None:
                            st.session_state.scan_key = key
                            st.session_state.scan = result
                            st.session_state.history.append(
                                {"top": result["top"], "confidence": result["confidence"], "latency": result["latency"]}
                            )

                scan_slot.markdown(preview_markup(uri), unsafe_allow_html=True)

                if model is None:
                    st.warning(MODEL_ERRORS[model_status])
                elif result is None:
                    st.error("Inference could not be completed for this image. Retry with a different sample.")
                else:
                    render_results(result)

with tab_impact:
    history = st.session_state.history
    total = len(history)
    detections = sum(1 for h in history if h["top"] != 0)
    mean_conf = sum(h["confidence"] for h in history) / total if total else None
    last_latency = history[-1]["latency"] if total else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scans this session", total)
    m2.metric("Disease detections", detections)
    m3.metric("Mean confidence", f"{mean_conf:.1f}%" if total else "—")
    m4.metric("Last inference", f"{last_latency:.0f} ms" if total else "—")

    chart_col, spec_col = st.columns([3, 2], gap="large")
    with chart_col:
        render('<div class="av-card"><div class="av-section-title">Diagnoses by class</div></div>')
        if total:
            names = [DISEASES[i]["name"] for i in range(NUM_CLASSES)]
            counts = [sum(1 for h in history if h["top"] == i) for i in range(NUM_CLASSES)]
            st.bar_chart(pd.DataFrame({"Scans": counts}, index=names))
        else:
            render('<div class="av-card"><p class="av-muted">No scans recorded in this session.</p></div>')

    with spec_col:
        render('<div class="av-card"><h4>Operating specification</h4></div>')
        st.markdown(
            "| Parameter | Specification |\n"
            "|---|---|\n"
            "| Input | RGB, resized to 224 × 224 |\n"
            "| Formats | JPG, JPEG, PNG, max 10 MB |\n"
            "| Inference | CPU, single image, no network call |\n"
            "| Output | 3-class softmax with top-1 confidence |\n"
            "| Decision threshold | 60% confidence |"
        )

with tab_arch:
    steps = "".join(
        f'<div class="av-step"><div class="idx">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for n, t, d in PIPELINE
    )
    render('<div class="av-section-title" style="margin-bottom:12px">Inference pipeline</div>')
    render(f'<div class="av-pipe">{steps}</div>')

    a1, a2 = st.columns(2, gap="large")
    with a1:
        render('<div class="av-card"><h4>Model</h4></div>')
        st.markdown(
            "| Component | Detail |\n"
            "|---|---|\n"
            "| Backbone | MobileNetV2 (torchvision) |\n"
            "| Classifier head | Linear, 1280 → 3 |\n"
            "| Weights | agrivision_model.pth |\n"
            "| Normalization | ImageNet mean and std |\n"
            "| Runtime | PyTorch, eval(), inference_mode |"
        )
    with a2:
        render('<div class="av-card"><h4>Classes</h4></div>')
        st.markdown(
            "| Index | Class | Crop |\n"
            "|---|---|---|\n"
            "| 0 | Cotton - Healthy | Cotton |\n"
            "| 1 | Cotton - Bacterial Blight | Cotton |\n"
            "| 2 | Sugarcane - Red Rot | Sugarcane |"
        )
        render('<div class="av-card"><h4>Stack</h4><p class="av-muted">Streamlit · Pillow · PyTorch · torchvision · pandas</p></div>')
