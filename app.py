iimport base64
import hashlib
import io
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

try:
    import altair as alt
    ALT_AVAILABLE = True
except ImportError:
    ALT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

st.set_page_config(
    page_title="AgriVision Maharashtra",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "agrivision_model.pth"
NUM_CLASSES = 3
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
DEFAULT_THRESHOLD = 60.0
RING_CIRCUMFERENCE = 339.292

DISEASES = {
    0: {
        "crop": "Cotton",
        "short": "Healthy",
        "name": "Cotton - Healthy",
        "marathi": "कापूस - निरोगी",
        "status": "No pathogen detected",
        "tone": "healthy",
        "color": "#2E7D4F",
        "chemical": "None",
        "organic": "Maintain soil moisture and compost.",
        "protocol": [
            "Continue weekly scouting across the plot.",
            "Hold irrigation and nutrition at the current schedule.",
            "Re-scan immediately if new lesions or yellowing appear.",
        ],
    },
    1: {
        "crop": "Cotton",
        "short": "Bacterial Blight",
        "name": "Cotton - Bacterial Blight",
        "marathi": "कापूस - जीवाणूजन्य करपा",
        "status": "Bacterial infection detected",
        "tone": "alert",
        "color": "#C0392B",
        "chemical": "Copper Oxychloride 50% WP @ 25g/10L.",
        "organic": "Neem seed kernel extract (5%) foliar spray.",
        "protocol": [
            "Mark affected rows and remove severely infected leaf debris.",
            "Spray in the early morning; repeat at 10-day intervals.",
            "Re-scan the same leaf position after 7 days to verify response.",
        ],
    },
    2: {
        "crop": "Sugarcane",
        "short": "Red Rot",
        "name": "Sugarcane - Red Rot",
        "marathi": "ऊस - तांबेरा / लाल कूज",
        "status": "Fungal infection detected",
        "tone": "alert",
        "color": "#C0392B",
        "chemical": "Uproot and burn infected clumps.",
        "organic": "Soil application of Trichoderma viride.",
        "protocol": [
            "Isolate the affected clump; do not ratoon from that plot.",
            "Destroy infected setts before the spore load spreads.",
            "Treat the remaining stand and re-scan within one week.",
        ],
    },
}

MODEL_ERRORS = {
    "runtime": "PyTorch is not installed in this runtime. The interface is live, but classification is disabled.",
    "missing": "Model weights are not present. Place agrivision_model.pth next to app.py and restart the service.",
    "incompatible": "The checkpoint does not match the MobileNetV2 three-class architecture. Re-export and redeploy.",
    "corrupt": "The model file could not be read. Re-export agrivision_model.pth and redeploy.",
}

SCAN_PHASES = [
    ("Decoding image buffer", 0.14),
    ("Analyzing cellular structure", 0.34),
    ("Isolating lesion boundaries", 0.54),
    ("Normalizing color channels", 0.72),
    ("Running MobileNetV2 inference", 0.88),
    ("Scoring class probabilities", 0.97),
]

PIPELINE = [
    ("01", "Capture", "Leaf photograph from a phone or field camera. JPG, JPEG or PNG, up to 10 MB."),
    ("02", "Validate", "Pillow decodes the buffer and forces RGB, flattening alpha and palette modes."),
    ("03", "Preprocess", "Resize to 224 × 224, cast to tensor, normalize with ImageNet mean and std."),
    ("04", "Infer", "MobileNetV2 backbone with a 3-class head, evaluated on CPU in inference mode."),
    ("05", "Score", "Softmax over logits. Top-1 class and the full distribution are reported."),
    ("06", "Advise", "Class-mapped chemical and organic remedies from the agronomy lookup table."),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

:root {
  --av-ink: #13211A;
  --av-muted: #5F6F66;
  --av-line: #E2E7DF;
  --av-green-900: #07291D;
  --av-green-700: #14543A;
  --av-green-500: #2E7D4F;
  --av-green-50: #EDF6F0;
  --av-amber-50: #FFF4E6;
  --av-amber: #C87916;
  --av-clay: #B4342B;
  --av-card-shadow: 0 1px 2px rgba(12,30,22,.05), 0 10px 26px -12px rgba(12,30,22,.18);
  --av-card-shadow-hover: 0 2px 4px rgba(12,30,22,.06), 0 20px 40px -16px rgba(12,30,22,.26);
}

html, body, .stApp {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: var(--av-ink);
}

.stApp {
  background-color: #F4F6F1;
  background-image:
    radial-gradient(1100px 520px at 92% -12%, rgba(46,125,79,.13), transparent 62%),
    radial-gradient(760px 420px at -8% 8%, rgba(200,121,22,.09), transparent 64%),
    repeating-linear-gradient(103deg, rgba(19,33,26,.030) 0 1px, transparent 1px 8px);
  background-attachment: fixed;
}
.stApp::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: .30;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.82' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)' opacity='.35'/%3E%3C/svg%3E");
}
.block-container { padding: 1.6rem 2rem 3.5rem; max-width: 1320px; position: relative; z-index: 1; }
.dev { font-family: 'Noto Sans Devanagari', 'IBM Plex Sans', sans-serif; }
.mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }

@keyframes av-rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes av-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes av-sweep { 0% { transform: translateX(-40%); } 100% { transform: translateX(140%); } }
@keyframes av-pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(94,207,140,.55); } 70% { box-shadow: 0 0 0 7px rgba(94,207,140,0); } }
@keyframes av-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes av-laser { 0% { top: 6px; } 50% { top: calc(100% - 9px); } 100% { top: 6px; } }
@keyframes av-dash { to { stroke-dashoffset: var(--off); } }
@keyframes av-spin { to { transform: rotate(360deg); } }

.av-rise { animation: av-rise .55s cubic-bezier(.22,.68,.32,1) both; animation-delay: var(--d, 0s); }

/* ── header instrument panel ─────────────────────────────── */
.av-hero {
  position: relative; overflow: hidden; border-radius: 12px;
  background: linear-gradient(118deg, #06231A 0%, #0E4230 46%, #1C6B4A 100%);
  box-shadow: 0 18px 44px -18px rgba(6,35,26,.55), inset 0 1px 0 rgba(255,255,255,.09);
  padding: 26px 30px; margin-bottom: 18px; color: #fff;
  display: flex; justify-content: space-between; align-items: center; gap: 24px; flex-wrap: wrap;
}
.av-hero::before {
  content: ""; position: absolute; inset: 0; opacity: .16; pointer-events: none;
  background-image: repeating-linear-gradient(74deg, rgba(255,255,255,.55) 0 1px, transparent 1px 13px);
  mask-image: linear-gradient(100deg, transparent 30%, #000 100%);
  -webkit-mask-image: linear-gradient(100deg, transparent 30%, #000 100%);
}
.av-hero::after {
  content: ""; position: absolute; top: 0; bottom: 0; width: 34%; pointer-events: none;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,.11), transparent);
  animation: av-sweep 7.5s ease-in-out infinite;
}
.av-hero-left { display: flex; gap: 16px; align-items: center; position: relative; z-index: 2; }
.av-mark {
  width: 52px; height: 52px; border-radius: 12px; flex: 0 0 auto; display: grid; place-items: center;
  background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.20);
}
.av-kicker {
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .16em;
  text-transform: uppercase; color: rgba(255,255,255,.62); margin-bottom: 5px;
}
.av-hero h1 {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: clamp(1.6rem, 3vw, 2.15rem);
  letter-spacing: -.02em; margin: 0; color: #fff; line-height: 1.06;
}
.av-hero-sub { margin: 7px 0 0; color: rgba(255,255,255,.74); font-size: .93rem; max-width: 60ch; line-height: 1.5; }

.av-tele { display: flex; gap: 10px; align-items: stretch; position: relative; z-index: 2; flex-wrap: wrap; }
.av-tele-item {
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); border-radius: 10px;
  padding: 9px 14px; min-width: 104px; transition: background .25s ease, transform .25s ease;
}
.av-tele-item:hover { background: rgba(255,255,255,.15); transform: translateY(-2px); }
.av-tele-k {
  font-family: 'IBM Plex Mono', monospace; font-size: .62rem; letter-spacing: .12em;
  text-transform: uppercase; color: rgba(255,255,255,.58);
}
.av-tele-v { font-family: 'Space Grotesk', sans-serif; font-size: 1.22rem; font-weight: 700; margin-top: 3px; }
.av-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 7px; }
.av-dot.on { background: #5ECF8C; animation: av-pulse 2s infinite; }
.av-dot.off { background: #F2A65A; animation: av-pulse 2s infinite; }
.av-tele-live { display: flex; align-items: center; font-size: .84rem; font-weight: 500; padding-top: 6px; }

/* ── cards ───────────────────────────────────────────────── */
.av-card {
  background: #fff; border: 1px solid var(--av-line); border-radius: 12px; padding: 20px 22px;
  box-shadow: var(--av-card-shadow); margin-bottom: 14px;
  transition: transform .28s cubic-bezier(.22,.68,.32,1), box-shadow .28s ease, border-color .28s ease;
}
.av-card:hover { transform: translateY(-3px); box-shadow: var(--av-card-shadow-hover); border-color: #D3DED3; }
.av-card.flat:hover { transform: none; box-shadow: var(--av-card-shadow); }
.av-eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: .67rem; letter-spacing: .14em; text-transform: uppercase;
  color: #7C8A81; margin: 0 0 12px; display: flex; align-items: center; gap: 8px;
}
.av-eyebrow::after { content: ""; flex: 1; height: 1px; background: var(--av-line); }
.av-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.16rem; font-weight: 600; letter-spacing: -.01em; margin: 0 0 6px; }
.av-muted { color: var(--av-muted); font-size: .89rem; line-height: 1.6; margin: 0; }
.av-list { margin: 10px 0 0; padding: 0; list-style: none; }
.av-list li {
  position: relative; padding-left: 20px; color: var(--av-muted); font-size: .87rem; line-height: 1.75;
  transition: color .2s ease, padding-left .2s ease;
}
.av-list li::before { content: ""; position: absolute; left: 3px; top: .68em; width: 6px; height: 6px; border-radius: 2px; background: #C3D2C6; transition: background .2s ease; }
.av-list li:hover { color: var(--av-ink); padding-left: 24px; }
.av-list li:hover::before { background: var(--av-green-500); }

.av-protocol { counter-reset: p; margin: 0; padding: 0; list-style: none; }
.av-protocol li {
  counter-increment: p; position: relative; padding: 11px 0 11px 40px; font-size: .89rem; line-height: 1.5;
  border-bottom: 1px dashed var(--av-line); color: #2A3833; transition: background .2s ease;
}
.av-protocol li:last-child { border-bottom: none; }
.av-protocol li:hover { background: #F8FAF7; }
.av-protocol li::before {
  content: counter(p, decimal-leading-zero); position: absolute; left: 0; top: 11px;
  font-family: 'IBM Plex Mono', monospace; font-size: .7rem; font-weight: 600; color: var(--av-green-500);
  background: var(--av-green-50); border-radius: 6px; padding: 3px 6px;
}

/* ── empty state ─────────────────────────────────────────── */
.av-empty {
  border: 1.5px dashed #BCCBBF; border-radius: 12px; background: rgba(255,255,255,.72);
  padding: 58px 30px; text-align: center; position: relative; overflow: hidden;
}
.av-empty::after {
  content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .5;
  background-image: repeating-linear-gradient(103deg, rgba(46,125,79,.06) 0 1px, transparent 1px 9px);
}
.av-empty-icon { animation: av-fade .8s ease both; }
.av-empty-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.14rem; margin-top: 16px; position: relative; z-index: 1; }
.av-empty-text { color: var(--av-muted); font-size: .89rem; margin: 8px auto 0; max-width: 400px; line-height: 1.6; position: relative; z-index: 1; }
.av-empty-meta {
  margin-top: 20px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; position: relative; z-index: 1;
}
.av-tag {
  font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .06em; padding: 5px 10px;
  border-radius: 999px; background: #fff; border: 1px solid var(--av-line); color: #4C5B53;
  transition: transform .2s ease, border-color .2s ease, color .2s ease;
}
.av-tag:hover { transform: translateY(-2px); border-color: #A9C6B3; color: var(--av-green-700); }

/* ── preview + scanner ───────────────────────────────────── */
.av-frame { position: relative; background: #fff; border: 1px solid var(--av-line); border-radius: 12px; padding: 10px; box-shadow: var(--av-card-shadow); margin-bottom: 14px; }
.av-frame img { display: block; width: 100%; border-radius: 8px; }
.av-frame-tag {
  position: absolute; top: 20px; left: 20px; z-index: 3; font-family: 'IBM Plex Mono', monospace;
  font-size: .64rem; letter-spacing: .1em; text-transform: uppercase; color: #fff;
  background: rgba(7,33,25,.66); border-radius: 6px; padding: 4px 9px; backdrop-filter: blur(4px);
  max-width: calc(100% - 40px); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
  .scan-laser { display: none; }
}
.scan-overlay { position: absolute; inset: 10px; border-radius: 8px; overflow: hidden; z-index: 2; background: linear-gradient(180deg, rgba(7,41,29,.05), rgba(7,41,29,.42)); }
.scan-grid { position: absolute; inset: 0; background-image: linear-gradient(rgba(126,255,178,.16) 1px, transparent 1px), linear-gradient(90deg, rgba(126,255,178,.16) 1px, transparent 1px); background-size: 26px 26px; }
.scan-laser {
  position: absolute; left: 0; right: 0; height: 3px; border-radius: 3px; z-index: 3;
  background: linear-gradient(90deg, transparent, #8DFFBE, transparent); box-shadow: 0 0 18px 2px rgba(141,255,190,.75);
  animation: av-laser 1.5s cubic-bezier(.45,0,.55,1) infinite;
}
.scan-corner { position: absolute; width: 22px; height: 22px; border: 2px solid #8DFFBE; z-index: 3; }
.scan-corner.tl { top: 8px; left: 8px; border-right: none; border-bottom: none; border-radius: 6px 0 0 0; }
.scan-corner.tr { top: 8px; right: 8px; border-left: none; border-bottom: none; border-radius: 0 6px 0 0; }
.scan-corner.bl { bottom: 8px; left: 8px; border-right: none; border-top: none; border-radius: 0 0 0 6px; }
.scan-corner.br { bottom: 8px; right: 8px; border-left: none; border-top: none; border-radius: 0 0 6px 0; }

.scan-panel { background: #0A2A1F; border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; box-shadow: var(--av-card-shadow); }
.scan-phase { display: flex; align-items: center; gap: 10px; padding: 7px 0; font-size: .86rem; color: rgba(255,255,255,.42); transition: color .3s ease; }
.scan-phase .glyph {
  width: 18px; height: 18px; border-radius: 50%; flex: 0 0 auto; display: grid; place-items: center;
  border: 1.5px solid rgba(255,255,255,.28); font-size: .58rem; font-family: 'IBM Plex Mono', monospace;
}
.scan-phase.done { color: rgba(255,255,255,.78); }
.scan-phase.done .glyph { border-color: #5ECF8C; color: #5ECF8C; background: rgba(94,207,140,.12); }
.scan-phase.live { color: #fff; font-weight: 500; }
.scan-phase.live .glyph { border-color: #8DFFBE; color: #8DFFBE; animation: av-spin 1.1s linear infinite; border-top-color: transparent; }

/* ── diagnosis ───────────────────────────────────────────── */
.av-diag { display: flex; gap: 22px; align-items: flex-start; flex-wrap: wrap; }
.av-diag-main { flex: 1 1 260px; min-width: 240px; }
.av-diag-crop { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: #7C8A81; }
.av-diag-name { font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.45rem, 2.4vw, 1.9rem); font-weight: 700; letter-spacing: -.02em; margin: 6px 0 3px; line-height: 1.12; }
.av-diag-mr { font-size: 1.06rem; color: #3B4A43; margin-bottom: 14px; }
.av-badge { display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px; border-radius: 999px; font-size: .76rem; font-weight: 600; }
.badge-healthy { background: #E4F4EA; color: #1B6639; border: 1px solid #C3E4CE; }
.badge-alert { background: #FBEAE7; color: #9C2A22; border: 1px solid #F0CCC6; }

.av-ring { display: flex; align-items: center; gap: 16px; }
.av-ring svg { transform: rotate(-90deg); }
.av-ring circle { fill: none; stroke-width: 11; stroke-linecap: round; }
.ring-track { stroke: #EDF1EC; }
.ring-value { stroke: var(--rc); stroke-dasharray: 339.292; stroke-dashoffset: var(--off); animation: av-dash 1.1s cubic-bezier(.22,.68,.32,1) both; }
.av-ring-num { font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; letter-spacing: -.03em; line-height: 1; color: var(--av-green-900); }
.av-ring-num small { font-size: 1rem; color: #7C8A81; margin-left: 2px; }
.av-ring-k { font-family: 'IBM Plex Mono', monospace; font-size: .65rem; letter-spacing: .12em; text-transform: uppercase; color: #7C8A81; margin-top: 6px; }

.av-conf-note { font-size: .84rem; color: var(--av-muted); line-height: 1.55; margin: 12px 0 0; padding-top: 12px; border-top: 1px solid var(--av-line); }
.av-conf-note.low { color: #9C5A16; }

.av-prob { margin-top: 14px; }
.av-prob-row { display: flex; justify-content: space-between; align-items: baseline; font-size: .82rem; color: #3B4A43; margin-bottom: 7px; }
.av-prob-row .pct { font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.bar-track { height: 7px; background: #EDF1EC; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; transform-origin: left; animation: av-grow .8s cubic-bezier(.22,.68,.32,1) both; }
.bar-top { background: linear-gradient(90deg, var(--av-green-700), #63BE8B); }
.bar-rest { background: #C6D2C8; }

.av-remedy { border-radius: 12px; padding: 16px 18px; border: 1px solid; box-shadow: var(--av-card-shadow); height: 100%; transition: transform .26s ease, box-shadow .26s ease; }
.av-remedy:hover { transform: translateY(-3px); box-shadow: var(--av-card-shadow-hover); }
.av-remedy.organic { background: var(--av-green-50); border-color: #CFE7D7; border-left: 4px solid var(--av-green-500); }
.av-remedy.chemical { background: var(--av-amber-50); border-color: #F6DCB9; border-left: 4px solid var(--av-amber); }
.av-remedy-label { font-family: 'IBM Plex Mono', monospace; font-size: .66rem; letter-spacing: .13em; text-transform: uppercase; display: flex; align-items: center; gap: 8px; }
.av-remedy.organic .av-remedy-label { color: var(--av-green-500); }
.av-remedy.chemical .av-remedy-label { color: #A85F0C; }
.av-remedy-body { font-size: .99rem; font-weight: 500; margin-top: 9px; color: var(--av-ink); line-height: 1.5; }

.av-note {
  border-radius: 12px; padding: 14px 18px; background: #FBFCFA; border: 1px solid var(--av-line);
  border-left: 3px solid #9FB3A6; color: var(--av-muted); font-size: .84rem; line-height: 1.55; margin-top: 4px;
}

/* ── pipeline + tables ───────────────────────────────────── */
.av-pipe { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 18px; }
.av-step {
  background: #fff; border: 1px solid var(--av-line); border-radius: 12px; padding: 18px 16px;
  box-shadow: var(--av-card-shadow); position: relative; overflow: hidden;
  transition: transform .28s cubic-bezier(.22,.68,.32,1), box-shadow .28s ease, border-color .28s ease;
}
.av-step::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px; background: var(--av-green-500); transform: scaleX(0); transform-origin: left; transition: transform .35s cubic-bezier(.22,.68,.32,1); }
.av-step:hover { transform: translateY(-5px); box-shadow: var(--av-card-shadow-hover); border-color: #CFE0D3; }
.av-step:hover::after { transform: scaleX(1); }
.av-step .idx { font-family: 'IBM Plex Mono', monospace; font-size: .72rem; font-weight: 600; letter-spacing: .1em; color: #93A299; transition: color .25s ease; }
.av-step:hover .idx { color: var(--av-green-500); }
.av-step .t { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem; margin: 6px 0 7px; }
.av-step .d { font-size: .83rem; color: var(--av-muted); line-height: 1.5; }

.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }

/* ── streamlit primitives ────────────────────────────────── */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--av-line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }

[data-testid="stMetric"] {
  background: #fff; border: 1px solid var(--av-line); border-radius: 12px; padding: 16px 18px;
  box-shadow: var(--av-card-shadow); transition: transform .28s ease, box-shadow .28s ease, border-color .28s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-4px); box-shadow: var(--av-card-shadow-hover); border-color: #CFE0D3; }
[data-testid="stMetricLabel"] p { font-family: 'IBM Plex Mono', monospace !important; font-size: .68rem !important; letter-spacing: .1em; text-transform: uppercase; color: #7C8A81 !important; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif !important; color: var(--av-green-900); font-size: 1.85rem !important; letter-spacing: -.02em; }

[data-testid="stFileUploaderDropzone"] {
  border-radius: 12px !important; border: 1.5px dashed #BCCBBF !important; background: #FBFCF9 !important;
  transition: border-color .25s ease, background .25s ease, transform .25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--av-green-500) !important; background: var(--av-green-50) !important; transform: translateY(-2px); }

.stButton > button, .stDownloadButton > button {
  border-radius: 10px !important; border: 1px solid #C6D3C9 !important; background: #fff !important;
  font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 500 !important; font-size: .85rem !important;
  color: #24342C !important; transition: all .22s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--av-green-500) !important; color: var(--av-green-700) !important;
  background: var(--av-green-50) !important; transform: translateY(-2px);
  box-shadow: 0 8px 18px -10px rgba(20,84,58,.5) !important;
}

.stProgress > div > div > div > div,
[data-testid="stProgress"] > div > div > div > div {
  background-image: linear-gradient(90deg, var(--av-green-700), #6FCF97) !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--av-line); background: transparent; }
.stTabs [data-baseweb="tab"] {
  height: 46px; padding: 0 18px; border-radius: 10px 10px 0 0; font-weight: 500; font-size: .92rem;
  color: #5F6F66; transition: color .2s ease, background .2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--av-green-700); background: rgba(46,125,79,.06); }
.stTabs [aria-selected="true"] { color: var(--av-green-900) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--av-green-500) !important; height: 3px !important; border-radius: 3px; }

[data-testid="stAlert"] { border-radius: 12px !important; border: 1px solid var(--av-line) !important; box-shadow: var(--av-card-shadow); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 12px; overflow: hidden; }

/* ── sidebar ─────────────────────────────────────────────── */
.av-side-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.av-side-brand .sq {
  width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center;
  background: linear-gradient(140deg, #0E4230, #2E7D4F); box-shadow: 0 6px 14px -6px rgba(14,66,48,.7);
}
.av-side-name { font-family: 'Space Grotesk', sans-serif; font-size: 1.16rem; font-weight: 700; letter-spacing: -.01em; color: var(--av-green-900); }
.av-side-sub { font-size: .79rem; color: var(--av-muted); margin: 6px 0 0; }
.av-side-h {
  font-family: 'IBM Plex Mono', monospace; font-size: .64rem; letter-spacing: .15em; text-transform: uppercase;
  color: var(--av-green-500); font-weight: 600; margin: 24px 0 4px;
}
.av-side-item { display: flex; justify-content: space-between; gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--av-line); font-size: .83rem; transition: padding-left .2s ease; }
.av-side-item:hover { padding-left: 4px; }
.av-side-item span { color: var(--av-muted); }
.av-side-item strong { color: var(--av-ink); font-weight: 600; text-align: right; }
.av-feed { margin-top: 4px; }
.av-feed-row {
  display: flex; justify-content: space-between; gap: 8px; align-items: center; padding: 8px 10px; margin-bottom: 6px;
  background: #F7F9F6; border: 1px solid var(--av-line); border-radius: 9px; font-size: .78rem;
  transition: transform .2s ease, background .2s ease;
}
.av-feed-row:hover { transform: translateX(3px); background: var(--av-green-50); }
.av-feed-row .t { font-family: 'IBM Plex Mono', monospace; color: #8A978E; font-size: .7rem; }
.av-feed-row .n { font-weight: 600; color: var(--av-ink); }
.av-feed-empty { font-size: .8rem; color: #93A299; padding: 8px 0; }
</style>
"""

LEAF_SVG = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#8DFFBE" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>'
    '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>'
)

LEAF_LARGE = LEAF_SVG.replace("#8DFFBE", "#1F6B4F").replace('width="26" height="26"', 'width="54" height="54"')

BRAND_SQUARE = (
    '<div class="sq">'
    + LEAF_SVG.replace("#8DFFBE", "#ffffff").replace('width="26" height="26"', 'width="18" height="18"')
    + "</div>"
)


def render(markup):
    """Emit an HTML block, collapsing source indentation."""
    st.markdown(
        "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip()),
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model():
    if not TORCH_AVAILABLE:
        return None, "runtime"
    if not MODEL_PATH.is_file():
        return None, "missing"
    try:
        try:
            model = models.mobilenet_v2(weights=None)
        except TypeError:
            model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
        try:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        model.eval()
        return model, None
    except FileNotFoundError:
        return None, "missing"
    except (RuntimeError, KeyError, AttributeError, TypeError, ValueError):
        return None, "incompatible"
    except Exception:
        return None, "corrupt"


def read_image(data):
    if not data:
        return None, "The upload arrived empty. Capture the image again and retry."
    if len(data) > MAX_UPLOAD_BYTES:
        return None, f"This image is larger than the {MAX_UPLOAD_MB} MB limit. Export a compressed copy and upload again."
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.convert("RGB"), None
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        return None, "That file is not a readable JPG or PNG image. Upload an unmodified photograph straight from the camera."


def to_data_uri(image, max_side=760):
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


def frame_markup(uri, label, scanning=False):
    overlay = ""
    if scanning:
        overlay = (
            '<div class="scan-overlay"><div class="scan-grid"></div></div>'
            '<div class="scan-laser"></div>'
            '<div class="scan-corner tl"></div><div class="scan-corner tr"></div>'
            '<div class="scan-corner bl"></div><div class="scan-corner br"></div>'
        )
    return (
        f'<div class="av-frame av-rise"><div class="av-frame-tag">{label}</div>'
        f'<img src="{uri}" alt="Leaf sample"/>{overlay}</div>'
    )


def phase_markup(active):
    rows = []
    for index, (label, _) in enumerate(SCAN_PHASES):
        if index < active:
            state, glyph = "done", "&#10003;"
        elif index == active:
            state, glyph = "live", ""
        else:
            state, glyph = "wait", ""
        rows.append(f'<div class="scan-phase {state}"><span class="glyph">{glyph}</span>{label}</div>')
    return f'<div class="scan-panel av-rise">{"".join(rows)}</div>'


def run_scan(model, image, uri, frame_slot, phase_slot, bar_slot):
    frame_slot.markdown(frame_markup(uri, "Scanning sample", scanning=True), unsafe_allow_html=True)
    phase_slot.markdown(phase_markup(0), unsafe_allow_html=True)
    bar = bar_slot.progress(0.04, text="Preparing sample")

    for index, (label, value) in enumerate(SCAN_PHASES):
        bar.progress(value, text=label + "...")
        phase_slot.markdown(phase_markup(index), unsafe_allow_html=True)
        time.sleep(0.3)

    try:
        probs, latency = predict(model, image)
    except (RuntimeError, ValueError, TypeError):
        frame_slot.empty()
        phase_slot.empty()
        bar_slot.empty()
        return None

    bar.progress(1.0, text="Diagnosis compiled")
    phase_slot.markdown(phase_markup(len(SCAN_PHASES)), unsafe_allow_html=True)
    time.sleep(0.3)

    top = max(range(len(probs)), key=probs.__getitem__)
    return {
        "probs": probs,
        "latency": latency,
        "top": top,
        "confidence": probs[top] * 100,
    }


def diagnosis_card(info, confidence, threshold):
    offset = RING_CIRCUMFERENCE * (1 - confidence / 100)
    ring_color = info["color"] if confidence >= threshold else "#C87916"
    return f"""
    <div class="av-card av-rise" style="--d:.04s">
      <p class="av-eyebrow">Classification result</p>
      <div class="av-diag">
        <div class="av-diag-main">
          <div class="av-diag-crop">{info["crop"]} · class {info["index"]}</div>
          <div class="av-diag-name">{info["name"]}</div>
          <div class="av-diag-mr dev">{info["marathi"]}</div>
          <span class="av-badge badge-{info["tone"]}">{info["status"]}</span>
        </div>
        <div class="av-ring">
          <svg width="120" height="120" viewBox="0 0 120 120">
            <circle class="ring-track" cx="60" cy="60" r="54"></circle>
            <circle class="ring-value" cx="60" cy="60" r="54" style="--off:{offset:.2f};--rc:{ring_color}"></circle>
          </svg>
          <div>
            <div class="av-ring-num">{confidence:.1f}<small>%</small></div>
            <div class="av-ring-k">Top-1 confidence</div>
          </div>
        </div>
      </div>
    </div>
    """


def confidence_note(confidence, threshold, latency):
    if confidence >= threshold:
        return (
            f'<p class="av-conf-note">Cleared the {threshold:.0f}% decision threshold. '
            f"Inference completed in {latency:.0f} ms on CPU.</p>"
        )
    return (
        f'<p class="av-conf-note low">Below the {threshold:.0f}% decision threshold. '
        "Re-capture the leaf in even daylight, fill the frame, and rescan before applying any treatment.</p>"
    )


def probability_card(probs, top, threshold, latency):
    rows = "".join(
        f"""
        <div class="av-prob">
          <div class="av-prob-row"><span>{DISEASES[i]["name"]}</span><span class="pct">{p * 100:.1f}%</span></div>
          <div class="bar-track"><div class="bar-fill {"bar-top" if i == top else "bar-rest"}" style="width:{max(p * 100, 0.6):.1f}%"></div></div>
        </div>
        """
        for i, p in enumerate(probs)
    )
    return f"""
    <div class="av-card av-rise" style="--d:.10s">
      <p class="av-eyebrow">Class distribution</p>
      {rows}
      {confidence_note(probs[top] * 100, threshold, latency)}
    </div>
    """


def remedy_markup(info):
    protocol = "".join(f"<li>{step}</li>" for step in info["protocol"])
    return f"""
    <div class="av-card av-rise" style="--d:.16s">
      <p class="av-eyebrow">Treatment &amp; field protocol</p>
      <div class="av-remedy organic">
        <div class="av-remedy-label">Organic control</div>
        <div class="av-remedy-body">{info["organic"]}</div>
      </div>
      <div style="height:12px"></div>
      <div class="av-remedy chemical">
        <div class="av-remedy-label">Chemical / mechanical control</div>
        <div class="av-remedy-body">{info["chemical"]}</div>
      </div>
      <div style="height:16px"></div>
      <ol class="av-protocol">{protocol}</ol>
    </div>
    """


def hero_markup(history, engine_ready):
    total = len(history)
    mean_conf = sum(h["confidence"] for h in history) / total if total else None
    dot = "on" if engine_ready else "off"
    state = "Engine ready" if engine_ready else "Engine offline"
    conf = f"{mean_conf:.0f}%" if total else "—"
    scans = str(total)
    return f"""
    <div class="av-hero av-rise">
      <div class="av-hero-left">
        <div class="av-mark">{LEAF_SVG}</div>
        <div>
          <div class="av-kicker">SIH Problem 61 · Cotton &amp; Sugarcane</div>
          <h1>AgriVision Maharashtra</h1>
          <p class="av-hero-sub">Leaf-level disease classification with class-mapped chemical and organic guidance, running on-device.</p>
        </div>
      </div>
      <div class="av-tele">
        <div class="av-tele-item">
          <div class="av-tele-k">Scans</div>
          <div class="av-tele-v">{scans}</div>
        </div>
        <div class="av-tele-item">
          <div class="av-tele-k">Mean conf.</div>
          <div class="av-tele-v">{conf}</div>
        </div>
        <div class="av-tele-item">
          <div class="av-tele-k">Status</div>
          <div class="av-tele-live"><span class="av-dot {dot}"></span>{state}</div>
        </div>
      </div>
    </div>
    """


EMPTY_STATE = f"""
<div class="av-empty av-rise" style="--d:.06s">
  <div class="av-empty-icon">{LEAF_LARGE}</div>
  <div class="av-empty-title">Awaiting leaf sample</div>
  <div class="av-empty-text">Load a close-up of a single cotton or sugarcane leaf. The console returns a diagnosis, confidence distribution and treatment protocol.</div>
  <div class="av-empty-meta">
    <span class="av-tag">JPG</span><span class="av-tag">JPEG</span><span class="av-tag">PNG</span>
    <span class="av-tag">≤ 10 MB</span><span class="av-tag">224 × 224 RGB</span>
  </div>
</div>
"""

GUIDE_CARD = """
<div class="av-card av-rise" style="--d:.14s">
  <p class="av-eyebrow">Capture protocol</p>
  <ul class="av-list">
    <li>One leaf per frame, lesion centered</li>
    <li>Diffuse daylight — no flash, no hard shadow</li>
    <li>Steady shot: motion blur drops confidence sharply</li>
    <li>Scan 4–5 leaves per plot before deciding on a spray</li>
  </ul>
</div>
"""

INTAKE_CARD = """
<div class="av-card av-rise">
  <p class="av-eyebrow">Sample intake</p>
  <div class="av-title">Load a leaf image</div>
  <p class="av-muted">Classification starts automatically on upload. Nothing leaves this process — inference runs locally on CPU.</p>
</div>
"""

ADVISORY = """
<div class="av-note av-rise" style="--d:.22s">
  Output is advisory. Cross-check severe or ambiguous cases with the nearest Krishi Vigyan Kendra or
  state plant protection officer before applying treatments at plot scale.
</div>
"""


def sidebar_markup(model, model_status, history):
    ready = model is not None
    dot = "on" if ready else "off"
    state = "Ready" if ready else "Unavailable"
    color = "#1B6639" if ready else "#9C5A16"
    detail = "MobileNetV2 · 3 classes" if ready else MODEL_ERRORS[model_status][:46] + "…"

    if history:
        feed = "".join(
            f"""
            <div class="av-feed-row">
              <div><div class="n">{DISEASES[h["top"]]["short"]}</div><div class="t">{h["time"]}</div></div>
              <div class="t">{h["confidence"]:.0f}%</div>
            </div>
            """
            for h in reversed(history[-5:])
        )
    else:
        feed = '<div class="av-feed-empty">No scans recorded yet.</div>'

    return f"""
    <div class="av-side-brand">{BRAND_SQUARE}<div class="av-side-name">AgriVision</div></div>
    <div class="av-side-sub">Maharashtra crop diagnostic console</div>

    <div class="av-side-h">Programme</div>
    <div class="av-side-item"><span>Challenge</span><strong>SIH Problem 61</strong></div>
    <div class="av-side-item"><span>Goal</span><strong>SDG 2: Zero Hunger</strong></div>
    <div class="av-side-item"><span>Stage</span><strong>Milestone 2 Build</strong></div>

    <div class="av-side-h">Runtime</div>
    <div class="av-side-item"><span>Engine</span><strong>{detail}</strong></div>
    <div class="av-side-item"><span>Status</span><strong style="color:{color}"><span class="av-dot {dot}"></span>{state}</strong></div>
    <div class="av-side-item"><span>Execution</span><strong>CPU · offline</strong></div>
    <div class="av-side-item"><span>Input</span><strong>224 × 224 RGB</strong></div>

    <div class="av-side-h">Session activity</div>
    <div class="av-feed">{feed}</div>
    """


model, model_status = load_model()
engine_ready = model is not None

st.session_state.setdefault("history", [])
st.session_state.setdefault("uploader_seq", 0)
st.session_state.setdefault("threshold", DEFAULT_THRESHOLD)

render(CSS)

history = st.session_state.history
render(hero_markup(history, engine_ready))

with st.sidebar:
    render(sidebar_markup(model, model_status, history))
    st.session_state.threshold = st.slider(
        "Decision threshold (%)",
        min_value=50.0,
        max_value=95.0,
        step=5.0,
        value=float(st.session_state.threshold),
        help="Confidence below this value flags the scan as needing a re-capture.",
    )

tab_diag, tab_impact, tab_arch = st.tabs(["Diagnostic Engine", "Impact Metrics", "Architecture"])

with tab_diag:
    left, right = st.columns([5, 6], gap="large")

    with left:
        render(INTAKE_CARD)
        uploaded = st.file_uploader(
            "Leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_seq}",
        )
        control_a, control_b = st.columns(2)
        with control_a:
            rerun_scan = st.button("Re-run scan", use_container_width=True, disabled=uploaded is None)
        with control_b:
            clear_sample = st.button("Clear sample", use_container_width=True, disabled=uploaded is None)
        render(GUIDE_CARD)

    with right:
        if clear_sample:
            st.session_state.uploader_seq += 1
            st.session_state.pop("scan_key", None)
            st.session_state.pop("scan", None)
            st.rerun()

        if rerun_scan:
            st.session_state.pop("scan_key", None)

        if uploaded is None:
            render(EMPTY_STATE)
        else:
            data = uploaded.getvalue()
            image, error = read_image(data)
            if error:
                st.error(error)
                render(EMPTY_STATE)
            else:
                uri = to_data_uri(image)
                key = hashlib.sha256(data).hexdigest()
                frame_slot = st.empty()
                phase_slot = st.empty()
                bar_slot = st.empty()
                result = None

                if engine_ready:
                    if st.session_state.get("scan_key") == key:
                        result = st.session_state.get("scan")
                    else:
                        result = run_scan(model, image, uri, frame_slot, phase_slot, bar_slot)
                        if result is not None:
                            st.session_state.scan_key = key
                            st.session_state.scan = result
                            st.session_state.history.append({
                                "top": result["top"],
                                "confidence": result["confidence"],
                                "latency": result["latency"],
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "file": uploaded.name,
                            })
                            history = st.session_state.history

                phase_slot.empty()
                bar_slot.empty()
                frame_label = (
                    f"Diagnosed · {DISEASES[result['top']]['crop']} · {DISEASES[result['top']]['short']}"
                    if result is not None
                    else "Sample loaded"
                )
                frame_slot.markdown(frame_markup(uri, frame_label), unsafe_allow_html=True)

                threshold = st.session_state.threshold

                if not engine_ready:
                    st.warning(MODEL_ERRORS[model_status])
                elif result is None:
                    st.error("Inference could not be completed for this image. Try a different, well-lit sample.")
                else:
                    info = dict(DISEASES[result["top"]], index=result["top"])
                    render(diagnosis_card(info, result["confidence"], threshold))

                    c1, c2 = st.columns([1, 1], gap="medium")
                    with c1:
                        render(probability_card(result["probs"], result["top"], threshold, result["latency"]))
                    with c2:
                        render(remedy_markup(info))

                    render(f"""
                    <div class="av-card flat av-rise" style="--d:.20s">
                      <p class="av-eyebrow">Sample record</p>
                      <div class="av-side-item"><span>File</span><strong>{uploaded.name}</strong></div>
                      <div class="av-side-item"><span>Source size</span><strong>{image.width} × {image.height} px</strong></div>
                      <div class="av-side-item"><span>Model input</span><strong>224 × 224 · RGB · normalized</strong></div>
                      <div class="av-side-item"><span>Latency</span><strong>{result["latency"]:.0f} ms</strong></div>
                      <div class="av-side-item"><span>Checksum</span><strong class="mono">{key[:16]}</strong></div>
                    </div>
                    """)

    render(ADVISORY)

with tab_impact:
    total = len(history)
    detections = sum(1 for h in history if h["top"] != 0)
    confident = sum(1 for h in history if h["confidence"] >= st.session_state.threshold)
    mean_conf = sum(h["confidence"] for h in history) / total if total else None
    mean_latency = sum(h["latency"] for h in history) / total if total else None

    render('<div class="av-card flat av-rise"><p class="av-eyebrow">Session telemetry</p>'
           '<div class="av-title">Diagnostic throughput</div>'
           '<p class="av-muted">Live counters for the current browser session. Values persist across tab switches.</p></div>')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scans", total)
    m2.metric("Detections", detections)
    m3.metric("Mean confidence", f"{mean_conf:.1f}%" if total else "—")
    m4.metric("Mean latency", f"{mean_latency:.0f} ms" if total else "—")

    chart_col, table_col = st.columns([3, 2], gap="large")

    with chart_col:
        render('<div class="av-card flat av-rise" style="--d:.06s"><p class="av-eyebrow">Distribution</p>'
               '<div class="av-title">Diagnoses by class</div></div>')
        if total:
            counts = [sum(1 for h in history if h["top"] == i) for i in range(NUM_CLASSES)]
            names = [DISEASES[i]["name"] for i in range(NUM_CLASSES)]
            colors = [DISEASES[i]["color"] for i in range(NUM_CLASSES)]
            df = pd.DataFrame({"Class": names, "Scans": counts})
            if ALT_AVAILABLE:
                chart = (
                    alt.Chart(df)
                    .mark_bar(cornerRadiusEnd=6, height=26)
                    .encode(
                        x=alt.X("Scans:Q", axis=alt.Axis(grid=True, gridColor="#EDF1EC", title=None)),
                        y=alt.Y("Class:N", sort="-x", axis=alt.Axis(title=None, labelFontSize=12)),
                        color=alt.Color("Class:N", scale=alt.Scale(domain=names, range=colors), legend=None),
                        tooltip=["Class", "Scans"],
                    )
                    .properties(height=190)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(df.set_index("Class"))

            if total > 1:
                trend = pd.DataFrame({
                    "Scan": range(1, total + 1),
                    "Latency (ms)": [h["latency"] for h in history],
                })
                render('<div class="av-card flat av-rise" style="--d:.10s"><p class="av-eyebrow">Throughput</p>'
                       '<div class="av-title">Inference latency per scan</div></div>')
                if ALT_AVAILABLE:
                    line = (
                        alt.Chart(trend)
                        .mark_line(point=True, color="#14543A", strokeWidth=2.5)
                        .encode(
                            x=alt.X("Scan:O", axis=alt.Axis(title=None, grid=False)),
                            y=alt.Y("Latency (ms):Q", axis=alt.Axis(grid=True, gridColor="#EDF1EC")),
                            tooltip=["Scan", alt.Tooltip("Latency (ms):Q", format=".0f")],
                        )
                        .properties(height=180)
                    )
                    st.altair_chart(line, use_container_width=True)
                else:
                    st.line_chart(trend.set_index("Scan"))
        else:
            render('<div class="av-card flat av-rise" style="--d:.06s"><p class="av-muted">'
                   "No scans recorded yet. Run a diagnosis in the Diagnostic Engine tab to populate this panel.</p></div>")

    with table_col:
        render('<div class="av-card flat av-rise" style="--d:.08s"><p class="av-eyebrow">Coverage</p>'
               '<div class="av-title">Operating envelope</div></div>')
        st.markdown(
            "| Parameter | Specification |\n"
            "|---|---|\n"
            "| Target crops | Cotton, Sugarcane |\n"
            "| Classes | 3 (1 healthy, 2 disease) |\n"
            "| Input | RGB, 224 × 224 |\n"
            "| Formats | JPG, JPEG, PNG ≤ 10 MB |\n"
            "| Inference | Local CPU, no network call |\n"
            "| Output | Softmax over 3 classes |\n"
            f"| Threshold | {st.session_state.threshold:.0f}% confidence |"
        )
        render(f"""
        <div class="av-card flat av-rise" style="--d:.12s">
          <p class="av-eyebrow">Quality gate</p>
          <div class="av-side-item"><span>Scans above threshold</span><strong>{confident} / {total}</strong></div>
          <div class="av-side-item"><span>Disease detection rate</span><strong>{(detections / total * 100):.0f}%</strong></div>
          <div class="av-side-item"><span>Advisories issued</span><strong>{detections}</strong></div>
        </div>
        """ if total else """
        <div class="av-card flat av-rise" style="--d:.12s">
          <p class="av-eyebrow">Quality gate</p>
          <p class="av-muted">Detection rate and advisory counts appear after the first completed scan.</p>
        </div>
        """)
        render("""
        <div class="av-card flat av-rise" style="--d:.16s">
          <p class="av-eyebrow">SDG 2 alignment</p>
          <p class="av-muted">Earlier detection shortens the interval between symptom onset and treatment,
          limiting yield loss in cotton and preventing red rot from spreading through ratooned sugarcane plots.</p>
        </div>
        """)

with tab_arch:
    render('<div class="av-card flat av-rise"><p class="av-eyebrow">System design</p>'
           '<div class="av-title">From field photograph to treatment card</div>'
           '<p class="av-muted">A single synchronous pipeline. Each stage is deterministic and reproducible from the input checksum.</p></div>')

    steps = "".join(
        f'<div class="av-step av-rise" style="--d:{0.04 * i:.2f}s">'
        f'<div class="idx">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for i, (n, t, d) in enumerate(PIPELINE)
    )
    render(f'<div class="av-pipe">{steps}</div>')

    a1, a2, a3 = st.columns([4, 4, 3], gap="medium")
    with a1:
        render('<div class="av-card flat av-rise"><p class="av-eyebrow">Model</p></div>')
        st.markdown(
            "| Component | Detail |\n"
            "|---|---|\n"
            "| Backbone | MobileNetV2 (torchvision) |\n"
            "| Classifier head | Linear, 1280 → 3 |\n"
            "| Weights | agrivision_model.pth |\n"
            "| Normalization | ImageNet mean / std |\n"
            "| Mode | eval(), inference_mode |\n"
            "| Device | CPU |"
        )
    with a2:
        render('<div class="av-card flat av-rise"><p class="av-eyebrow">Class map</p></div>')
        st.markdown(
            "| Index | Class | Marathi |\n"
            "|---|---|---|\n"
            "| 0 | Cotton - Healthy | कापूस - निरोगी |\n"
            "| 1 | Cotton - Bacterial Blight | कापूस - जीवाणूजन्य करपा |\n"
            "| 2 | Sugarcane - Red Rot | ऊस - तांबेरा / लाल कूज |"
        )
    with a3:
        render("""
        <div class="av-card flat av-rise" style="--d:.06s">
          <p class="av-eyebrow">Stack</p>
          <div class="chip-row">
            <span class="av-tag">Streamlit</span><span class="av-tag">PyTorch</span>
            <span class="av-tag">torchvision</span><span class="av-tag">Pillow</span>
            <span class="av-tag">pandas</span><span class="av-tag">Altair</span>
          </div>
        </div>
        <div class="av-card flat av-rise" style="--d:.10s">
          <p class="av-eyebrow">Failure handling</p>
          <ul class="av-list">
            <li>Missing or mismatched weights → warning, UI stays live</li>
            <li>Oversized or corrupt upload → readable rejection</li>
            <li>Alpha / palette images → forced RGB conversion</li>
            <li>Inference fault → recoverable error, no traceback</li>
          </ul>
        </div>
        """)

    render("""
    <div class="av-card flat av-rise" style="--d:.14s">
      <p class="av-eyebrow">Deployment</p>
      <div class="av-side-item"><span>Entry point</span><strong class="mono">streamlit run app.py</strong></div>
      <div class="av-side-item"><span>Artifact</span><strong class="mono">agrivision_model.pth (state dict)</strong></div>
      <div class="av-side-item"><span>Network</span><strong>None required at inference time</strong></div>
      <div class="av-side-item"><span>State</span><strong>Per-session; nothing persisted to disk</strong></div>
    </div>
    """)
