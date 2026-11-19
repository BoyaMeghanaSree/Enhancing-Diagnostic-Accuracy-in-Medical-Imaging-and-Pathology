"""
AI-Assisted Medical Scan Triage (Streamlit).   Run: streamlit run app.py
EDUCATIONAL PROTOTYPE - not a medical device; do not use for real diagnosis.
Env vars: CLINICIAN_PASSWORD (set it!), MEDSCAN_HOME (project root).
"""
import hmac
import html as _html
import json
import os
import sqlite3
import uuid
from datetime import datetime
import streamlit as st
from PIL import Image
from explain_and_triage import load_run, analyse


def get_root():
    if "MEDSCAN_HOME" in os.environ:
        return os.path.abspath(os.environ["MEDSCAN_HOME"])
    current_dir = os.path.abspath(os.path.dirname(__file__))
    if os.path.basename(current_dir).lower() == "src":
        return os.path.abspath(os.path.join(current_dir, ".."))
    return current_dir


ROOT = get_root()
DB, CASE_DIR, CFG_PATH = (os.path.join(ROOT, p) for p in ("cases_v2.db", "cases", "modalities.json"))
os.makedirs(CASE_DIR, exist_ok=True)

DEFAULT_CFG = {
    "Chest X-Ray": {"model": "resnet50", "run_dir": "runs_xray", "normal_classes": ["NORMAL"]},
    "Brain Tumor MRI": {"model": "resnet50", "run_dir": "runs_mri", "normal_classes": ["no_tumor", "notumor", "Healthy"]},
}

if not os.path.exists(CFG_PATH):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CFG, f, indent=2)

with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = json.load(f)

for c in CFG.values():
    c["run_dir"] = os.path.abspath(os.path.join(ROOT, c["run_dir"]))

st.set_page_config(page_title="Scan triage", page_icon="🩻", layout="wide")

# ----------------------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root{
  --ink:#12303A; --muted:#5B7580; --canvas:#EEF3F4; --surface:#FFFFFF; --line:#D8E3E6;
  --teal:#0E7C86; --teal-d:#0A5E66; --risk:#C2362F; --risk-bg:#FBEDEC; --ok:#1F7A5A; --ok-bg:#E8F4EF;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{font-family:'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif; color:var(--ink);}
.stApp{background:var(--canvas);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
.block-container{max-width:1180px; padding-top:2rem; padding-bottom:4rem;}

/* hero */
.hero{display:flex; justify-content:space-between; align-items:flex-end; gap:1.5rem; flex-wrap:wrap; padding:.2rem 0 1.3rem;}
.hero h1{font-family:'Newsreader',Georgia,serif; font-weight:600; font-size:clamp(2rem,4vw,2.9rem); line-height:1.05; letter-spacing:-.01em; margin:0 0 .55rem; padding:0; color:var(--ink);}
.hero p{margin:0; max-width:54ch; color:var(--muted); font-size:1.02rem; line-height:1.5;}
.chips{display:flex; gap:.5rem; flex-wrap:wrap;}
.chip{display:inline-flex; align-items:center; gap:.5rem; border:1px solid var(--line); background:var(--surface); border-radius:999px; padding:.35rem .85rem; font-size:.88rem; font-weight:500;}
.chip i{width:.5rem; height:.5rem; border-radius:50%; background:var(--ok); display:inline-block;}
.chip.off{color:var(--muted);} .chip.off i{background:#B5C4C9;}
.fineprint{color:var(--muted); font-size:.85rem; margin:-.4rem 0 1rem;}
.sect{font-family:'Newsreader',Georgia,serif; font-weight:600; font-size:1.35rem; margin:.1rem 0 .6rem;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:.3rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:46px; padding:0 1.1rem; color:var(--muted); font-weight:500; border-radius:10px 10px 0 0; transition:color .2s, background .2s;}
.stTabs [data-baseweb="tab"]:hover{color:var(--teal-d); background:rgba(14,124,134,.06);}
.stTabs [aria-selected="true"]{color:var(--teal-d); font-weight:600;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--teal); height:3px;}

/* cards */
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface); border-radius:16px;}
[data-testid="stExpander"]{border:1px solid var(--line); border-radius:14px; background:var(--surface); overflow:hidden; margin-bottom:.6rem;}
[data-testid="stExpander"] summary{font-weight:500;}
[data-testid="stImage"] img{border-radius:12px;}
[data-testid="stAlert"]{border-radius:12px;}

/* inputs */
.stTextInput input, .stTextArea textarea{border-radius:10px;}
[data-baseweb="select"] > div{border-radius:10px;}
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section{border:1.5px dashed #9FB7BD; border-radius:14px; background:#F8FBFB; transition:border-color .2s, background .2s;}
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploader"] section:hover{border-color:var(--teal); background:#F1F8F8;}

/* buttons */
.stButton > button{border-radius:10px; border:1px solid var(--teal); background:var(--teal); color:#fff; font-weight:600; padding:.6rem 1.3rem; transition:background .2s, box-shadow .2s, transform .15s;}
.stButton > button:hover{background:var(--teal-d); border-color:var(--teal-d); color:#fff; box-shadow:0 6px 16px rgba(14,124,134,.25);}
.stButton > button:active{transform:translateY(1px);}
.stButton > button:focus-visible{outline:3px solid rgba(14,124,134,.4); outline-offset:2px;}
.stButton > button:disabled{background:#C9D6DA; border-color:#C9D6DA; color:#fff; box-shadow:none;}

/* sidebar */
[data-testid="stSidebar"]{background:var(--surface); border-right:1px solid var(--line);}

/* verdict panel with risk meter */
.verdict{--c:var(--ok); --bg:var(--ok-bg); border:1px solid var(--line); border-left:6px solid var(--c); border-radius:14px; background:var(--surface); padding:1.1rem 1.25rem 1.15rem; margin-bottom:1rem;}
.verdict.high{--c:var(--risk); --bg:var(--risk-bg);}
.v-top{display:flex; align-items:center; justify-content:space-between; gap:.75rem; flex-wrap:wrap;}
.badge{display:inline-flex; align-items:center; gap:.5rem; padding:.3rem .85rem; border-radius:999px; background:var(--bg); color:var(--c); font-weight:600; font-size:.95rem;}
.badge i{width:.55rem; height:.55rem; border-radius:50%; background:var(--c); display:inline-block;}
.v-pred{font-family:'Newsreader',Georgia,serif; font-size:1.8rem; line-height:1.15; margin:.6rem 0 .1rem; overflow-wrap:anywhere;}
.v-sub{color:var(--muted); font-size:.9rem;}
.meter{position:relative; height:12px; border-radius:999px; background:#E3ECEE; margin:1.2rem 0 .45rem;}
.meter .fill{position:absolute; inset:0 auto 0 0; width:var(--w); border-radius:999px; background:var(--c); animation:fill .9s cubic-bezier(.2,.7,.2,1);}
.meter .tick{position:absolute; top:-6px; bottom:-6px; left:var(--t); width:2px; background:var(--ink); border-radius:2px;}
.meter-l{display:flex; justify-content:space-between; gap:1rem; color:var(--muted); font-size:.85rem; flex-wrap:wrap;}
@keyframes fill{from{width:0;}}

/* class probabilities */
.prob{display:grid; grid-template-columns:minmax(100px,170px) 1fr 56px; align-items:center; gap:.75rem; margin:.5rem 0; font-size:.93rem;}
.prob .track{height:8px; border-radius:999px; background:#E3ECEE; overflow:hidden;}
.prob .bar{height:100%; border-radius:999px; background:#9FB7BD; width:var(--w); transition:width .6s ease;}
.prob .pct{text-align:right; font-variant-numeric:tabular-nums; color:var(--muted);}
.prob.top .bar{background:var(--teal);}
.prob.top .nm, .prob.top .pct{color:var(--ink); font-weight:600;}

/* queue stats and empty state */
.stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.8rem; margin:.5rem 0 1.2rem;}
.stat{background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:.85rem 1rem;}
.stat b{display:block; font-family:'Newsreader',Georgia,serif; font-size:2rem; line-height:1.1;}
.stat span{color:var(--muted); font-size:.88rem;}
.stat.risk b{color:var(--risk);}
.empty{border:1.5px dashed #B5C9CE; border-radius:16px; padding:2.6rem 1.5rem; text-align:center; color:var(--muted); background:#F8FBFB;}
.empty b{display:block; color:var(--ink); font-family:'Newsreader',Georgia,serif; font-size:1.3rem; font-weight:600; margin-bottom:.35rem;}

@media (max-width:640px){ .prob{grid-template-columns:1fr 48px;} .prob .track{grid-column:1 / -1; order:3;} }
@media (prefers-reduced-motion:reduce){ *{animation:none !important; transition:none !important;} }

/* polished interactions and responsive refinements */
.stApp {
  background:
    radial-gradient(ellipse at 8% 0%, rgba(14,124,134,.075), transparent 34rem),
    var(--canvas);
}
.block-container { padding-left: clamp(1rem, 4vw, 2.5rem); padding-right: clamp(1rem, 4vw, 2.5rem); }
.hero { padding: .6rem 0 1.6rem; }
.hero h1 { letter-spacing: -.035em; }
.hero p { max-width: 62ch; }
.chip { box-shadow: 0 2px 8px rgba(18,48,58,.035); }
.stTabs [data-baseweb="tab-list"] { padding: .25rem .25rem 0; }
.stTabs [data-baseweb="tab"] { transition: all .18s ease; }
.stTabs [aria-selected="true"] { background: rgba(14,124,134,.08); }
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: rgba(216,227,230,.9);
  box-shadow: 0 8px 28px rgba(18,48,58,.045);
  transition: box-shadow .2s ease, transform .2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  box-shadow: 0 12px 32px rgba(18,48,58,.075);
}
.stTextInput label, .stSelectbox label, .stFileUploader label, .stCheckbox label {
  font-weight: 500;
}
.stButton > button { min-height: 2.8rem; }
.stButton > button:focus-visible, input:focus-visible, textarea:focus-visible {
  outline: 3px solid rgba(14,124,134,.35);
  outline-offset: 2px;
}
[data-testid="stFileUploaderDropzone"] { padding: 1.2rem; }
[data-testid="stExpander"] { transition: border-color .18s ease, box-shadow .18s ease; }
[data-testid="stExpander"]:hover { border-color: #A9C6CC; box-shadow: 0 4px 14px rgba(18,48,58,.04); }
[data-testid="stImage"] img { box-shadow: 0 3px 12px rgba(18,48,58,.08); }
@media (max-width: 760px) {
  .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
  .hero { align-items: flex-start; gap: 1rem; }
  .hero h1 { font-size: clamp(1.8rem, 8vw, 2.4rem); }
  .hero p { font-size: .95rem; }
  .chips { gap: .35rem; }
  .chip { font-size: .78rem; padding: .3rem .65rem; }
  .stTabs [data-baseweb="tab"] { padding: 0 .75rem; font-size: .9rem; }
}
@media (prefers-reduced-motion: reduce) {
  [data-testid="stVerticalBlockBorderWrapper"],
  [data-testid="stExpander"], .stTabs [data-baseweb="tab"] { transition: none !important; }
}



/* Final contrast + consistency pass: all visible form labels stay dark. */
:root { color-scheme: light; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { color: #12303A !important; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
.stTextInput label, .stSelectbox label, .stFileUploader label,
.stCheckbox label, .stSlider label, .stTextArea label,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
  color: #45616B !important;
  opacity: 1 !important;
}
.stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] * {
  color: #45616B !important;
  opacity: 1 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] * { color: #0A5E66 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] label * {
  color: #45616B !important; opacity: 1 !important;
}
/* Consistent light inputs, including BaseWeb's nested select control. */
.stTextInput input, .stTextArea textarea,
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] section {
  background: #FFFFFF !important; color: #12303A !important;
  border-color: #C9D9DE !important;
}
[data-baseweb="select"] *, [data-baseweb="input"] *,
[data-testid="stFileUploaderDropzone"] * { color: #12303A !important; }
[data-baseweb="popover"] *, [role="listbox"] *, [role="option"] {
  color: #12303A !important; background-color: #FFFFFF !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #718891 !important; opacity: 1 !important; }
.stCheckbox label, .stCheckbox label * { color: #12303A !important; }
.stButton > button:disabled { color: #45616B !important; background: #DCE6E8 !important; }

/* Contrast fixes: keep the sidebar and every form control readable. */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
  background: #ffffff !important;
}
[data-testid="stSidebar"] * { color: #12303A !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] span { color: #45616B !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] * { color: #45616B !important; }

/* Light controls with dark readable text (Streamlit/BaseWeb variants). */
.stTextInput input, .stTextArea textarea,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] section {
  background: #ffffff !important;
  color: #12303A !important;
  border-color: #C9D9DE !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #718891 !important; }
[data-baseweb="select"] *, [data-baseweb="input"] * { color: #12303A !important; }
[data-testid="stFileUploaderDropzone"] button { background: #EAF4F5 !important; color: #0A5E66 !important; }
.stTabs [data-baseweb="tab"] { color: #45616B !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #0A5E66 !important; }
.stMarkdown, .stCaption, label { color: #12303A; }
@media (max-width: 700px) {
  [data-testid="stSidebar"] { min-width: 0 !important; }
  .block-container { padding-top: 1rem !important; }
}

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------
# Data helpers (unchanged logic)
# ----------------------------------------------------------------------------------------
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    with con:
        con.execute("""CREATE TABLE IF NOT EXISTS cases(
            id TEXT PRIMARY KEY, created TEXT, modality TEXT, reference TEXT, pred TEXT,
            confidence REAL, p_abnormal REAL, flag TEXT, low_conf INTEGER, probs TEXT,
            img_path TEXT, cam_path TEXT, status TEXT, note TEXT)""")
    return con


def safe_rerun():
    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()


def trained(m):
    if m not in CFG:
        return False
    d = os.path.join(CFG[m]["run_dir"], CFG[m]["model"])
    return os.path.exists(os.path.join(d, "best.pt")) and os.path.exists(os.path.join(d, "results.json"))


@st.cache_resource
def cached_model(model_key, run_dir):
    return load_run(run_dir, model_key)


# ----------------------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------------------
def card():
    try:
        return st.container(border=True)
    except TypeError:  # older Streamlit without bordered containers
        return st.container()


def verdict_html(flag, pred, conf, p_abn, threshold=None):
    high = flag == "HIGH-RISK"
    p_abn = min(1.0, max(0.0, float(p_abn)))
    label = "High risk: review first" if high else "Routine"
    tick = ""
    right = ""
    if threshold is not None:
        tick = f'<div class="tick" style="--t:{threshold * 100:.0f}%"></div>'
        right = f"<span>Flagged at {threshold * 100:.0f}% or more</span>"
    return (
        f'<div class="verdict {"high" if high else "routine"}">'
        f'<div class="v-top"><span class="badge"><i></i>{label}</span>'
        f'<span class="v-sub">Model confidence {conf * 100:.0f}%</span></div>'
        f'<div class="v-pred">{_html.escape(str(pred))}</div>'
        f'<div class="v-sub">Top prediction</div>'
        f'<div class="meter" style="--w:{p_abn * 100:.1f}%"><div class="fill"></div>{tick}</div>'
        f'<div class="meter-l"><span>Chance of abnormal: {p_abn * 100:.0f}%</span>{right}</div>'
        f'</div>'
    )


def prob_html(probs):
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    rows = []
    for i, (k, v) in enumerate(items):
        v = min(1.0, max(0.0, float(v)))
        cls = "prob top" if i == 0 else "prob"
        rows.append(
            f'<div class="{cls}"><span class="nm">{_html.escape(str(k))}</span>'
            f'<div class="track"><div class="bar" style="--w:{v * 100:.1f}%"></div></div>'
            f'<span class="pct">{v * 100:.1f}%</span></div>'
        )
    return "".join(rows)


# ----------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------
st.sidebar.markdown('<div class="sect">Settings</div>', unsafe_allow_html=True)
threshold = st.sidebar.slider("High-risk threshold P(abnormal) >=", 0.05, 0.95, 0.50, 0.05)
st.sidebar.caption(
    "Scans at or above this chance of abnormal are flagged high risk. "
    "Suggested: 0.50 for chest X-ray, 0.05 for brain MRI. Set it before you submit."
)
avail = [m for m in CFG if trained(m)]
st.sidebar.write("Trained scan types:", ", ".join(avail) if avail else "none yet")

# ----------------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------------
chips = "".join(
    f'<span class="chip{"" if m in avail else " off"}"><i></i>{_html.escape(m)}</span>' for m in CFG
)
st.markdown(
    '<div class="hero"><div><h1>Scan triage</h1>'
    '<p>Upload a chest X-ray or brain MRI. The model flags what a clinician should see first '
    'and shows the regions that drove its answer.</p></div>'
    f'<div class="chips">{chips}</div></div>'
    '<div class="fineprint">Educational prototype. AI output is a screening aid only and must be reviewed by a clinician.</div>',
    unsafe_allow_html=True,
)

tab_p, tab_d = st.tabs(["Upload a scan", "Clinician queue"])

# ----------------------------------------------------------------------------------------
# Upload tab
# ----------------------------------------------------------------------------------------
with tab_p:
    if not avail:
        st.error("No trained models found.")
        st.info(
            "Train a model first:\n"
            "```bash\n"
            "python medical_imaging_models.py --data data_xray --model resnet50 --out runs_xray\n"
            "```\n"
            "Then place the run folder next to app.py and restart the app."
        )
    else:
        left, right = st.columns([5, 6], gap="large")
        with left:
            with card():
                st.markdown('<div class="sect">Scan details</div>', unsafe_allow_html=True)
                modality = st.selectbox("Type of scan", avail)
                f = st.file_uploader("Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
                consent = st.checkbox("I acknowledge this is an AI screening tool.")
                submit = st.button("Submit scan", disabled=not (f and consent), use_container_width=True)

        with right:
            if submit and f:
                with st.spinner("Analysing scan..."):
                    img = Image.open(f).convert("RGB")
                    c = CFG[modality]
                    model, meta = cached_model(c["model"], c["run_dir"])
                    r = analyse(img, model, c["model"], meta, c["normal_classes"], threshold)
                    cid = uuid.uuid4().hex[:8].upper()
                    ip, cp = os.path.join(CASE_DIR, f"{cid}.png"), os.path.join(CASE_DIR, f"{cid}_cam.png")
                    img.save(ip)
                    r["heat"].save(cp)
                    con = db()
                    try:
                        with con:
                            con.execute(
                                "INSERT INTO cases VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (cid, datetime.now().strftime("%Y-%m-%d %H:%M"), modality, "", r["pred"], r["conf"],
                                 r["p_abn"], r["flag"], r["low_conf"], json.dumps(r["probs"]), ip, cp,
                                 "Pending review", "")
                            )
                    finally:
                        con.close()
                st.session_state["last"] = dict(cid=cid, img=img, r=r, threshold=threshold, modality=modality)
                st.success(f"Scan submitted. Case ID: **{cid}**")

            last = st.session_state.get("last")
            if last:
                r = last["r"]
                st.markdown(f'<div class="sect">Screening summary: case {last["cid"]}</div>', unsafe_allow_html=True)
                st.markdown(verdict_html(r["flag"], r["pred"], r["conf"], r["p_abn"], last["threshold"]),
                            unsafe_allow_html=True)
                if r["low_conf"]:
                    st.warning("Low model confidence - requires prompt clinical review.")
                i1, i2 = st.columns(2)
                i1.image(last["img"], caption="Original scan", use_container_width=True)
                i2.image(r["heat"], caption="Grad-CAM: regions driving the prediction", use_container_width=True)
                st.markdown('<div class="sect" style="margin-top:1rem">Class probabilities</div>', unsafe_allow_html=True)
                st.markdown(prob_html(r["probs"]), unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="empty"><b>Your result will appear here</b>'
                    'Choose a scan type, add an image and tick the acknowledgement, then select Submit scan.</div>',
                    unsafe_allow_html=True,
                )

# ----------------------------------------------------------------------------------------
# Clinician queue
# ----------------------------------------------------------------------------------------
with tab_d:
    expected = os.environ.get("CLINICIAN_PASSWORD", "doctor123")
    if "CLINICIAN_PASSWORD" not in os.environ:
        st.warning("Using default demo password (`doctor123`). Set CLINICIAN_PASSWORD environment variable for production.")
    pw = st.text_input("Clinician password", type="password")
    if pw and hmac.compare_digest(pw.encode(), expected.encode()):
        con = db()
        try:
            rows = con.execute("SELECT * FROM cases ORDER BY (status='Reviewed'), (flag='HIGH-RISK') DESC, created DESC").fetchall()
        finally:
            con.close()

        pending = sum(1 for x in rows if x["status"] != "Reviewed")
        high_pending = sum(1 for x in rows if x["status"] != "Reviewed" and x["flag"] == "HIGH-RISK")
        reviewed = len(rows) - pending
        st.markdown(
            '<div class="stats">'
            f'<div class="stat risk"><b>{high_pending}</b><span>High risk, waiting</span></div>'
            f'<div class="stat"><b>{pending}</b><span>Waiting for review</span></div>'
            f'<div class="stat"><b>{reviewed}</b><span>Reviewed</span></div>'
            f'<div class="stat"><b>{len(rows)}</b><span>Total cases</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if not rows:
            st.markdown(
                '<div class="empty"><b>The queue is empty</b>Submitted scans appear here, high risk first.</div>',
                unsafe_allow_html=True,
            )
        for r in rows:
            dot = "🔴" if r["flag"] == "HIGH-RISK" else "🟢"
            with st.expander(f"{dot} [{r['flag']}] {r['id']} | {r['modality']} | {r['pred']} ({r['confidence']*100:.0f}%) | {r['status']}"):
                st.caption(f"Created {r['created']}")
                st.markdown(verdict_html(r["flag"], r["pred"], r["confidence"], r["p_abnormal"]),
                            unsafe_allow_html=True)
                if r["low_conf"]:
                    st.warning("Low model confidence - review carefully.")
                c1, c2 = st.columns(2)
                if os.path.exists(r["img_path"]):
                    try:
                        c1.image(Image.open(r["img_path"]), caption="Original", use_container_width=True)
                    except Exception as e:
                        c1.warning(f"Error loading original image: {e}")
                else:
                    c1.warning("Original scan file missing on disk.")

                if os.path.exists(r["cam_path"]):
                    try:
                        c2.image(Image.open(r["cam_path"]), caption="Grad-CAM (regions driving prediction)", use_container_width=True)
                    except Exception as e:
                        c2.warning(f"Error loading CAM image: {e}")
                else:
                    c2.warning("Grad-CAM visualization file missing on disk.")

                st.markdown('<div class="sect" style="margin-top:.8rem">Class probabilities</div>', unsafe_allow_html=True)
                try:
                    st.markdown(prob_html(json.loads(r["probs"])), unsafe_allow_html=True)
                except Exception:
                    st.write(r["probs"])

                note = st.text_area("Clinician note", value=r["note"] or "", key=f"note_{r['id']}")
                if st.button("Mark as reviewed", key=f"rev_{r['id']}"):
                    con = db()
                    try:
                        with con:
                            con.execute("UPDATE cases SET status='Reviewed', note=? WHERE id=?", (note, r["id"]))
                    finally:
                        con.close()
                    safe_rerun()
    elif pw:
        st.error("Wrong password.")
