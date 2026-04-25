"""
CS4241 — Introduction to Artificial Intelligence
Student: Farima Konaré | Index: 10012200004

Application: app.py
Purpose: Streamlit chat UI for the Academic City RAG Assistant.

Run: streamlit run app.py
"""

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import RAGPipeline
from src.memory import ConversationMemory

CONFIDENCE_THRESHOLD = 0.25

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACity RAG Assistant",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load API key from secrets (no UI input) ──────────────────────────────────
_api_key = ""
try:
    _api_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    pass
if _api_key:
    os.environ["GROQ_API_KEY"] = _api_key

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stApp"] {
    background: #f0f4f8 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* ── Dark sidebar ── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(175deg, #0d1b2e 0%, #13294b 55%, #1a3a5c 100%) !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption p {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] svg {
    fill: #94a3b8 !important;
}
section[data-testid="stSidebar"] .stSlider label p,
section[data-testid="stSidebar"] .stSelectbox label p {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
    color: #64748b !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] div {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #e2e8f0 !important;
}
/* segmented control in sidebar */
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-checked="true"] {
    color: #0d1b2e !important;
}

/* ── Main area container ── */
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
}

/* ── Header card ── */
.rag-header {
    background: linear-gradient(135deg, #0d1b2e 0%, #1e3a5f 60%, #1d4ed8 100%);
    border-radius: 14px;
    padding: 26px 32px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(13,27,46,0.3);
}
.rag-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 5px; height: 100%;
    background: linear-gradient(180deg, #FCD116 33%, #CE1126 33%, #CE1126 66%, #006B3F 66%);
}
.rag-header-title {
    color: #f1f5f9;
    margin: 0 0 4px 0;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.025em;
}
.rag-header-sub {
    color: #94a3b8;
    margin: 0 0 14px 0;
    font-size: 0.875rem;
    line-height: 1.5;
}
.badge-row { display: flex; gap: 7px; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    color: #cbd5e1;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
}

/* ── Sidebar brand block ── */
.sb-brand { padding: 8px 0 4px; }
.sb-brand-name {
    font-size: 1.2rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.03em;
}
.sb-brand-sub { font-size: 0.72rem; color: #64748b; margin-top: 1px; }
.ghana-strip {
    display: flex; height: 3px; border-radius: 3px;
    overflow: hidden; margin: 10px 0 4px; width: 72px;
}
.gs-r { background: #CE1126; flex: 1; }
.gs-g { background: #FCD116; flex: 1; }
.gs-b { background: #006B3F; flex: 1; }

/* ── Inline status pill ── */
.status-pill {
    display: flex; align-items: center; justify-content: center;
    gap: 7px; padding: 7px 14px; border-radius: 24px;
    font-size: 0.78rem; font-weight: 600; width: 100%;
    box-sizing: border-box;
}
.status-ok  { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: #16a34a; }
.status-err { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.3);  color: #dc2626; }
.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-ok  { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.dot-err { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

/* ── Welcome screen ── */
.welcome-wrap { text-align: center; padding: 20px 0 8px; }
.welcome-title { font-size: 1.05rem; font-weight: 600; color: #334155; margin-bottom: 4px; }
.welcome-sub   { font-size: 0.85rem; color: #94a3b8; }

/* ── Suggestion buttons ── */
div[data-testid="stColumns"] .stButton > button {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    color: #334155 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    text-align: left !important;
    height: auto !important;
    white-space: normal !important;
    line-height: 1.45 !important;
    font-size: 0.84rem !important;
    transition: border-color .15s, box-shadow .15s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
div[data-testid="stColumns"] .stButton > button:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.14) !important;
}

/* ── Chunk card ── */
.chunk-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #e5e7eb;
    border-radius: 0 10px 10px 0;
    padding: 11px 15px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.chunk-card.election { border-left-color: #3b82f6; }
.chunk-card.budget   { border-left-color: #8b5cf6; }
.chunk-meta {
    display: flex; align-items: center;
    gap: 8px; margin-bottom: 8px; flex-wrap: wrap;
}
.src-tag {
    display: inline-block; padding: 2px 9px;
    border-radius: 20px; font-size: 0.62rem;
    font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
}
.src-election { background: #dbeafe; color: #1d4ed8; }
.src-budget   { background: #ede9fe; color: #6d28d9; }
.chunk-num { font-size: 0.72rem; font-weight: 600; color: #9ca3af; }
.scores {
    margin-left: auto; display: flex;
    align-items: center; gap: 10px;
}
.score-label { font-size: 0.68rem; color: #9ca3af; }
.score-val   { font-size: 0.72rem; font-weight: 600; color: #374151; }
.bar-wrap {
    display: inline-flex; align-items: center; gap: 4px;
}
.bar-bg {
    width: 44px; height: 4px; background: #e5e7eb;
    border-radius: 2px; display: inline-block; vertical-align: middle;
}
.bar-fill {
    height: 4px; border-radius: 2px; display: inline-block;
}
.fill-high { background: #22c55e; }
.fill-mid  { background: #f59e0b; }
.fill-low  { background: #ef4444; }
.chunk-text {
    font-size: 0.82rem; color: #4b5563;
    line-height: 1.58; margin: 0;
}

/* ── Log panel ── */
.log-total {
    background: #f8fafc; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 10px 14px;
    text-align: center; margin-bottom: 12px;
}
.log-total-val   { font-size: 1.3rem; font-weight: 700; color: #0f172a; }
.log-total-label { font-size: 0.65rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .07em; }
.stage-row { font-size: 0.78rem; color: #64748b; margin-bottom: 2px; }
.stage-name { font-weight: 600; color: #374151; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 6px !important;
}

/* ── Low confidence warning ── */
.low-conf {
    display: flex; align-items: center; gap: 8px;
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 8px; padding: 8px 14px;
    font-size: 0.82rem; color: #92400e; margin-top: 6px;
}

/* ── Confidence badge ── */
.conf-pill {
    display: inline-flex; align-items: center; gap: 8px;
    border-radius: 999px; padding: 5px 10px; font-size: 0.74rem;
    font-weight: 700; margin: 0 0 8px 0; border: 1px solid transparent;
}
.conf-high {
    background: rgba(34,197,94,0.12);
    color: #15803d;
    border-color: rgba(34,197,94,0.25);
}
.conf-mid {
    background: rgba(245,158,11,0.15);
    color: #b45309;
    border-color: rgba(245,158,11,0.25);
}
.conf-low {
    background: rgba(239,68,68,0.12);
    color: #b91c1c;
    border-color: rgba(239,68,68,0.25);
}

/* ── Hide Streamlit footer/menu ── */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stMainBlockContainer"] { padding-top: 0.5rem !important; }

/* ── Sidebar toggle button — CLOSED state (arrow in main area) ── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[kind="sidebarButton"] {
    background: #1d4ed8 !important;
    border-radius: 0 8px 8px 0 !important;
    width: 28px !important;
    opacity: 1 !important;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.35) !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
button[kind="sidebarButton"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* ── Sidebar toggle button — OPEN state (arrow on sidebar right edge) ── */
[data-testid="stSidebarNavCollapseButton"] button,
section[data-testid="stSidebar"] button[aria-label*="ollapse"],
section[data-testid="stSidebar"] button[aria-label*="sidebar"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] {
    background: rgba(255, 255, 255, 0.18) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.55) !important;
    border-radius: 6px !important;
    opacity: 1 !important;
    width: 32px !important;
    height: 32px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stSidebarNavCollapseButton"] svg,
section[data-testid="stSidebar"] button[aria-label*="ollapse"] svg,
section[data-testid="stSidebar"] button[aria-label*="sidebar"] svg,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
    stroke: #ffffff !important;
    width: 18px !important;
    height: 18px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-name">ACity RAG</div>
        <div class="ghana-strip">
            <div class="gs-r"></div>
            <div class="gs-g"></div>
            <div class="gs-b"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Connection status — inline pill, no st.success/st.error box
    if os.environ.get("GROQ_API_KEY"):
        st.markdown(
            '<div class="status-pill status-ok">'
            '<span class="dot dot-ok"></span>LLM connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-err">'
            '<span class="dot dot-err"></span>No API key</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("**Knowledge Base**")
    source = st.segmented_control(
        "source",
        options=["Both", "Elections", "Budget"],
        default="Both",
        label_visibility="collapsed",
    )
    source_map = {"Both": None, "Elections": "election", "Budget": "budget"}
    source_filter = source_map[source or "Both"]

    st.divider()

    with st.expander(":material/tune: Advanced settings"):
        top_k = st.slider(
            "Chunks to retrieve",
            min_value=1, max_value=10, value=5,
            help="More chunks = broader context, slower response.",
        )
        template_id = st.selectbox(
            "Prompt template",
            options=[1, 2, 3],
            index=2,
            format_func=lambda x: {
                1: "1 — Baseline",
                2: "2 — Hallucination-controlled",
                3: "3 — With memory",
            }[x],
        )

    with st.expander(":material/science: Quick adversarial tests"):
        st.caption("Run edge-case queries quickly for demo and evaluation.")
        quick_tests = [
            "Who won?",
            "What is the capital of France?",
            "What did the president say about free healthcare in the 2025 budget?",
        ]
        for i, q in enumerate(quick_tests):
            if st.button(q, key=f"quick_test_{i}", use_container_width=True):
                st.session_state["pending_query"] = q
                st.rerun()

    st.divider()

    st.caption("**Model** llama-3.3-70b-versatile")
    st.caption("**Embed** all-MiniLM-L6-v2 · 384-dim")
    st.caption("**Index** FAISS IndexFlatIP")
    st.caption("**Retrieval** BM25 + Vector (0.4 / 0.6)")

    st.divider()

    if st.button(
        ":material/delete_sweep: Clear conversation",
        use_container_width=True,
        type="secondary",
    ):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()


# ─── Session state ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "pipeline" not in st.session_state or st.session_state.get("pipeline_source") != source_filter:
    with st.spinner("Loading knowledge base..."):
        pipeline = RAGPipeline(source_filter=source_filter)
        try:
            pipeline.initialize()
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_source = source_filter
        except Exception as exc:
            st.error(
                f"**Pipeline failed to initialize.**\n\n`{exc}`\n\n"
                "Run `python scripts/download_data.py` then restart."
            )
            st.stop()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _bar(score: float, width: int = 44) -> str:
    fill_w = max(2, int(score * width))
    cls = "fill-high" if score >= 0.6 else "fill-mid" if score >= 0.35 else "fill-low"
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-bg">'
        f'<div class="bar-fill {cls}" style="width:{fill_w}px"></div>'
        f'</div>'
        f'<span class="score-val">{score:.2f}</span>'
        f'</div>'
    )


def _render_chunk_cards(retrieved_chunks):
    for i, r in enumerate(retrieved_chunks, 1):
        meta    = r["chunk"]["metadata"]
        src     = meta.get("source", "?")
        text    = r["chunk"]["text"]
        preview = text[:300] + ("…" if len(text) > 300 else "")
        c_score = r["combined_score"]
        v_score = r["vector_score"]
        b_score = r["bm25_score"]
        src_cls = "election" if src == "election" else "budget"
        tag_lbl = src.upper()

        st.markdown(
            f"""<div class="chunk-card {src_cls}">
              <div class="chunk-meta">
                <span class="src-tag src-{src_cls}">{tag_lbl}</span>
                <span class="chunk-num">#{i}</span>
                <div class="scores">
                  <span class="score-label">combined</span>{_bar(c_score)}
                  <span class="score-label">vec</span>{_bar(v_score)}
                  <span class="score-label">bm25</span>{_bar(b_score)}
                </div>
              </div>
              <p class="chunk-text">{preview}</p>
            </div>""",
            unsafe_allow_html=True,
        )


def _best_score(retrieved_chunks) -> float:
    if not retrieved_chunks:
        return 0.0
    return float(retrieved_chunks[0].get("combined_score", 0.0))


def _confidence_bucket(score: float):
    if score >= 0.65:
        return "HIGH", "conf-high"
    if score >= CONFIDENCE_THRESHOLD:
        return "MEDIUM", "conf-mid"
    return "LOW", "conf-low"


def _render_confidence_badge(retrieved_chunks):
    score = _best_score(retrieved_chunks)
    label, css = _confidence_bucket(score)
    st.markdown(
        f'<span class="conf-pill {css}">Confidence: {label} · score {score:.2f} · threshold {CONFIDENCE_THRESHOLD:.2f}</span>',
        unsafe_allow_html=True,
    )


def _render_answer_with_citations(response: str, retrieved_chunks):
    if not retrieved_chunks:
        st.write(response)
        return

    top_refs = retrieved_chunks[:3]
    markers = " ".join(f"[{i}]" for i in range(1, len(top_refs) + 1))
    st.markdown(f"{response}\n\nSources: {markers}")

    refs = []
    for i, r in enumerate(top_refs, start=1):
        meta = r["chunk"]["metadata"]
        src = str(meta.get("source", "unknown")).title()
        detail = []
        if "year" in meta:
            detail.append(f"Year {meta['year']}")
        if "region" in meta:
            detail.append(str(meta["region"]))
        if "page" in meta:
            detail.append(f"Page {meta['page']}")
        detail_txt = f" ({', '.join(detail)})" if detail else ""
        refs.append(f"[{i}] {src}{detail_txt} · score {r['combined_score']:.2f}")

    st.caption("Source map: " + " | ".join(refs))


def _render_details(retrieved_chunks, prompt, pipeline_log):
    col1, col2, col3 = st.columns([5, 4, 3])

    with col1:
        with st.expander(
            f":material/article: Retrieved chunks ({len(retrieved_chunks)})",
            expanded=False,
        ):
            _render_chunk_cards(retrieved_chunks)

    with col2:
        with st.expander(":material/description: Final prompt", expanded=False):
            st.code(prompt, language="text")

    with col3:
        with st.expander(":material/monitoring: Pipeline log", expanded=False):
            log_dict = pipeline_log.to_dict()
            total_ms = log_dict["total_duration_ms"]
            st.markdown(
                f'<div class="log-total">'
                f'<div class="log-total-val">{total_ms:.0f} ms</div>'
                f'<div class="log-total-label">total latency</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for stage in log_dict["stages"]:
                st.markdown(
                    f'<div class="stage-row">'
                    f'<span class="stage-name">{stage["stage"]}</span>'
                    f' &mdash; {stage["duration_ms"]:.1f} ms'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.json(stage["data"], expanded=False)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <div class="rag-header-title">Ghana Knowledge Assistant</div>
  <div class="rag-header-sub">
    Grounded answers on Ghana's presidential election history and the 2025 national budget.
  </div>
  <div class="badge-row">
    <span class="badge">Hybrid BM25 + Vector</span>
    <span class="badge">llama-3.3-70b</span>
    <span class="badge">FAISS · 1 565 chunks</span>
    <span class="badge">Conversation memory</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Welcome screen ───────────────────────────────────────────────────────────
pending_query = st.session_state.pop("pending_query", None)

if not st.session_state.chat_history and pending_query is None:
    st.markdown("""
    <div class="welcome-wrap">
      <div class="welcome-title">What would you like to know?</div>
      <div class="welcome-sub">Select a suggestion or type your own question below</div>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        ("Election", "Who won the 2020 presidential election in Ghana?"),
        ("Election", "Which party dominated the Ashanti Region?"),
        ("Budget",   "What is the total expenditure in the 2025 budget?"),
        ("Election", "How did NDC perform in Northern Ghana in 2016?"),
    ]
    cola, colb = st.columns(2)
    for idx, (cat, text) in enumerate(suggestions):
        col = cola if idx % 2 == 0 else colb
        label = f"{'🗳' if cat == 'Election' else '📊'}  {text}"
        if col.button(label, use_container_width=True, key=f"sug_{idx}"):
            st.session_state["pending_query"] = text
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)


# ─── Chat history ─────────────────────────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message("user", avatar=":material/person:"):
        st.write(entry["query"])
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        _render_confidence_badge(entry["retrieved_chunks"])
        _render_answer_with_citations(entry["response"], entry["retrieved_chunks"])
        if not entry.get("is_relevant", True):
            st.markdown(
                '<div class="low-conf">⚠ Low-confidence retrieval — answer may be limited.</div>',
                unsafe_allow_html=True,
            )
        _render_details(entry["retrieved_chunks"], entry["prompt"], entry["pipeline_log"])


# ─── Chat input ───────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about Ghana's elections or budget...") or pending_query

if user_input:
    if not os.environ.get("GROQ_API_KEY"):
        st.error(
            "No Groq API key configured. "
            "Add `GROQ_API_KEY` to `.streamlit/secrets.toml` and restart."
        )
        st.stop()

    with st.chat_message("user", avatar=":material/person:"):
        st.write(user_input)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        with st.spinner("Retrieving context and generating response..."):
            result = st.session_state.pipeline.query(
                user_query=user_input,
                memory=st.session_state.memory,
                k=top_k,
                template_id=template_id,
            )

        _render_confidence_badge(result.retrieved_chunks)
        _render_answer_with_citations(result.response, result.retrieved_chunks)

        if not result.is_relevant:
            st.markdown(
                '<div class="low-conf">⚠ Low-confidence retrieval — answer may be limited.</div>',
                unsafe_allow_html=True,
            )

        _render_details(result.retrieved_chunks, result.prompt, result.pipeline_log)

    st.session_state.chat_history.append({
        "query":            user_input,
        "response":         result.response,
        "retrieved_chunks": result.retrieved_chunks,
        "prompt":           result.prompt,
        "pipeline_log":     result.pipeline_log,
        "is_relevant":      result.is_relevant,
    })
