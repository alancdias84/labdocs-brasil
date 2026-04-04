import streamlit as st
import re, json, pickle, hashlib, os, urllib.request
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from io import BytesIO

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LabDocs Brasil",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CSS — PASTEL THEME
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
.stApp {
    background: linear-gradient(145deg, #fdf6f0 0%, #f0f4fd 50%, #f5fdf0 100%);
    min-height: 100vh;
}
.main .block-container { padding-top: 1.5rem; max-width: 860px; }

.lab-header {
    background: white; border-radius: 16px; padding: 28px 32px;
    margin-bottom: 24px; border: 1px solid #e8e0f5;
    box-shadow: 0 2px 20px rgba(180,160,220,0.10);
}
.lab-title {
    font-family: 'Lora', serif; font-size: 2rem; font-weight: 600;
    color: #3d3560; margin: 0 0 4px 0; line-height: 1.2;
}
.lab-subtitle { color: #8c7baa; font-size: 14px; font-weight: 400; margin: 0; }
.badge-row { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
.badge {
    background: #f0ebff; color: #6b4fa0; border-radius: 100px;
    padding: 3px 12px; font-size: 11px; font-weight: 500; border: 1px solid #ddd5f5;
}

.card {
    background: white; border-radius: 14px; padding: 22px 26px;
    margin-bottom: 16px; border: 1px solid #ece8f5;
    box-shadow: 0 1px 12px rgba(160,140,210,0.08);
}

.answer-card {
    background: linear-gradient(135deg, #fdf8ff 0%, #f5f0ff 100%);
    border-radius: 14px; padding: 24px 28px; margin-bottom: 16px;
    border: 1px solid #ddd0f5; box-shadow: 0 2px 16px rgba(150,120,210,0.10);
}
.answer-meta { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
.meta-pill {
    background: white; border: 1px solid #e0d8f5; border-radius: 100px;
    padding: 3px 12px; font-size: 12px; color: #6b4fa0; font-weight: 500;
}
.meta-pill.conf-alta  { background: #f0fff4; border-color: #a8e6bf; color: #2d6a4f; }
.meta-pill.conf-media { background: #fffbf0; border-color: #f5d78e; color: #8a6914; }
.meta-pill.conf-baixa { background: #fff5f5; border-color: #f5b8b8; color: #8a2020; }
.answer-text { font-size: 15px; line-height: 1.85; color: #2d2840; white-space: pre-wrap; word-break: break-word; }

.src-card {
    background: white; border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
    border: 1px solid #ece8f5; border-left: 4px solid #b39ddb;
    box-shadow: 0 1px 6px rgba(150,120,200,0.06);
}
.src-title { font-weight: 600; font-size: 13px; color: #4a3880; margin-bottom: 4px; }
.src-meta { font-size: 11px; color: #9e8fc0; font-family: monospace; margin-bottom: 6px; }
.src-snip { font-size: 13px; color: #4a445a; line-height: 1.6; }

.conflict-alta  { border-left-color: #f5a0a0 !important; }
.conflict-media { border-left-color: #f5d78e !important; }

.stButton > button {
    background: linear-gradient(135deg, #7c5cbf 0%, #9b7dd4 100%);
    color: white; border: none; border-radius: 10px;
    font-family: 'Source Sans 3', sans-serif; font-weight: 600;
    font-size: 14px; padding: 10px 28px; transition: opacity 0.2s; width: 100%;
}
.stButton > button:hover { opacity: 0.88; }
.stButton > button:disabled { opacity: 0.45; }
.stTextInput > div > input { border-radius: 10px; border-color: #ddd5f5; font-family: 'Source Sans 3', sans-serif; }
.stTextInput > div > input:focus { border-color: #9b7dd4; box-shadow: 0 0 0 2px rgba(155,125,212,0.15); }
div[data-testid="stSidebar"] { background: white; border-right: 1px solid #ece8f5; }
.stSlider > div > div > div { background: #9b7dd4; }
.stSelectbox > div > div { border-radius: 10px; border-color: #ddd5f5; cursor: pointer; }
.stSelectbox > div > div > div { cursor: pointer; }
.stSelectbox span { cursor: pointer; }
[data-baseweb="select"] * { cursor: pointer !important; }
.stSuccess { background: #f0fff4; border-radius: 10px; }
.stWarning { background: #fffbf0; border-radius: 10px; }
.stError   { background: #fff5f5; border-radius: 10px; }
hr { border-color: #ece8f5; margin: 1rem 0; }
.stChatInput > div { border-radius: 14px; border-color: #ddd5f5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# API KEY — lida dos Secrets do Streamlit (sem expor ao usuário)
# ─────────────────────────────────────────────────────────────────────
api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────
@dataclass
class DocMeta:
    doc_id: str; filename: str; file_type: str
    title: str=''; version: str='N/A'; date: str='N/A'
    sector: str='N/A'; page_count: int=0; char_count: int=0
    indexed_at: str=field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Chunk:
    chunk_id: str; doc_id: str; filename: str
    section: str; text: str; page: int; position: int

# ─────────────────────────────────────────────────────────────────────
# NLP
# ─────────────────────────────────────────────────────────────────────
STOPWORDS_PT = set([
    'a','ao','aos','as','até','com','como','da','das','de','dela','dele',
    'do','dos','e','ela','ele','em','entre','era','essa','esse','esta','este',
    'eu','foi','há','isso','isto','já','mais','mas','me','muito','na','nas',
    'nem','no','nos','num','o','os','ou','para','pela','pelo','por','qual',
    'quando','que','quem','se','seu','seus','si','sobre','sua','suas','também',
    'tem','tudo','um','uma','uns','você','é','à','ser','são','está','pode',
    'não','sendo','assim','cada','onde','pois','todo','toda','todos','todas',
])

def tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 1 and t not in STOPWORDS_PT]

def get_version(t):
    m = re.search(r'[Vv]ers[ãa]o\s*[:\-]?\s*([\d\.]+)|\bRev\.?\s*([\d\.]+)', t[:2000])
    return (m.group(1) or m.group(2)) if m else 'N/A'

def get_date(t):
    m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})|(\d{4}[\/\-]\d{2}[\/\-]\d{2})', t[:2000])
    return m.group(0) if m else 'N/A'

def make_id(name): return hashlib.md5(name.encode()).hexdigest()[:12]

# ─────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────
def extract_pdf(data: bytes):
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t: pages.append((i + 1, t))
        if pages: return pages, len(pages)
    except: pass
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(data))
    pages = [(i+1, p.extract_text() or '') for i, p in enumerate(reader.pages) if p.extract_text()]
    return pages, len(reader.pages)

def extract_text(data: bytes):
    text = data.decode('utf-8', errors='replace')
    text = re.sub(r'^---[\s\S]+?---\n', '', text)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.M)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    return text

# ─────────────────────────────────────────────────────────────────────
# CHUNKER
# ─────────────────────────────────────────────────────────────────────
SECTION_RE = re.compile(
    r'^(#{1,4}\s+.{3,80}|(?:capítulo|seção|item)\s+[\d\.]+.{0,60}|\d+[\.]\s+[A-ZÁÉÍÓÚ].{5,60})',
    re.I | re.M
)

def chunk_pdf(pages_text, doc_id, filename, chunk_words=200, overlap=40):
    chunks, gid = [], 0
    for page_num, text in pages_text:
        text = re.sub(r'[ \t]+', ' ', text)
        sections, last_h, last_p = [], 'Início', 0
        for m in SECTION_RE.finditer(text):
            if m.start() > last_p: sections.append((last_h, text[last_p:m.start()]))
            last_h = m.group(0).strip()[:60]; last_p = m.end()
        sections.append((last_h, text[last_p:]))
        for sec, sec_text in sections:
            paras = [p.strip() for p in re.split(r'\n\n+', sec_text) if len(p.strip()) > 30]
            buf = []
            for para in paras:
                words = para.split()
                if len(buf) + len(words) > chunk_words and buf:
                    txt = ' '.join(buf)
                    if len(txt) > 50:
                        chunks.append(Chunk(f'{doc_id}-{gid:04d}', doc_id, filename, sec, txt, page_num, gid)); gid += 1
                    buf = buf[-overlap:]
                buf.extend(words)
            if len(' '.join(buf)) > 50:
                chunks.append(Chunk(f'{doc_id}-{gid:04d}', doc_id, filename, sec, ' '.join(buf), page_num, gid)); gid += 1
    return chunks

def chunk_plain(text, doc_id, filename, chunk_words=200, overlap=40):
    sections, last_h, last_p = [], 'Início', 0
    for m in SECTION_RE.finditer(text):
        if m.start() > last_p: sections.append((last_h, text[last_p:m.start()]))
        last_h = m.group(0).strip()[:60]; last_p = m.end()
    sections.append((last_h, text[last_p:]))
    chunks, gid = [], 0
    for sec, sec_text in sections:
        paras = [p.strip() for p in re.split(r'\n\n+', sec_text) if len(p.strip()) > 30]
        buf = []
        for para in paras:
            words = para.split()
            if len(buf) + len(words) > chunk_words and buf:
                txt = ' '.join(buf)
                if len(txt) > 50:
                    chunks.append(Chunk(f'{doc_id}-{gid:04d}', doc_id, filename, sec, txt, 0, gid)); gid += 1
                buf = buf[-overlap:]
            buf.extend(words)
        if len(' '.join(buf)) > 50:
            chunks.append(Chunk(f'{doc_id}-{gid:04d}', doc_id, filename, sec, ' '.join(buf), 0, gid)); gid += 1
    return chunks

# ─────────────────────────────────────────────────────────────────────
# INDEX BUILDER — lê documentos da pasta documents/ no repositório
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_index_from_folder():
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import faiss

    docs_path = Path("documents")
    SUPPORTED = {'.pdf', '.rmd', '.md', '.txt'}
    file_contents = []
    for f in sorted(docs_path.glob("*")):
        if f.suffix.lower() in SUPPORTED:
            file_contents.append((f.name, f.read_bytes()))

    if not file_contents:
        return None

    all_meta, all_chunks = [], []
    for fname, data in file_contents:
        ext = Path(fname).suffix.lower()
        doc_id = make_id(fname)
        if ext == '.pdf':
            pages_text, page_count = extract_pdf(data)
            raw = re.sub(r'\n{3,}', '\n\n', '\n\n'.join(t for _, t in pages_text)).strip()
            chunks = chunk_pdf(pages_text, doc_id, fname)
        else:
            raw = extract_text(data)
            raw = re.sub(r'[ \t]+', ' ', raw); raw = re.sub(r'\n{3,}', '\n\n', raw).strip()
            chunks = chunk_plain(raw, doc_id, fname)
            page_count = 0
        all_meta.append(DocMeta(
            doc_id=doc_id, filename=fname,
            file_type=ext.lstrip('.').upper(), title=Path(fname).stem.replace('_', ' '),
            version=get_version(raw), date=get_date(raw),
            page_count=page_count, char_count=len(raw)
        ))
        all_chunks.extend(chunks)

    tok_corpus = [tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tok_corpus)

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    embeddings = model.encode(
        [c.text for c in all_chunks], batch_size=32,
        show_progress_bar=False, normalize_embeddings=True
    ).astype(np.float32)

    dim = embeddings.shape[1]
    faiss_idx = faiss.IndexFlatIP(dim)
    faiss_idx.add(embeddings)

    return {
        'chunks':     all_chunks,
        'meta':       {m.doc_id: m for m in all_meta},
        'bm25':       bm25,
        'faiss':      faiss_idx,
        'embeddings': embeddings,
        'model':      model,
        'filenames':  [f for f, _ in file_contents],
    }

# ─────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────
def retrieve(query, idx, top_k=4, doc_filter=None):
    tokens = tokenize(query)
    bm25_sc = idx['bm25'].get_scores(tokens) if tokens else np.zeros(len(idx['chunks']))
    bm25_rank = {int(i): r for r, i in enumerate(np.argsort(bm25_sc)[::-1][:50])}
    q_vec = idx['model'].encode([query], normalize_embeddings=True).astype(np.float32)
    sc, ids = idx['faiss'].search(q_vec, 50)
    sem_rank = {int(ids[0][i]): i for i in range(len(ids[0])) if ids[0][i] >= 0}
    sem_sc   = {int(ids[0][i]): float(sc[0][i]) for i in range(len(ids[0])) if ids[0][i] >= 0}
    all_ids = set(bm25_rank) | set(sem_rank)
    rrf = {cid: 1/(60+bm25_rank.get(cid,50)) + 1/(60+sem_rank.get(cid,50)) for cid in all_ids}
    top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    mx = top[0][1] if top else 1
    results = []
    for cid, score in top:
        c = idx['chunks'][cid]
        if doc_filter and doc_filter != 'Todos' and c.filename != doc_filter: continue
        results.append({'chunk': c, 'rrf': score, 'rel': score/mx,
                        'sem': sem_sc.get(cid,0), 'bm25': float(bm25_sc[cid])})
    return results

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def best_snippet(text, query, max_w=100):
    sents = re.split(r'(?<=[.!?\n])\s+', text)
    if len(sents) <= 2: return ' '.join(text.split()[:max_w])
    qw = set(query.lower().split())
    scores = [len(set(s.lower().split()) & qw) for s in sents]
    bi = max(range(len(scores)), key=lambda i: scores[i])
    start, end, wc = bi, bi, len(sents[bi].split())
    while wc < max_w:
        cl, cr = start > 0, end < len(sents)-1
        if not cl and not cr: break
        lw = len(sents[start-1].split()) if cl else 9999
        rw = len(sents[end+1].split())   if cr else 9999
        if cl and lw <= rw:
            if wc+lw > max_w: break
            start -= 1; wc += lw
        elif cr:
            if wc+rw > max_w: break
            end += 1; wc += rw
        else: break
    return ' '.join(sents[start:end+1])

NUM_RE   = re.compile(r'\b(\d+[\.,]?\d*)\s*(%|mg|mL|g|dL|µL|UI|mmol|nmol|h|min|°C|dias?|horas?)\b', re.I)
PRESC_RE = re.compile(r'\b(deve|deverá|não deve|obrigatório|proibido|shall|must|required)\b', re.I)

def detect_conflicts(results, embeddings):
    conflicts = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            ca, cb = results[i]['chunk'], results[j]['chunk']
            if ca.doc_id == cb.doc_id: continue
            sim = float(np.dot(embeddings[ca.position], embeddings[cb.position]))
            if sim < 0.82: continue
            na = set(m.group(0) for m in NUM_RE.finditer(ca.text))
            nb = set(m.group(0) for m in NUM_RE.finditer(cb.text))
            if (na-nb) and (nb-na):
                conflicts.append({'sev':'Alta','desc':f'Divergência numérica entre "{ca.filename}" e "{cb.filename}"'})
            elif PRESC_RE.search(ca.text) and PRESC_RE.search(cb.text):
                conflicts.append({'sev':'Média','desc':f'Orientações conflitantes entre "{ca.filename}" e "{cb.filename}"'})
    return conflicts

def generate(query, results, meta_map, model_id):
    parts = []
    for i, r in enumerate(results, 1):
        c = r['chunk']
        m = meta_map.get(c.doc_id)
        snip = best_snippet(c.text, query)
        page_ref = f" | p. {c.page}" if c.page > 0 else ''
        v = m.version if m else 'N/A'; d = m.date if m else 'N/A'
        parts.append(f"[{i}] {c.filename} | v{v} | {d}{page_ref} | {c.section[:50]}\n{snip}")
    ctx = '\n---\n'.join(parts)
    prompt = (
        "Assistente de medicina laboratorial. Responda em português com base nos trechos abaixo.\n\n"
        "INSTRUÇÕES:\n"
        "1. Se o termo exato estiver nos trechos, responda diretamente e cite a fonte como [n].\n"
        "2. Se o termo exato NÃO estiver, busque nos trechos conceitos relacionados que possam "
        "contextualizar a pergunta e apresente-os como informação de contexto, citando a fonte como [n].\n"
        "3. Apenas diga 'não encontrado' se absolutamente nenhuma informação relacionada existir nos trechos.\n\n"
        f"TRECHOS:\n{ctx}\n\nPERGUNTA: {query}"
    )
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
    text = resp['choices'][0]['message']['content']
    return text, len(prompt.split()), len(text.split())

# ─────────────────────────────────────────────────────────────────────
# INICIALIZAÇÃO — indexa documentos automaticamente ao carregar
# ─────────────────────────────────────────────────────────────────────
if 'index'   not in st.session_state: st.session_state.index   = None
if 'history' not in st.session_state: st.session_state.history = []

if st.session_state.index is None:
    with st.spinner("Carregando e indexando documentos... (pode levar 1-2 min na primeira vez)"):
        st.session_state.index = build_index_from_folder()

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    top_k = st.slider("Top-K fontes", 2, 8, 4)
    model_id = st.selectbox("Modelo (gratuito)", [
        "deepseek/deepseek-chat-v3.1:free",
        "meta-llama/llama-4-maverick:free",
        "qwen/qwen3.6-plus:free",
        "deepseek/deepseek-r1:free",
        "meta-llama/llama-3.1-8b-instruct:free",
    ])

    if st.session_state.index:
        st.markdown("---")
        idx = st.session_state.index
        docs = sorted(set(c.filename for c in idx['chunks']))
        doc_filter = st.selectbox("Filtrar por documento", ['Todos'] + docs)
        st.caption(f"{len(idx['chunks'])} chunks · {len(docs)} documento(s)")
        st.markdown("---")
        st.markdown("**Documentos indexados:**")
        for fname in idx.get('filenames', docs):
            st.caption(f"📄 {fname}")
        if st.button("↩ Limpar histórico"):
            st.session_state.history = []
            st.rerun()
    else:
        doc_filter = 'Todos'

# ─────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lab-header">
    <div class="lab-title">🧪 LabDocs Brasil</div>
    <div class="lab-subtitle">Desafio SBPC/ML 2026–2027 · RAG + LLM para Medicina Laboratorial</div>
    <div class="badge-row">
        <span class="badge">BM25 + Semântico</span>
        <span class="badge">Rastreabilidade ISO 15189</span>
        <span class="badge">Detecção de Conflitos</span>
        <span class="badge">PDF · RMD · MD · TXT</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.index:
    st.error("⚠️ Nenhum documento encontrado na pasta `documents/`. Adicione arquivos PDF, MD ou TXT ao repositório.")
    st.stop()

idx = st.session_state.index

# Chat history
for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h['q'])
    with st.chat_message("assistant", avatar="🧪"):
        conf_class = {'Alta':'conf-alta','Média':'conf-media','Baixa':'conf-baixa'}[h['conf_label']]
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-meta">
                <span class="meta-pill {conf_class}">Confiança: {h['conf_label']} ({h['conf']:.0%})</span>
                <span class="meta-pill">📥 {h['tok_in']} tokens</span>
                <span class="meta-pill">📤 {h['tok_out']} tokens</span>
                <span class="meta-pill">gratuito</span>
            </div>
            <div class="answer-text">{h['answer']}</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander(f"📄 {len(h['results'])} fonte(s)"):
            for i, r in enumerate(h['results'], 1):
                c = r['chunk']
                m = idx['meta'].get(c.doc_id)
                v = m.version if m else 'N/A'; d = m.date if m else 'N/A'
                page_str = f"p. {c.page} · " if c.page > 0 else ''
                snip = best_snippet(c.text, h['q'], 150)
                st.markdown(f"""
                <div class="src-card">
                    <div class="src-title">[{i}] {c.filename}</div>
                    <div class="src-meta">v{v} · {d} · {page_str}{c.section[:55]} · {int(r['rel']*100)}% relevância</div>
                    <div class="src-snip">{snip}…</div>
                </div>""", unsafe_allow_html=True)
        if h['conflicts']:
            with st.expander(f"⚠️ {len(h['conflicts'])} conflito(s) detectado(s)"):
                for conf in h['conflicts']:
                    cls = 'conflict-alta' if conf['sev']=='Alta' else 'conflict-media'
                    st.markdown(f'<div class="src-card {cls}"><b>{conf["sev"]}</b> · {conf["desc"]}</div>',
                                unsafe_allow_html=True)
        else:
            st.success("✓ Nenhum conflito detectado")

# Chat input
query = st.chat_input("Faça uma pergunta sobre os documentos...")
if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant", avatar="🧪"):
        with st.spinner("Buscando fontes e gerando resposta..."):
            try:
                results = retrieve(query, idx, top_k=top_k,
                                   doc_filter=doc_filter if doc_filter != 'Todos' else None)
                if not results:
                    st.warning("Nenhuma fonte relevante encontrada. Tente reformular.")
                    st.stop()
                conflicts = detect_conflicts(results, idx['embeddings'])
                answer, tok_in, tok_out = generate(query, results, idx['meta'], model_id)
                avg_rel = sum(r['rel'] for r in results) / len(results)
                n_docs  = len(set(r['chunk'].doc_id for r in results))
                conf = min(1.0, avg_rel*0.6 + n_docs/3*0.3 - len([c for c in conflicts if c['sev']=='Alta'])*0.15)
                conf_label = 'Alta' if conf >= 0.65 else 'Média' if conf >= 0.35 else 'Baixa'
                conf_class = {'Alta':'conf-alta','Média':'conf-media','Baixa':'conf-baixa'}[conf_label]

                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-meta">
                        <span class="meta-pill {conf_class}">Confiança: {conf_label} ({conf:.0%})</span>
                        <span class="meta-pill">📥 {tok_in} tokens</span>
                        <span class="meta-pill">📤 {tok_out} tokens</span>
                        <span class="meta-pill">gratuito</span>
                    </div>
                    <div class="answer-text">{answer}</div>
                </div>""", unsafe_allow_html=True)

                with st.expander(f"📄 {len(results)} fonte(s) recuperada(s)"):
                    for i, r in enumerate(results, 1):
                        c = r['chunk']; m = idx['meta'].get(c.doc_id)
                        v = m.version if m else 'N/A'; d = m.date if m else 'N/A'
                        page_str = f"p. {c.page} · " if c.page > 0 else ''
                        snip = best_snippet(c.text, query, 150)
                        st.markdown(f"""
                        <div class="src-card">
                            <div class="src-title">[{i}] {c.filename}</div>
                            <div class="src-meta">v{v} · {d} · {page_str}{c.section[:55]} · {int(r['rel']*100)}% relevância</div>
                            <div class="src-snip">{snip}…</div>
                        </div>""", unsafe_allow_html=True)

                if conflicts:
                    with st.expander(f"⚠️ {len(conflicts)} conflito(s) detectado(s)"):
                        for conf_item in conflicts:
                            cls = 'conflict-alta' if conf_item['sev']=='Alta' else 'conflict-media'
                            st.markdown(f'<div class="src-card {cls}"><b>{conf_item["sev"]}</b> · {conf_item["desc"]}</div>',
                                        unsafe_allow_html=True)
                else:
                    st.success("✓ Nenhum conflito detectado")

                st.session_state.history.append({
                    'q': query, 'answer': answer, 'results': results,
                    'conflicts': conflicts, 'conf': conf, 'conf_label': conf_label,
                    'tok_in': tok_in, 'tok_out': tok_out
                })
            except Exception as e:
                err = str(e)
                if '429' in err:
                    st.warning(
                        "⚠️ **Limite de requisições atingido** para este modelo.\n\n"
                        "O modelo gratuito atingiu o limite de uso da OpenRouter. "
                        "**Troque o modelo na barra lateral** (ex: `meta-llama/llama-3.1-8b-instruct:free`) "
                        "e tente novamente. Os limites são resetados automaticamente após alguns minutos."
                    )
                elif '404' in err:
                    st.warning(
                        "⚠️ **Modelo não encontrado.**\n\n"
                        "O modelo selecionado não está disponível na OpenRouter no momento. "
                        "**Troque o modelo na barra lateral** e tente novamente."
                    )
                else:
                    st.error(f"Erro inesperado: {e}")
