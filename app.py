import streamlit as st
import re, json, pickle, hashlib, os, urllib.request
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from io import BytesIO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LabDocs Brasil",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0d1117; color: #e6edf3; }
  .main .block-container { padding-top: 2rem; max-width: 800px; }
  h1 { font-size: 2.2rem !important; color: #ffffff !important; }
  .stButton > button {
    background: #00e5ff; color: #000; font-weight: 700;
    border: none; border-radius: 4px; padding: 10px 28px;
    font-size: 14px; letter-spacing: 1px; width: 100%;
  }
  .stButton > button:hover { background: #00b8cc; color: #000; }
  .stTextInput > div > input {
    background: #161b22; border: 1px solid #30363d;
    color: #e6edf3; border-radius: 4px;
  }
  .stFileUploader { background: #161b22; border-radius: 4px; }
  .answer-box {
    background: #161b22; border: 1px solid #30363d;
    border-left: 4px solid #00e5ff; border-radius: 6px;
    padding: 20px; margin: 16px 0; white-space: pre-wrap;
    line-height: 1.8; color: #e6edf3; font-size: 15px;
  }
  .src-box {
    background: #161b22; border: 1px solid #21262d;
    border-left: 4px solid #ffd166; border-radius: 6px;
    padding: 12px 16px; margin: 6px 0;
  }
  .src-title { color: #58a6ff; font-weight: 600; font-size: 13px; }
  .src-meta  { color: #6e7681; font-size: 11px; font-family: monospace; margin: 4px 0; }
  .src-snip  { color: #c9d1d9; font-size: 13px; line-height: 1.6; }
  .conf-alta  { color: #2ecc71; font-weight: 700; }
  .conf-media { color: #ffd166; font-weight: 700; }
  .conf-baixa { color: #ff4757; font-weight: 700; }
  .step-badge {
    background: #161b22; border: 1px solid #30363d; border-radius: 4px;
    padding: 6px 14px; display: inline-block; font-size: 12px;
    color: #8b949e; margin-bottom: 8px; font-family: monospace;
  }
  .divider { border-top: 1px solid #21262d; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class DocMeta:
    doc_id: str; filename: str; file_type: str
    title: str = ''; version: str = 'N/A'; date: str = 'N/A'
    sector: str = 'N/A'; char_count: int = 0
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Chunk:
    chunk_id: str; doc_id: str; filename: str
    section: str; text: str; position: int

# ── NLP ───────────────────────────────────────────────────────────────────────
STOPWORDS = set([
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
    return [t for t in text.split() if len(t) > 1 and t not in STOPWORDS]

def get_version(t):
    m = re.search(r'[Vv]ers[ãa]o\s*[:\-]?\s*([\d\.]+)|\bRev\.?\s*([\d\.]+)', t[:2000])
    return (m.group(1) or m.group(2)) if m else 'N/A'

def get_date(t):
    m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})|(\d{4}[\/\-]\d{2}[\/\-]\d{2})', t[:2000])
    return m.group(0) if m else 'N/A'

def make_id(name): return hashlib.md5(name.encode()).hexdigest()[:12]

# ── Text extraction ───────────────────────────────────────────────────────────
def extract_pdf(data: bytes) -> str:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: pages.append(t)
        if pages: return '\n\n'.join(pages)
    except: pass
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(data))
    return '\n\n'.join(p.extract_text() or '' for p in reader.pages)

def extract_text(data: bytes) -> str:
    text = data.decode('utf-8', errors='replace')
    text = re.sub(r'^---[\s\S]+?---\n', '', text)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.M)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    return text

# ── Chunker ───────────────────────────────────────────────────────────────────
SECTION_RE = re.compile(
    r'^(#{1,4}\s+.{3,80}|(?:capítulo|seção|item)\s+[\d\.]+.{0,60}|\d+[\.]\s+[A-ZÁÉÍÓÚ].{5,60})',
    re.I | re.M
)

def chunk_text(text, doc_id, filename, chunk_words=200, overlap=40):
    sections, last_h, last_p = [], 'Início', 0
    for m in SECTION_RE.finditer(text):
        if m.start() > last_p:
            sections.append((last_h, text[last_p:m.start()]))
        last_h = m.group(0).strip()[:60]
        last_p = m.end()
    sections.append((last_h, text[last_p:]))

    chunks, gid = [], 0
    for sec_title, sec_text in sections:
        paras = [p.strip() for p in re.split(r'\n\n+', sec_text) if len(p.strip()) > 30]
        buf = []
        for para in paras:
            words = para.split()
            if len(buf) + len(words) > chunk_words and buf:
                txt = ' '.join(buf)
                if len(txt) > 50:
                    chunks.append(Chunk(f'{doc_id}-{gid:04d}', doc_id, filename, sec_title, txt, gid))
                    gid += 1
                buf = buf[-overlap:]
            buf.extend(words)
        if len(' '.join(buf)) > 50:
            chunks.append(Chunk(f'{doc_id}-{gid:04d}', doc_id, filename, sec_title, ' '.join(buf), gid))
            gid += 1
    return chunks

# ── Index builder ─────────────────────────────────────────────────────────────
def build_index(uploaded_files):
    from rank_bm25 import BM25Okapi

    all_meta, all_chunks = [], []

    for uf in uploaded_files:
        data = uf.read()
        ext  = Path(uf.name).suffix.lower()
        raw  = extract_pdf(data) if ext == '.pdf' else extract_text(data)
        raw  = re.sub(r'[ \t]+', ' ', raw)
        raw  = re.sub(r'\n{3,}', '\n\n', raw).strip()
        doc_id = make_id(uf.name)

        meta = DocMeta(
            doc_id=doc_id, filename=uf.name,
            file_type=ext.lstrip('.').upper(),
            title=Path(uf.name).stem.replace('_', ' '),
            version=get_version(raw), date=get_date(raw),
            char_count=len(raw),
        )
        chunks = chunk_text(raw, doc_id, uf.name)
        all_meta.append(meta)
        all_chunks.extend(chunks)

    # BM25
    tok_corpus = [tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tok_corpus)

    # Embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    texts = [c.text for c in all_chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False,
                              normalize_embeddings=True).astype(np.float32)

    # FAISS
    import faiss
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
    }

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query, idx, top_k=4):
    tokens  = tokenize(query)
    bm25_sc = idx['bm25'].get_scores(tokens) if tokens else np.zeros(len(idx['chunks']))
    bm25_rank = {int(i): r for r, i in enumerate(np.argsort(bm25_sc)[::-1][:50])}

    q_vec = idx['model'].encode([query], normalize_embeddings=True).astype(np.float32)
    sc, ids = idx['faiss'].search(q_vec, 50)
    sem_rank  = {int(ids[0][i]): i for i in range(len(ids[0])) if ids[0][i] >= 0}
    sem_sc    = {int(ids[0][i]): float(sc[0][i]) for i in range(len(ids[0])) if ids[0][i] >= 0}

    all_ids = set(bm25_rank) | set(sem_rank)
    rrf = {cid: 1/(60+bm25_rank.get(cid,50)) + 1/(60+sem_rank.get(cid,50)) for cid in all_ids}
    top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    mx  = top[0][1] if top else 1

    return [{'chunk': idx['chunks'][cid], 'rrf': score, 'rel': score/mx,
             'sem': sem_sc.get(cid,0), 'bm25': float(bm25_sc[cid])}
            for cid, score in top]

# ── Snippet ───────────────────────────────────────────────────────────────────
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

# ── Conflict detection ────────────────────────────────────────────────────────
NUM_RE   = re.compile(r'\b(\d+[\.,]?\d*)\s*(%|mg|mL|g|dL|µL|UI|mmol|nmol|h|min|°C|dias?|horas?)\b', re.I)
PRESC_RE = re.compile(r'\b(deve|deverá|não deve|obrigatório|proibido|shall|must|required)\b', re.I)

def detect_conflicts(results, embeddings):
    conflicts = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            ca, cb = results[i]['chunk'], results[j]['chunk']
            if ca.doc_id == cb.doc_id: continue
            ea = embeddings[ca.position].astype(np.float32)
            eb = embeddings[cb.position].astype(np.float32)
            sim = float(np.dot(ea, eb))
            if sim < 0.82: continue
            na = set(m.group(0) for m in NUM_RE.finditer(ca.text))
            nb = set(m.group(0) for m in NUM_RE.finditer(cb.text))
            if (na-nb) and (nb-na):
                conflicts.append({'sev':'Alta', 'desc': f'Valores divergentes entre "{ca.filename}" e "{cb.filename}"'})
            elif PRESC_RE.search(ca.text) and PRESC_RE.search(cb.text):
                conflicts.append({'sev':'Média', 'desc': f'Orientações prescritivas conflitantes entre "{ca.filename}" e "{cb.filename}"'})
    return conflicts

# ── Generate ──────────────────────────────────────────────────────────────────
def generate_answer(query, results, meta_map, api_key, model_id):
    parts = []
    for i, r in enumerate(results, 1):
        c = r['chunk']
        m = meta_map.get(c.doc_id)
        snip = best_snippet(c.text, query)
        v = m.version if m else 'N/A'
        d = m.date    if m else 'N/A'
        parts.append(f"[{i}] {c.filename} | v{v} | {d} | {c.section[:50]}\n{snip}")

    ctx    = '\n---\n'.join(parts)
    prompt = (
        "Assistente de medicina laboratorial. Responda em português baseado EXCLUSIVAMENTE "
        "nos trechos abaixo. Cite a fonte como [n]. Se não encontrar, diga explicitamente.\n\n"
        f"TRECHOS:\n{ctx}\n\nPERGUNTA: {query}"
    )
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
    text = resp['choices'][0]['message']['content']
    return text, len(prompt.split()), len(text.split())

# ════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════
if 'screen' not in st.session_state:
    st.session_state.screen   = 'start'   # start | setup | chat
if 'index'  not in st.session_state:
    st.session_state.index    = None
if 'api_key' not in st.session_state:
    st.session_state.api_key  = ''
if 'model_id' not in st.session_state:
    st.session_state.model_id = 'qwen/qwen3-30b-a3b:free'
if 'history' not in st.session_state:
    st.session_state.history  = []

# ════════════════════════════════════════════════════════
# SCREEN: START
# ════════════════════════════════════════════════════════
if st.session_state.screen == 'start':
    st.markdown("<br><br>", unsafe_allow_html=True)
    col = st.columns([1, 3, 1])[1]
    with col:
        st.markdown("""
        <div style="text-align:center">
          <div style="font-family:monospace;font-size:11px;color:#00e5ff;letter-spacing:2px;margin-bottom:16px">
            DESAFIO SBPC/ML 2026–2027
          </div>
          <h1 style="font-size:3rem;font-weight:800;color:#fff;line-height:1.1;margin-bottom:8px">
            🧪 LabDocs Brasil
          </h1>
          <p style="color:#8b949e;font-size:15px;margin-bottom:40px;line-height:1.7">
            Inteligência Artificial para documentos laboratoriais.<br>
            BM25 + Embeddings semânticos + Detecção de conflitos.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:40px">
          <span style="background:#161b22;border:1px solid #30363d;border-radius:100px;padding:4px 14px;font-family:monospace;font-size:11px;color:#8b949e">PDF · RMD · MD · TXT</span>
          <span style="background:#161b22;border:1px solid #30363d;border-radius:100px;padding:4px 14px;font-family:monospace;font-size:11px;color:#8b949e">BM25 + Semântico</span>
          <span style="background:#161b22;border:1px solid #30363d;border-radius:100px;padding:4px 14px;font-family:monospace;font-size:11px;color:#8b949e">Rastreabilidade ISO 15189</span>
          <span style="background:#161b22;border:1px solid #30363d;border-radius:100px;padding:4px 14px;font-family:monospace;font-size:11px;color:#8b949e">Detecção de Conflitos</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶  INICIAR", key="btn_start"):
            st.session_state.screen = 'setup'
            st.rerun()

# ════════════════════════════════════════════════════════
# SCREEN: SETUP
# ════════════════════════════════════════════════════════
elif st.session_state.screen == 'setup':
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("### Configuração")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── API Key ──
        st.markdown("<div class='step-badge'>PASSO 1 DE 2 · API KEY</div>", unsafe_allow_html=True)
        st.markdown("**OpenRouter API Key**")
        st.caption("Acesse openrouter.ai → Keys → Create API Key (gratuito, sem cartão)")
        api_key = st.text_input("Cole sua chave aqui", type="password",
                                placeholder="sk-or-v1-...",
                                value=st.session_state.api_key)
        if api_key:
            st.session_state.api_key = api_key
            if api_key.startswith('sk-or-'):
                st.success("✓ Chave com formato válido")
            else:
                st.warning("Formato inválido — deve começar com sk-or-")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Upload ──
        st.markdown("<div class='step-badge'>PASSO 2 DE 2 · DOCUMENTOS</div>", unsafe_allow_html=True)
        st.markdown("**Selecione seus documentos**")
        uploaded = st.file_uploader(
            "PDF, RMD, MD, TXT",
            type=['pdf','rmd','md','txt'],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            st.caption(f"{len(uploaded)} arquivo(s) selecionado(s)")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Build ──
        can_build = bool(api_key) and bool(uploaded)
        if st.button("⚡  INDEXAR E INICIAR", disabled=not can_build, key="btn_build"):
            with st.spinner("Indexando documentos... (pode levar 1-2 minutos na primeira vez)"):
                try:
                    st.session_state.index = build_index(uploaded)
                    st.session_state.screen = 'chat'
                    st.session_state.history = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")

        if not can_build:
            st.caption("⚠ Preencha a API key e selecione ao menos um documento para continuar.")

# ════════════════════════════════════════════════════════
# SCREEN: CHAT
# ════════════════════════════════════════════════════════
elif st.session_state.screen == 'chat':
    idx = st.session_state.index

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configurações")
        top_k = st.slider("Top-K fontes", 2, 8, 4)
        model_id = st.selectbox("Modelo (OpenRouter free)", [
            "qwen/qwen3-30b-a3b:free",
            "qwen/qwen3.6-plus:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ])
        st.session_state.model_id = model_id
        st.divider()
        docs = list(set(c.filename for c in idx['chunks']))
        st.markdown(f"**{len(idx['chunks'])} chunks** · {len(docs)} doc(s)")
        for d in docs:
            st.caption(f"📄 {d}")
        st.divider()
        if st.button("↩ Voltar ao início"):
            st.session_state.screen = 'start'
            st.session_state.index  = None
            st.session_state.history = []
            st.rerun()

    # Header
    st.markdown("""
    <div style="margin-bottom:24px">
      <div style="font-family:monospace;font-size:11px;color:#00e5ff;letter-spacing:2px;margin-bottom:4px">
        LABDOCS BRASIL · INTERFACE DE CONSULTA
      </div>
      <h2 style="color:#fff;font-weight:700;margin:0">Faça sua pergunta</h2>
    </div>
    """, unsafe_allow_html=True)

    # History
    for h in st.session_state.history:
        with st.chat_message("user"):
            st.write(h['q'])
        with st.chat_message("assistant"):
            st.markdown(f'<div class="answer-box">{h["a"]}</div>', unsafe_allow_html=True)
            for i, src in enumerate(h['sources'], 1):
                c   = src['chunk']
                m   = idx['meta'].get(c.doc_id)
                v   = m.version if m else 'N/A'
                d   = m.date    if m else 'N/A'
                pct = int(src['rel'] * 100)
                snip = best_snippet(c.text, h['q'], 120)
                st.markdown(f"""
                <div class="src-box">
                  <div class="src-title">[{i}] {c.filename}</div>
                  <div class="src-meta">v{v} &nbsp;|&nbsp; {d} &nbsp;|&nbsp; {c.section[:55]} &nbsp;|&nbsp; {pct}% relevância</div>
                  <div class="src-snip">{snip}…</div>
                </div>""", unsafe_allow_html=True)

    # Input
    query = st.chat_input("Ex: Qual o critério de rejeição de amostra para potássio?")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Buscando fontes e gerando resposta..."):
                try:
                    results   = retrieve(query, idx, top_k=top_k)
                    conflicts = detect_conflicts(results, idx['embeddings'])
                    answer, tok_in, tok_out = generate_answer(
                        query, results, idx['meta'],
                        st.session_state.api_key,
                        st.session_state.model_id,
                    )

                    # Confidence
                    avg_rel = sum(r['rel'] for r in results) / len(results)
                    n_docs  = len(set(r['chunk'].doc_id for r in results))
                    conf = min(1.0, avg_rel * 0.6 + n_docs/3 * 0.3 - len([c for c in conflicts if c['sev']=='Alta'])*0.15)
                    conf_label = 'Alta' if conf >= 0.65 else 'Média' if conf >= 0.35 else 'Baixa'
                    conf_class = {'Alta':'conf-alta','Média':'conf-media','Baixa':'conf-baixa'}[conf_label]

                    st.markdown(
                        f'<span class="{conf_class}">Confiança: {conf_label} ({conf:.0%})</span>'
                        f' &nbsp;·&nbsp; <span style="color:#6e7681;font-size:12px;font-family:monospace">'
                        f'entrada {tok_in} tokens · saída {tok_out} tokens · gratuito</span>',
                        unsafe_allow_html=True
                    )

                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                    st.markdown(f"**📄 {len(results)} fonte(s) recuperada(s)**")
                    for i, r in enumerate(results, 1):
                        c   = r['chunk']
                        m   = idx['meta'].get(c.doc_id)
                        v   = m.version if m else 'N/A'
                        d   = m.date    if m else 'N/A'
                        pct = int(r['rel'] * 100)
                        snip = best_snippet(c.text, query, 120)
                        st.markdown(f"""
                        <div class="src-box">
                          <div class="src-title">[{i}] {c.filename}</div>
                          <div class="src-meta">v{v} &nbsp;|&nbsp; {d} &nbsp;|&nbsp; {c.section[:55]} &nbsp;|&nbsp; {pct}% relevância</div>
                          <div class="src-snip">{snip}…</div>
                        </div>""", unsafe_allow_html=True)

                    if conflicts:
                        st.markdown("**⚠️ Conflitos detectados**")
                        for c in conflicts:
                            col_c = {'Alta':'#ff4757','Média':'#ffd166'}[c['sev']]
                            st.markdown(
                                f'<div style="border-left:3px solid {col_c};padding:8px 12px;'
                                f'background:#161b22;border-radius:4px;font-size:12px;color:#c9d1d9;margin:4px 0">'
                                f'{c["desc"]}</div>', unsafe_allow_html=True)
                    else:
                        st.success("✓ Nenhum conflito detectado")

                    st.session_state.history.append({
                        'q': query, 'a': answer, 'sources': results
                    })

                except Exception as e:
                    st.error(f"Erro: {e}")
