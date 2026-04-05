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

.answer-card {
    background: white; border-radius: 14px; padding: 24px 28px; margin-bottom: 16px;
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
}
.src-title { font-weight: 600; font-size: 13px; color: #4a3880; margin-bottom: 4px; }
.src-meta { font-size: 11px; color: #9e8fc0; font-family: monospace; margin-bottom: 6px; }
.src-snip { font-size: 13px; color: #4a445a; line-height: 1.6; }

.conflict-alta  { border-left-color: #f5a0a0 !important; }
.conflict-media { border-left-color: #f5d78e !important; }

.stButton > button {
    background: linear-gradient(135deg, #7c5cbf 0%, #9b7dd4 100%);
    color: white; border-radius: 10px; font-weight: 600;
}
div[data-testid="stSidebar"] { background: white; border-right: 1px solid #ece8f5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# API CONFIG
# ─────────────────────────────────────────────────────────────────────
api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────
# DATACLASSES & NLP
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

STOPWORDS_PT = set(['a','ao','aos','as','até','com','como','da','das','de','do','dos','e','ela','ele','em','na','nas','no','nos','ou','para','por','que','se','um','uma','é','à'])

def tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 1 and t not in STOPWORDS_PT]

def make_id(name): return hashlib.md5(name.encode()).hexdigest()[:12]

# ─────────────────────────────────────────────────────────────────────
# EXTRACTION & INDEXING
# ─────────────────────────────────────────────────────────────────────
def extract_pdf(data: bytes):
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t: pages.append((i + 1, t))
        return pages, len(pages)
    except:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        return [(i+1, p.extract_text() or '') for i, p in enumerate(reader.pages)], len(reader.pages)

@st.cache_resource(show_spinner=False)
def build_index():
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import faiss

    docs_path = Path("documents")
    if not docs_path.exists(): docs_path.mkdir()
    
    files = sorted([f for f in docs_path.glob("*") if f.suffix.lower() in {'.pdf', '.md', '.txt'}])
    if not files: return None

    all_meta, all_chunks = [], []
    for f in files:
        doc_id = make_id(f.name)
        data = f.read_bytes()
        if f.suffix.lower() == '.pdf':
            pages, _ = extract_pdf(data)
            for p_num, text in pages:
                paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 40]
                for p in paras:
                    all_chunks.append(Chunk(f"{doc_id}-{len(all_chunks)}", doc_id, f.name, "Geral", p, p_num, len(all_chunks)))
        else:
            text = data.decode('utf-8', errors='replace')
            paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 40]
            for p in paras:
                all_chunks.append(Chunk(f"{doc_id}-{len(all_chunks)}", doc_id, f.name, "Geral", p, 0, len(all_chunks)))
        
        all_meta.append(DocMeta(doc_id=doc_id, filename=f.name, file_type=f.suffix[1:].upper()))

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode([c.text for c in all_chunks], normalize_embeddings=True).astype(np.float32)
    faiss_idx = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_idx.add(embeddings)
    
    return {'chunks': all_chunks, 'meta': {m.doc_id: m for m in all_meta}, 
            'bm25': BM25Okapi([tokenize(c.text) for c in all_chunks]), 
            'faiss': faiss_idx, 'embeddings': embeddings, 'model': model}

# ─────────────────────────────────────────────────────────────────────
# RAG CORE
# ─────────────────────────────────────────────────────────────────────
def retrieve(query, idx, top_k=4):
    q_vec = idx['model'].encode([query], normalize_embeddings=True).astype(np.float32)
    sc, ids = idx['faiss'].search(q_vec, top_k)
    return [{'chunk': idx['chunks'][ids[0][i]], 'rel': float(sc[0][i])} for i in range(len(ids[0])) if ids[0][i] >= 0]

def generate(query, results, model_id):
    ctx = "\n---\n".join([f"Fonte [{i+1}]: {r['chunk'].text}" for i, r in enumerate(results)])
    prompt = f"Você é um especialista em medicina laboratorial. Responda em Português usando:\n{ctx}\n\nPergunta: {query}"
    
    data = json.dumps({"model": model_id, "messages": [{"role": "user", "content": prompt}]}).encode()
    
    # AJUSTE DE HEADERS PARA EVITAR 404
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://labdocs-brasil.streamlit.app", # Streamlit Cloud URL
        "X-Title": "LabDocs Brasil"
    }
    
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            res = json.loads(response.read())
            ans = res['choices'][0]['message']['content']
            return ans, len(prompt.split()), len(ans.split())
    except Exception as e:
        raise Exception(f"Erro na OpenRouter: {str(e)}")

# ─────────────────────────────────────────────────────────────────────
# UI SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    # MODELOS GRATUITOS PRINCIPAIS
    model_id = st.selectbox("Modelo (gratuito)", [
        "qwen/qwen3.6-plus:free",
        "stepfun/step-3.5-flash:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "arcee-ai/trinity-large-preview:free",
        "minimax/minimax-m2.5:free"
    ])
    top_k = st.slider("Fontes", 2, 8, 4)
    if st.button("↩ Limpar histórico"):
        st.session_state.history = []; st.rerun()

# ─────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="lab-header"><div class="lab-title">🧪 LabDocs Brasil</div><div class="lab-subtitle">Desafio SBPC/ML · RAG Laboratorial</div></div>', unsafe_allow_html=True)

if 'index' not in st.session_state:
    with st.spinner("Indexando documentos..."): st.session_state.index = build_index()
if 'history' not in st.session_state: st.session_state.history = []

for h in st.session_state.history:
    with st.chat_message("user"): st.write(h['q'])
    with st.chat_message("assistant", avatar="🧪"):
        st.markdown(f'<div class="answer-card"><div class="answer-text">{h["answer"]}</div></div>', unsafe_allow_html=True)

query = st.chat_input("Pergunte sobre os documentos...")
if query and st.session_state.index:
    with st.chat_message("user"): st.write(query)
    with st.chat_message("assistant", avatar="🧪"):
        with st.spinner("Analisando documentos..."):
            try:
                res = retrieve(query, st.session_state.index, top_k=top_k)
                ans, t_in, t_out = generate(query, res, model_id)
                
                # Interface de Resposta
                avg_rel = sum(r['rel'] for r in res)/len(res)
                label = "Alta" if avg_rel > 0.7 else "Média" if avg_rel > 0.4 else "Baixa"
                
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-meta">
                        <span class="meta-pill">Confiança: {label}</span>
                        <span class="meta-pill">📥 {t_in} | 📤 {t_out} tokens</span>
                    </div>
                    <div class="answer-text">{ans}</div>
                </div>""", unsafe_allow_html=True)
                
                st.session_state.history.append({'q': query, 'answer': ans, 'conf_label': label})
            except Exception as e:
                st.error(f"Erro: {e}")
