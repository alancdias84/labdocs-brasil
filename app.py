import streamlit as st
import re, json, hashlib, html, urllib.request
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
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
.main .block-container { padding-top: 1.5rem; max-width: 980px; }

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

.answer-text {
    font-size: 15px; line-height: 1.85; color: #2d2840;
    white-space: pre-wrap; word-break: break-word;
}

.src-card {
    background: white; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
    border: 1px solid #ece8f5; border-left: 4px solid #b39ddb;
}
.src-title { font-weight: 600; font-size: 13px; color: #4a3880; margin-bottom: 4px; }
.src-meta { font-size: 11px; color: #9e8fc0; font-family: monospace; margin-bottom: 6px; }
.src-snip { font-size: 13px; color: #4a445a; line-height: 1.6; white-space: pre-wrap; }

.warning-box {
    background: #fff8e8; border: 1px solid #f1d48a; color: #6e5612;
    padding: 14px 16px; border-radius: 12px; margin-top: 12px;
}
.error-box {
    background: #fff1f1; border: 1px solid #f0b4b4; color: #8f1f1f;
    padding: 14px 16px; border-radius: 12px; margin-top: 12px;
}

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
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────
@dataclass
class DocMeta:
    doc_id: str
    filename: str
    file_type: str
    title: str = ""
    version: str = "N/A"
    date: str = "N/A"
    sector: str = "N/A"
    page_count: int = 0
    char_count: int = 0
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    section: str
    text: str
    page: int
    position: int

# ─────────────────────────────────────────────────────────────────────
# NLP HELPERS
# ─────────────────────────────────────────────────────────────────────
STOPWORDS_PT = set([
    'a','ao','aos','as','até','com','como','da','das','de','do','dos','e','ela','ele',
    'em','na','nas','no','nos','ou','para','por','que','se','um','uma','é','à','o','os',
    'ser','foi','são','uma','uns','umas','dos','das'
])

def tokenize(text: str):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 1 and t not in STOPWORDS_PT]

def normalize_text(text: str) -> str:
    text = text.replace('\x00', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def make_id(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()[:12]

def escape_html(text: str) -> str:
    return html.escape(text)

# ─────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────
def extract_pdf(data: bytes):
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t:
                    pages.append((i + 1, t))
        return pages, len(pages)
    except Exception:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        pages = []
        for i, p in enumerate(reader.pages):
            text = p.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))
        return pages, len(reader.pages)

def split_text_into_chunks(text: str, max_chars: int = 1200, overlap: int = 180):
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]

        if end < len(text):
            last_break = max(chunk.rfind('. '), chunk.rfind('; '), chunk.rfind(': '), chunk.rfind(' '))
            if last_break > int(max_chars * 0.6):
                end = start + last_break + 1
                chunk = text[start:end]

        parts.append(chunk.strip())
        start = max(end - overlap, start + 1)

    return [p for p in parts if len(p) > 60]

# ─────────────────────────────────────────────────────────────────────
# INDEXING
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_index():
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import faiss

    docs_path = Path("documents")
    if not docs_path.exists():
        docs_path.mkdir()

    files = sorted([f for f in docs_path.glob("*") if f.suffix.lower() in {'.pdf', '.md', '.txt'}])
    if not files:
        return None

    all_meta = []
    all_chunks = []

    for f in files:
        doc_id = make_id(f.name)
        data = f.read_bytes()

        if f.suffix.lower() == '.pdf':
            pages, page_count = extract_pdf(data)
            total_chars = 0
            for p_num, text in pages:
                text = normalize_text(text)
                total_chars += len(text)
                split_chunks = split_text_into_chunks(text)
                for idx_local, piece in enumerate(split_chunks):
                    all_chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}-{len(all_chunks)}",
                            doc_id=doc_id,
                            filename=f.name,
                            section="Geral",
                            text=piece,
                            page=p_num,
                            position=len(all_chunks),
                        )
                    )
            all_meta.append(
                DocMeta(
                    doc_id=doc_id,
                    filename=f.name,
                    file_type=f.suffix[1:].upper(),
                    page_count=page_count,
                    char_count=total_chars
                )
            )
        else:
            text = data.decode('utf-8', errors='replace')
            text = normalize_text(text)
            split_chunks = split_text_into_chunks(text)
            for idx_local, piece in enumerate(split_chunks):
                all_chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-{len(all_chunks)}",
                        doc_id=doc_id,
                        filename=f.name,
                        section="Geral",
                        text=piece,
                        page=0,
                        position=len(all_chunks),
                    )
                )
            all_meta.append(
                DocMeta(
                    doc_id=doc_id,
                    filename=f.name,
                    file_type=f.suffix[1:].upper(),
                    page_count=0,
                    char_count=len(text)
                )
            )

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(
        [c.text for c in all_chunks],
        normalize_embeddings=True
    ).astype(np.float32)

    faiss_idx = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_idx.add(embeddings)

    bm25 = BM25Okapi([tokenize(c.text) for c in all_chunks])

    return {
        'chunks': all_chunks,
        'meta': {m.doc_id: m for m in all_meta},
        'bm25': bm25,
        'faiss': faiss_idx,
        'embeddings': embeddings,
        'model': model
    }

# ─────────────────────────────────────────────────────────────────────
# HYBRID RETRIEVAL
# ─────────────────────────────────────────────────────────────────────
def retrieve(query, idx, top_k=6, faiss_k=12, bm25_k=12):
    chunks = idx['chunks']

    # Dense retrieval
    q_vec = idx['model'].encode([query], normalize_embeddings=True).astype(np.float32)
    dense_scores, dense_ids = idx['faiss'].search(q_vec, faiss_k)

    dense_map = {}
    for score, cid in zip(dense_scores[0], dense_ids[0]):
        if cid >= 0:
            dense_map[int(cid)] = float(score)

    # Sparse retrieval
    bm25_scores = idx['bm25'].get_scores(tokenize(query))
    bm25_top_ids = np.argsort(bm25_scores)[::-1][:bm25_k]

    # Normalize sparse scores
    sparse_map = {}
    sparse_values = [float(bm25_scores[i]) for i in bm25_top_ids]
    sparse_max = max(sparse_values) if sparse_values else 1.0
    sparse_min = min(sparse_values) if sparse_values else 0.0
    denom = (sparse_max - sparse_min) if (sparse_max - sparse_min) > 1e-9 else 1.0

    for i in bm25_top_ids:
        sparse_map[int(i)] = (float(bm25_scores[i]) - sparse_min) / denom

    # Fusion
    candidate_ids = set(dense_map.keys()) | set(sparse_map.keys())
    fused = []
    for cid in candidate_ids:
        dense = dense_map.get(cid, 0.0)
        sparse = sparse_map.get(cid, 0.0)
        hybrid = 0.65 * dense + 0.35 * sparse
        fused.append({
            'chunk': chunks[cid],
            'dense': dense,
            'sparse': sparse,
            'hybrid': hybrid
        })

    fused.sort(key=lambda x: x['hybrid'], reverse=True)

    # Remove near duplicates
    selected = []
    seen_texts = set()
    for item in fused:
        text_key = item['chunk'].text[:280].strip().lower()
        if text_key not in seen_texts:
            selected.append(item)
            seen_texts.add(text_key)
        if len(selected) >= top_k:
            break

    return selected

# ─────────────────────────────────────────────────────────────────────
# PROMPT + GENERATION
# ─────────────────────────────────────────────────────────────────────
def build_context(results):
    blocks = []
    for i, r in enumerate(results, start=1):
        c = r['chunk']
        page_info = f"p. {c.page}" if c.page and c.page > 0 else "sem página"
        block = (
            f"[{i}] Arquivo: {c.filename} | Seção: {c.section} | {page_info}\n"
            f"Trecho:\n{c.text}"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)

def build_grounded_prompt(query, results):
    context = build_context(results)

    prompt = f"""
Você é um assistente de consulta documental extremamente restritivo.

Sua tarefa é responder APENAS com base nos trechos fornecidos em CONTEXTO.
Você NÃO pode usar conhecimento externo, inferências livres, opinião, complementação enciclopédica, exemplos inventados ou linguagem especulativa.

REGRAS OBRIGATÓRIAS:
1. Use exclusivamente informações literalmente apoiadas pelos trechos do CONTEXTO.
2. Toda afirmação factual da resposta deve ter citação no formato [1], [2], [3] etc.
3. As citações devem apontar para os trechos numerados do CONTEXTO.
4. Não escreva nada que não possa ser vinculado a pelo menos uma citação.
5. Se o CONTEXTO não trouxer informação suficiente para responder de forma objetiva, responda exatamente:
   "Não foram encontradas informações específicas sobre esse assunto nos documentos analisados."
6. Não use expressões como "em geral", "normalmente", "costuma", "pode", "é importante considerar", salvo se isso estiver explicitamente dito no CONTEXTO.
7. Seja objetivo, técnico e assertivo.
8. Não faça introdução desnecessária.
9. Não invente conclusão.
10. Quando houver múltiplos pontos documentais, organize a resposta em tópicos curtos.

FORMATO DA RESPOSTA:
- Entregue apenas a resposta final para o usuário.
- Use citações no corpo do texto.
- Não inclua seção chamada "contexto".
- Não inclua referências completas no final. Isso será exibido separadamente pela interface.

CONTEXTO:
{context}

PERGUNTA:
{query}
""".strip()

    return prompt

def call_openrouter(prompt, model_id):
    data = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "top_p": 0.9
    }).encode()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://labdocs-brasil.streamlit.app",
        "X-Title": "LabDocs Brasil"
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers=headers
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        res = json.loads(response.read())
        ans = res['choices'][0]['message']['content']
        return ans, len(prompt.split()), len(ans.split())

def validate_answer(answer: str):
    answer = answer.strip()

    fallback = "Não foram encontradas informações específicas sobre esse assunto nos documentos analisados."

    if not answer:
        return fallback, False, "Resposta vazia."

    if answer == fallback:
        return answer, True, ""

    cites = re.findall(r'\[(\d+)\]', answer)
    if not cites:
        return fallback, False, "Resposta sem citações."

    # Se houver linha factual sem citação, ainda assim força fallback.
    # Aqui a regra é dura de propósito.
    non_empty_lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    factual_lines_without_citation = [
        ln for ln in non_empty_lines
        if not re.search(r'\[\d+\]', ln) and not ln.startswith("- ") and len(ln) > 25
    ]
    if factual_lines_without_citation:
        return fallback, False, "Há linhas sem citação."

    return answer, True, ""

def generate(query, results, model_id):
    if not results:
        fallback = "Não foram encontradas informações específicas sobre esse assunto nos documentos analisados."
        return fallback, 0, len(fallback.split()), False, "Nenhum trecho recuperado."

    prompt = build_grounded_prompt(query, results)
    raw_answer, t_in, t_out = call_openrouter(prompt, model_id)
    validated_answer, ok, reason = validate_answer(raw_answer)

    return validated_answer, t_in, t_out, ok, reason

# ─────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────
def confidence_label(results):
    if not results:
        return "Baixa", "conf-baixa"

    avg_rel = sum(r['hybrid'] for r in results) / len(results)

    if avg_rel >= 0.70:
        return "Alta", "conf-alta"
    elif avg_rel >= 0.45:
        return "Média", "conf-media"
    else:
        return "Baixa", "conf-baixa"

def render_sources(results):
    if not results:
        return

    st.markdown("#### Fontes utilizadas")
    for i, r in enumerate(results, start=1):
        c = r['chunk']
        page_info = f"Página {c.page}" if c.page and c.page > 0 else "Sem página"
        title = f"[{i}] {c.filename}"
        meta = f"{page_info} | score híbrido={r['hybrid']:.3f} | dense={r['dense']:.3f} | sparse={r['sparse']:.3f}"
        snippet = c.text

        st.markdown(
            f"""
            <div class="src-card">
                <div class="src-title">{escape_html(title)}</div>
                <div class="src-meta">{escape_html(meta)}</div>
                <div class="src-snip">{escape_html(snippet)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações")

    model_id = st.selectbox("Modelo (gratuito)", [
        "qwen/qwen3.6-plus:free",
        "stepfun/step-3.5-flash:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "arcee-ai/trinity-large-preview:free",
        "minimax/minimax-m2.5:free"
    ])

    top_k = st.slider("Número de fontes recuperadas", 2, 10, 6)
    show_sources = st.checkbox("Mostrar trechos-fonte", value=True)

    st.markdown("---")
    st.caption("Modo documental restritivo: a resposta deve se limitar aos trechos recuperados.")

    if st.button("↩ Limpar histórico"):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="lab-header"><div class="lab-title">🧪 LabDocs Brasil</div><div class="lab-subtitle">Desafio SBPC/ML · RAG Laboratorial</div></div>',
    unsafe_allow_html=True
)

if 'index' not in st.session_state:
    with st.spinner("Indexando documentos..."):
        st.session_state.index = build_index()

if 'history' not in st.session_state:
    st.session_state.history = []

if not st.session_state.index:
    st.error("Nenhum documento encontrado na pasta 'documents'.")
    st.stop()

for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h['q'])

    with st.chat_message("assistant", avatar="🧪"):
        st.markdown(
            f"""
            <div class="answer-card">
                <div class="answer-meta">
                    <span class="meta-pill {h['conf_class']}">Consistência do resgate: {h['conf_label']}</span>
                    <span class="meta-pill">📥 {h['t_in']} | 📤 {h['t_out']} tokens</span>
                </div>
                <div class="answer-text">{escape_html(h["answer"])}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if h.get("warn_msg"):
            st.markdown(f'<div class="warning-box">{escape_html(h["warn_msg"])}</div>', unsafe_allow_html=True)
        if show_sources and h.get("sources"):
            render_sources(h["sources"])

query = st.chat_input("Pergunte sobre os documentos...")

if query and st.session_state.index:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant", avatar="🧪"):
        with st.spinner("Analisando documentos..."):
            try:
                res = retrieve(query, st.session_state.index, top_k=top_k)
                ans, t_in, t_out, ok, reason = generate(query, res, model_id)
                conf_label, conf_class = confidence_label(res)

                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-meta">
                            <span class="meta-pill {conf_class}">Consistência do resgate: {conf_label}</span>
                            <span class="meta-pill">📥 {t_in} | 📤 {t_out} tokens</span>
                        </div>
                        <div class="answer-text">{escape_html(ans)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                warn_msg = ""
                if not ok:
                    warn_msg = f"Saída original do modelo foi rejeitada pelo validador documental: {reason}"

                if warn_msg:
                    st.markdown(f'<div class="warning-box">{escape_html(warn_msg)}</div>', unsafe_allow_html=True)

                if show_sources:
                    render_sources(res)

                st.session_state.history.append({
                    'q': query,
                    'answer': ans,
                    'conf_label': conf_label,
                    'conf_class': conf_class,
                    't_in': t_in,
                    't_out': t_out,
                    'warn_msg': warn_msg,
                    'sources': res
                })

            except Exception as e:
                st.markdown(
                    f'<div class="error-box">Erro: {escape_html(str(e))}</div>',
                    unsafe_allow_html=True
                )
