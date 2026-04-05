import streamlit as st
import re, json, hashlib, html, urllib.request, urllib.error, socket, unicodedata
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
.info-box {
    background: #eef6ff; border: 1px solid #bfd8ff; color: #284b7a;
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

MODEL_CANDIDATES = [
    "qwen/qwen3.6-plus:free",
    "stepfun/step-3.5-flash:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "arcee-ai/trinity-large-preview:free",
    "minimax/minimax-m2.5:free"
]

# Controle interno de recuperação
RETRIEVE_FAISS_K = 16
RETRIEVE_BM25_K = 16
MIN_FINAL_CHUNKS = 3
MAX_FINAL_CHUNKS = 6
HYBRID_SCORE_MIN = 0.28
HIGH_CONFIDENCE_THRESHOLD = 0.72
MEDIUM_CONFIDENCE_THRESHOLD = 0.48

DOMAIN_SYNONYMS = {
    "resultado": ["valor", "laudo"],
    "critico": ["critico", "criticos", "critic", "urgente"],
    "referencia": ["referencia", "limite", "intervalo"],
    "poct": ["point", "care", "testing"],
}

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
    'a','ao','aos','as','ate','com','como','da','das','de','do','dos','e','ela','ele',
    'em','na','nas','no','nos','ou','para','por','que','se','um','uma','o','os',
    'ser','foi','sao','uns','umas'
])

def normalize_text(text: str) -> str:
    text = (text or "").replace('\x00', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

def normalize_for_search(text: str) -> str:
    text = normalize_text(text).lower()
    text = strip_accents(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_term(term: str) -> str:
    term = strip_accents(term.lower().strip())

    if len(term) > 4:
        if term.endswith("oes"):
            term = term[:-3] + "ao"
        elif term.endswith("aes"):
            term = term[:-3] + "ao"
        elif term.endswith("eis"):
            term = term[:-3] + "el"
        elif term.endswith("ais"):
            term = term[:-3] + "al"
        elif term.endswith("is"):
            term = term[:-2] + "il"
        elif term.endswith("ns"):
            term = term[:-2] + "m"
        elif term.endswith("es"):
            term = term[:-2]
        elif term.endswith("s"):
            term = term[:-1]

    return term

def tokenize(text: str):
    text = normalize_for_search(text)
    terms = [normalize_term(t) for t in text.split()]
    return [t for t in terms if len(t) > 1 and t not in STOPWORDS_PT]

def expand_query_terms(query: str):
    base_terms = tokenize(query)
    expanded = set(base_terms)

    for term in list(base_terms):
        expanded.add(term)

        if len(term) > 3:
            expanded.add(term + "s")

        if term.endswith("ao"):
            expanded.add(term[:-2] + "oes")
        if term.endswith("al"):
            expanded.add(term[:-2] + "ais")
        if term.endswith("el"):
            expanded.add(term[:-2] + "eis")
        if term.endswith("m"):
            expanded.add(term[:-1] + "ns")

        for syn in DOMAIN_SYNONYMS.get(term, []):
            expanded.add(normalize_term(syn))

    return [t for t in expanded if len(t) > 1]

def estimate_final_chunk_budget(query: str) -> int:
    q_tokens = tokenize(query)
    q_len = len(q_tokens)

    has_disjunction = any(x in normalize_for_search(query).split() for x in ["ou", "and", "or"])
    has_question_complexity = any(sym in query for sym in [":", ";", ","]) or len(query) > 80

    if q_len <= 2 and not has_question_complexity:
        return 3
    if q_len <= 5 and not has_disjunction:
        return 4
    if q_len <= 9:
        return 5
    return 6

def make_id(name: str) -> str:
    return hashlib.md5(name.encode()).hexdigest()[:12]

def escape_html(text: str) -> str:
    return html.escape(text)

def friendly_model_error(raw_error: str) -> str:
    txt = (raw_error or "").lower()

    if "429" in txt or "rate-limit" in txt or "rate limited" in txt or "temporarily rate-limited" in txt:
        return "Os modelos automáticos estavam temporariamente sobrecarregados no provedor de IA."
    if "401" in txt or "unauthorized" in txt or "invalid api key" in txt or "authentication" in txt:
        return "Houve um problema de autenticação com o serviço de IA."
    if "403" in txt or "forbidden" in txt:
        return "O acesso ao serviço de IA foi recusado pelo provedor."
    if "timeout" in txt or "timed out" in txt:
        return "O serviço de IA demorou demais para responder."
    if "connection reset" in txt or "temporary failure" in txt or "name resolution" in txt:
        return "Houve uma falha temporária de conexão com o serviço de IA."
    if "provider returned error" in txt:
        return "O provedor de IA retornou uma falha temporária durante a geração."
    if "empty response" in txt:
        return "O serviço de IA não retornou conteúdo utilizável."
    return "Não foi possível gerar a resposta com os modelos automáticos disponíveis neste momento."

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
                for piece in split_chunks:
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
            for piece in split_chunks:
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

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
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
def retrieve(query, idx):
    chunks = idx['chunks']

    query_for_dense = normalize_for_search(query)
    query_terms_expanded = expand_query_terms(query)
    query_for_sparse = " ".join(query_terms_expanded)

    q_vec = idx['model'].encode([query_for_dense], normalize_embeddings=True).astype(np.float32)
    dense_scores, dense_ids = idx['faiss'].search(q_vec, RETRIEVE_FAISS_K)

    dense_map = {}
    for score, cid in zip(dense_scores[0], dense_ids[0]):
        if cid >= 0:
            dense_map[int(cid)] = float(score)

    bm25_scores = idx['bm25'].get_scores(tokenize(query_for_sparse))
    bm25_top_ids = np.argsort(bm25_scores)[::-1][:RETRIEVE_BM25_K]

    sparse_map = {}
    sparse_values = [float(bm25_scores[i]) for i in bm25_top_ids]
    sparse_max = max(sparse_values) if sparse_values else 1.0
    sparse_min = min(sparse_values) if sparse_values else 0.0
    denom = (sparse_max - sparse_min) if (sparse_max - sparse_min) > 1e-9 else 1.0

    for i in bm25_top_ids:
        sparse_map[int(i)] = (float(bm25_scores[i]) - sparse_min) / denom

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

    deduped = []
    seen_texts = set()
    for item in fused:
        text_key = normalize_for_search(item['chunk'].text[:400])
        if text_key not in seen_texts:
            deduped.append(item)
            seen_texts.add(text_key)

    score_filtered = [x for x in deduped if x['hybrid'] >= HYBRID_SCORE_MIN]

    target_k = estimate_final_chunk_budget(query)

    if len(score_filtered) >= MIN_FINAL_CHUNKS:
        final_results = score_filtered[:min(target_k, MAX_FINAL_CHUNKS)]
    else:
        final_results = deduped[:min(max(target_k, MIN_FINAL_CHUNKS), MAX_FINAL_CHUNKS)]

    return final_results

# ─────────────────────────────────────────────────────────────────────
# LOCAL FALLBACK
# ─────────────────────────────────────────────────────────────────────
def chunk_matches_query(query: str, text: str) -> bool:
    q_terms = [normalize_term(t) for t in tokenize(query) if len(t) >= 3]
    if not q_terms:
        return False

    text_norm = normalize_for_search(text)
    text_terms = set(normalize_term(t) for t in text_norm.split())

    hits = sum(1 for t in q_terms if t in text_terms)
    min_hits = 1 if len(q_terms) <= 2 else 2
    return hits >= min_hits

def local_extractive_answer(query: str, results):
    fallback = "Não foram encontradas informações específicas sobre esse assunto nos documentos analisados."

    if not results:
        return fallback

    matched = []
    for i, r in enumerate(results, start=1):
        txt = normalize_text(r['chunk'].text)
        if chunk_matches_query(query, txt):
            matched.append((i, txt))

    if not matched:
        return fallback

    selected = matched[:3]
    bullet_lines = []

    for ref_id, txt in selected:
        sentences = re.split(r'(?<=[\.\!\?])\s+', txt)
        chosen = []

        for s in sentences:
            s = s.strip()
            if len(s) < 40:
                continue
            if chunk_matches_query(query, s):
                chosen.append(s)
            if len(chosen) == 2:
                break

        if not chosen:
            chosen = [txt[:350].strip()]

        joined = " ".join(chosen).strip()
        bullet_lines.append(f"- {joined} [{ref_id}]")

    return "\n".join(bullet_lines)

# ─────────────────────────────────────────────────────────────────────
# PROMPT + GENERATION
# ─────────────────────────────────────────────────────────────────────
def build_context(results):
    blocks = []
    for i, r in enumerate(results, start=1):
        c = r['chunk']
        page_info = f"p. {c.page}" if c.page and c.page > 0 else "sem página"
        cleaned_text = normalize_text(c.text)
        block = (
            f"[{i}] Arquivo: {c.filename} | Seção: {c.section} | {page_info}\n"
            f"Trecho:\n{cleaned_text}"
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
6. Se houver informação parcial no CONTEXTO, responda apenas com o que foi encontrado e cite as fontes correspondentes.
7. Não use expressões como "em geral", "normalmente", "costuma", "pode", "é importante considerar", salvo se isso estiver explicitamente dito no CONTEXTO.
8. Seja objetivo, técnico e assertivo.
9. Não faça introdução desnecessária.
10. Não invente conclusão.
11. Quando houver múltiplos pontos documentais, organize a resposta em tópicos curtos.

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

def _request_openrouter(prompt, model_name):
    data = json.dumps({
        "model": model_name,
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
        ans = res["choices"][0]["message"]["content"]
        if not ans or not ans.strip():
            raise Exception("empty response")
        return ans, len(prompt.split()), len(ans.split())

def call_openrouter_with_fallback(prompt):
    tried = []
    last_error = None

    for model_name in MODEL_CANDIDATES:
        if model_name in tried:
            continue
        tried.append(model_name)

        try:
            ans, t_in, t_out = _request_openrouter(prompt, model_name)
            return ans, t_in, t_out, model_name, ""
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            last_error = detail or str(e)
            continue
        except socket.timeout as e:
            last_error = str(e)
            continue
        except Exception as e:
            last_error = str(e)
            continue

    raise Exception(last_error or "all models failed")

def validate_answer(answer: str):
    answer = (answer or "").strip()

    fallback = "Não foram encontradas informações específicas sobre esse assunto nos documentos analisados."

    if not answer:
        return "", False, "Resposta vazia."

    if answer == fallback:
        return answer, True, ""

    cites = re.findall(r'\[(\d+)\]', answer)
    if not cites:
        return "", False, "Resposta sem citações."

    sentences = re.split(r'(?<=[\.\!\?])\s+|\n+', answer)
    cited_sentences = [s for s in sentences if re.search(r'\[\d+\]', s)]

    if not cited_sentences:
        return "", False, "Nenhuma sentença citada."

    return answer, True, ""

def generate(query, results):
    real_fallback = "Não foram encontradas informações específicas sobre esse assunto nos documentos analisados."

    if not results:
        return real_fallback, 0, len(real_fallback.split()), False, "Nenhum trecho recuperado.", "fallback_local"

    prompt = build_grounded_prompt(query, results)

    try:
        raw_answer, t_in, t_out, used_model, _ = call_openrouter_with_fallback(prompt)
    except Exception as e:
        local_answer = local_extractive_answer(query, results)
        friendly_reason = friendly_model_error(str(e))
        return local_answer, len(prompt.split()), len(local_answer.split()), False, friendly_reason, "fallback_local"

    validated_answer, ok, reason = validate_answer(raw_answer)

    if ok:
        return validated_answer, t_in, t_out, True, "", used_model

    local_answer = local_extractive_answer(query, results)
    return local_answer, t_in, len(local_answer.split()), False, "A resposta automática não veio em formato documental válido, então foi usada uma resposta de contingência baseada nos trechos recuperados.", used_model

# ─────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────
def confidence_label(results):
    if not results:
        return "Baixa", "conf-baixa"

    avg_rel = sum(r['hybrid'] for r in results) / len(results)

    if avg_rel >= HIGH_CONFIDENCE_THRESHOLD:
        return "Alta", "conf-alta"
    elif avg_rel >= MEDIUM_CONFIDENCE_THRESHOLD:
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
    show_sources = st.checkbox("Mostrar trechos-fonte", value=True)

    st.markdown("---")
    st.caption("Modelos de IA são selecionados automaticamente em cascata.")
    st.caption("O número de fontes é definido automaticamente pelo pipeline de recuperação.")
    st.caption("O tratamento linguístico da busca normaliza acentos, plural e variações simples de escrita.")

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
                    <span class="meta-pill">Modelo usado: {escape_html(h.get('used_model', 'N/A'))}</span>
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
                res = retrieve(query, st.session_state.index)
                ans, t_in, t_out, ok, reason, used_model = generate(query, res)
                conf_label, conf_class = confidence_label(res)

                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-meta">
                            <span class="meta-pill {conf_class}">Consistência do resgate: {conf_label}</span>
                            <span class="meta-pill">Modelo usado: {escape_html(used_model)}</span>
                            <span class="meta-pill">📥 {t_in} | 📤 {t_out} tokens</span>
                        </div>
                        <div class="answer-text">{escape_html(ans)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                warn_msg = ""
                if not ok and reason:
                    warn_msg = f"A resposta foi gerada com mecanismo de contingência. Motivo: {reason}"

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
                    'sources': res,
                    'used_model': used_model
                })

            except Exception:
                st.markdown(
                    '<div class="error-box">Erro ao processar a consulta. Tente novamente em instantes.</div>',
                    unsafe_allow_html=True
                )
