# ═══════════════════════════════════════════════════
# CÉLULA 1 — Instalação (execute uma vez)
# ═══════════════════════════════════════════════════

print('Instalando dependências... (pode levar 2-3 minutos)')
!pip install "numpy<2"

import subprocess, sys
pkgs = [
    'rank-bm25==0.2.2',
    'sentence-transformers==3.0.1',
    'faiss-cpu==1.8.0',
    'google-generativeai==0.7.2',
    'pdfplumber==0.11.4',
    'pypdf==4.3.1',
    'python-docx==1.1.2',
]
for pkg in pkgs:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)

print('✅ Instalação concluída!')


# ═══════════════════════════════════════════════════
# CÉLULA 2 — Configuração de API Key e Diretório
# ═══════════════════════════════════════════════════

import os
from pathlib import Path
from google.colab import userdata

# Carrega a API Key de forma segura usando Colab Secrets (Menu lateral esquerdo -> Chave)
try:
    os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
    print('✅ API Key carregada com sucesso dos Segredos do Colab.')
except userdata.SecretNotFoundError:
    print('❌ ERRO: Chave não encontrada! Vá no menu lateral esquerdo (ícone de chave), adicione um segredo chamado OPENROUTER_API_KEY com sua chave e ative o acesso.')

# Define a pasta onde os documentos já estão
docs_dir = Path('/content/documents')
docs_dir.mkdir(exist_ok=True)

# Lista os arquivos que já estão na pasta para confirmar
SUPPORTED = {'.pdf', '.rmd', '.md', '.txt', '.docx'}
arquivos = [f for f in docs_dir.rglob('*') if f.suffix.lower() in SUPPORTED]

if arquivos:
    print(f'\n✅ {len(arquivos)} arquivo(s) encontrados em {docs_dir} e prontos para indexação:')
    for arq in arquivos:
        print(f'  ✓ {arq.name}')
else:
    print(f'\n⚠️ Nenhum arquivo suportado foi encontrado em {docs_dir}.')

# ═══════════════════════════════════════════════════
# CÉLULA 3 — Indexação (BM25 + Embeddings semânticos)
# ═══════════════════════════════════════════════════

import re, json, pickle, hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from collections import Counter

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import pdfplumber
from pypdf import PdfReader

DOCS_DIR  = Path('/content/documents')
INDEX_DIR = Path('/content/index')
INDEX_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
STOPWORDS_PT = set([
    'a','ao','aos','as','até','com','como','da','das','de','dela','dele',
    'do','dos','e','ela','ele','em','entre','era','essa','esse','esta','este',
    'eu','foi','há','isso','isto','já','mais','mas','me','muito','na','nas',
    'nem','no','nos','num','o','os','ou','para','pela','pelo','por','qual',
    'quando','que','quem','se','seu','seus','si','sobre','sua','suas','também',
    'tem','tudo','um','uma','uns','você','é','à','ser','são','está','pode',
    'não','sendo','assim','cada','onde','pois','todo','toda','todos','todas',
])

# ── Dataclasses ──────────────────────────────────────
@dataclass
class DocMeta:
    doc_id: str; filename: str; file_type: str
    title: str=''; version: str='N/A'; date: str='N/A'
    sector: str='N/A'; criticality: str='N/A'
    page_count: int=0; char_count: int=0
    indexed_at: str=field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Chunk:
    chunk_id: str; doc_id: str; filename: str
    section: str; text: str; page: int; position: int

# ── NLP ──────────────────────────────────────────────
def normalize(t):
    return t.lower().strip()

def tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 1 and t not in STOPWORDS_PT]

# ── Metadata extraction ──────────────────────────────
def get_version(t):
    m = re.search(r'[Vv]ers[ãa]o\s*[:\-]?\s*([\d\.]+)|\bRev\.?\s*([\d\.]+)', t[:2000])
    return m.group(1) or m.group(2) if m else 'N/A'

def get_date(t):
    m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})|(\d{4}[\/\-]\d{2}[\/\-]\d{2})', t[:2000])
    return m.group(0) if m else 'N/A'

def make_id(p): return hashlib.md5(str(p).encode()).hexdigest()[:12]

# ── Text extraction ──────────────────────────────────
def extract_pdf(path):
    try:
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: pages.append(t)
        if pages: return '\n\n'.join(pages), len(pages)
    except: pass
    reader = PdfReader(str(path))
    pages = [p.extract_text() or '' for p in reader.pages]
    return '\n\n'.join(pages), len(pages)

def extract_text_file(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    text = re.sub(r'^---[\s\S]+?---\n', '', text)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.M)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    return text, 0

# ── Chunker ──────────────────────────────────────────
SECTION_RE = re.compile(r'^(#{1,4}\s+.{3,80}|(?:capítulo|seção|item)\s+[\d\.]+.{0,60}|\d+[\.]\s+[A-ZÁÉÍÓÚ].{5,60})', re.I | re.M)

def chunk_text(text, doc_id, filename, chunk_words=200, overlap=40):
    sections, last_h, last_p = [], 'Início', 0
    for m in SECTION_RE.finditer(text):
        if m.start() > last_p: sections.append((last_h, text[last_p:m.start()]))
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
                    chunks.append(Chunk(f'{doc_id}-{gid:04d}', doc_id, filename, sec_title, txt, 0, gid))
                    gid += 1
                buf = buf[-overlap:]
            buf.extend(words)
        if len(' '.join(buf)) > 50:
            chunks.append(Chunk(f'{doc_id}-{gid:04d}', doc_id, filename, sec_title, ' '.join(buf), 0, gid))
            gid += 1
    return chunks

# ── Main indexing ─────────────────────────────────────
print('📂 Lendo e fragmentando documentos...')
all_meta, all_chunks = [], []
SUPPORTED = {'.pdf', '.rmd', '.md', '.txt', '.docx'}

for f in DOCS_DIR.rglob('*'):
    if f.suffix.lower() not in SUPPORTED: continue
    try:
        ext = f.suffix.lower()
        raw, pages = extract_pdf(f) if ext == '.pdf' else extract_text_file(f)
        raw = re.sub(r'[ \t]+', ' ', raw)
        raw = re.sub(r'\n{3,}', '\n\n', raw).strip()
        doc_id = make_id(f)
        meta = DocMeta(
            doc_id=doc_id, filename=f.name,
            file_type=ext.lstrip('.').upper(),
            title=f.stem.replace('_',' '),
            version=get_version(raw), date=get_date(raw),
            page_count=pages, char_count=len(raw)
        )
        chunks = chunk_text(raw, doc_id, f.name)
        all_meta.append(meta)
        all_chunks.extend(chunks)
        print(f'  ✓ {f.name} → {len(chunks)} chunks')
    except Exception as e:
        print(f'  ✗ {f.name}: {e}')

print(f'\n⚡ Indexando {len(all_chunks)} chunks...')

# BM25
print('  [1/3] BM25...')
tok_corpus = [tokenize(c.text) for c in all_chunks]
bm25 = BM25Okapi(tok_corpus)

# Embeddings
print(f'  [2/3] Embeddings multilíngues (download na 1ª vez ~420MB)...')
model = SentenceTransformer(EMBEDDING_MODEL)
texts = [c.text for c in all_chunks]
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)

# FAISS
print('  [3/3] FAISS...')
dim = embeddings.shape[1]
faiss_idx = faiss.IndexFlatIP(dim)
faiss_idx.add(embeddings)

# Save
with open(INDEX_DIR / 'bm25.pkl', 'wb') as f: pickle.dump(bm25, f)
faiss.write_index(faiss_idx, str(INDEX_DIR / 'faiss.index'))
np.save(INDEX_DIR / 'embeddings.npy', embeddings)
with open(INDEX_DIR / 'chunks.json', 'w', encoding='utf-8') as f:
    json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False)
with open(INDEX_DIR / 'meta.json', 'w', encoding='utf-8') as f:
    json.dump([asdict(m) for m in all_meta], f, ensure_ascii=False)

print(f'\n✅ Índice pronto: {len(all_chunks)} chunks de {len(all_meta)} documento(s)')
print('Execute a Célula 4 para abrir a interface.')


# ═══════════════════════════════════════════════════
# CÉLULA 4 — Interface de perguntas
# ═══════════════════════════════════════════════════
# Não precisa de Streamlit — roda direto no Colab com interface interativa


import re, json, pickle, os, urllib.request
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from pathlib import Path

INDEX_DIR = Path('/content/index')
EMBEDDING_MODEL = 'paraphrase-multilingual-mpnet-base-v2'
STOPWORDS_PT = set([
    'a','ao','aos','as','com','como','da','das','de','do','dos','e','ela','ele',
    'em','era','essa','esse','esta','este','eu','foi','há','isso','isto','já',
    'mais','mas','muito','na','nas','não','nem','no','nos','o','os','ou','para',
    'pela','pelo','por','qual','quando','que','se','seu','seus','si','sua','suas',
    'também','tem','um','uma','uns','você','é','ser','são','está','pode',
])

print('Carregando índice...')
with open(INDEX_DIR / 'bm25.pkl', 'rb') as f:
    BM25 = pickle.load(f)
FAISS_IDX = faiss.read_index(str(INDEX_DIR / 'faiss.index'))
EMBEDDINGS = np.load(INDEX_DIR / 'embeddings.npy')
with open(INDEX_DIR / 'chunks.json', encoding='utf-8') as f:
    CHUNKS = json.load(f)
with open(INDEX_DIR / 'meta.json', encoding='utf-8') as f:
    META = {m['doc_id']: m for m in json.load(f)}
MODEL = SentenceTransformer(EMBEDDING_MODEL)
print(f'✅ {len(CHUNKS)} chunks carregados')

def tokenize(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if len(t) > 1 and t not in STOPWORDS_PT]

def retrieve(query, top_k=4):
    tokens = tokenize(query)
    bm25_sc = BM25.get_scores(tokens) if tokens else np.zeros(len(CHUNKS))
    bm25_rank = {int(i): r for r, i in enumerate(np.argsort(bm25_sc)[::-1][:50])}
    q_vec = MODEL.encode([query], normalize_embeddings=True).astype(np.float32)
    sc, ids = FAISS_IDX.search(q_vec, 50)
    sem_rank = {int(ids[0][i]): i for i in range(len(ids[0])) if ids[0][i] >= 0}
    sem_sc   = {int(ids[0][i]): float(sc[0][i]) for i in range(len(ids[0])) if ids[0][i] >= 0}
    all_ids = set(bm25_rank) | set(sem_rank)
    rrf = {cid: 1/(60+bm25_rank.get(cid,50)) + 1/(60+sem_rank.get(cid,50)) for cid in all_ids}
    top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    mx = top[0][1] if top else 1
    return [{'chunk': CHUNKS[cid], 'rrf': score, 'rel': score/mx, 'sem': sem_sc.get(cid,0), 'bm25': float(bm25_sc[cid]) if cid < len(bm25_sc) else 0} for cid, score in top]

def best_snippet(text, query_words, max_w=100):
    sents = re.split(r'(?<=[.!?\n])\s+', text)
    if len(sents) <= 2: return ' '.join(text.split()[:max_w])
    qw = set(query_words.lower().split())
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

NUM_RE = re.compile(r'\b(\d+[\.,]?\d*)\s*(%|mg|mL|g|dL|µL|UI|mmol|nmol|h|min|°C|dias?|horas?)\b', re.I)
PRESC_RE = re.compile(r'\b(deve|deverá|não deve|obrigatório|proibido|shall|must|required)\b', re.I)

def detect_conflicts(results):
    conflicts = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            ca, cb = results[i]['chunk'], results[j]['chunk']
            if ca['doc_id'] == cb['doc_id']: continue
            ea = EMBEDDINGS[ca['position']].astype(np.float32)
            eb = EMBEDDINGS[cb['position']].astype(np.float32)
            sim = float(np.dot(ea, eb))
            if sim < 0.82: continue
            na = set(m.group(0) for m in NUM_RE.finditer(ca['text']))
            nb = set(m.group(0) for m in NUM_RE.finditer(cb['text']))
            if (na-nb) and (nb-na):
                conflicts.append({'sev':'Alta','desc':f'Valores divergentes entre "{ca["filename"]}" e "{cb["filename"]}"','sim':sim})
            elif PRESC_RE.search(ca['text']) and PRESC_RE.search(cb['text']):
                conflicts.append({'sev':'Média','desc':f'Orientações prescritivas conflitantes entre "{ca["filename"]}" e "{cb["filename"]}"','sim':sim})
    return conflicts

def generate_answer(query, results):
    parts = []
    for i, r in enumerate(results, 1):
        c = r['chunk']
        m = META.get(c['doc_id'], {})
        snip = best_snippet(c['text'], query)
        parts.append(f"[{i}] {c['filename']} | v{m.get('version','N/A')} | {c['section'][:50]}\n{snip}")
    ctx = '\n---\n'.join(parts)
    prompt = (
        "Assistente de medicina laboratorial. Responda em português baseado EXCLUSIVAMENTE "
        "nos trechos abaixo. Cite a fonte como [n]. Se não encontrar, diga explicitamente.\n\n"
        f"TRECHOS:\n{ctx}\n\nPERGUNTA: {query}"
    )
    body = json.dumps({
        "model": os.environ.get('OPENROUTER_MODEL', 'qwen/qwen3.6-plus:free'),
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY','')}",
            "Content-Type": "application/json"
        }
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    text = resp['choices'][0]['message']['content']
    return text, len(prompt.split()), len(text.split())

query_box = widgets.Text(
    placeholder='Ex: Qual o critério de rejeição de amostra para potássio?',
    layout=widgets.Layout(width='80%'),
    description='Pergunta:',
)
topk_slider = widgets.IntSlider(value=4, min=2, max=8, step=1, description='Top-K:', layout=widgets.Layout(width='40%'))
ask_btn = widgets.Button(description='Perguntar', button_style='primary', icon='search')
output = widgets.Output()

def on_ask(_):
    q = query_box.value.strip()
    if not q: return
    with output:
        clear_output()
        print('Buscando fontes...')
        results = retrieve(q, top_k=topk_slider.value)
        if not results:
            print('Nenhuma fonte encontrada.')
            return
        conflicts = detect_conflicts(results)
        print(f'Gerando resposta com {len(results)} fontes...')
        answer, tok_in, tok_out = generate_answer(q, results)
        clear_output()
        avg_rel = sum(r['rel'] for r in results) / len(results)
        n_docs  = len(set(r['chunk']['doc_id'] for r in results))
        conf = min(1.0, avg_rel * 0.6 + n_docs/3 * 0.3 - len([c for c in conflicts if c['sev']=='Alta'])*0.15)
        conf_label = 'Alta' if conf >= 0.65 else 'Média' if conf >= 0.35 else 'Baixa'
        conf_color = {'Alta':'#2ecc71','Média':'#ffd166','Baixa':'#ff4757'}[conf_label]
        html = f"""
<div style="font-family:sans-serif;max-width:860px">
  <div style="background:#161b22;border:1px solid #30363d;border-left:4px solid #00e5ff;border-radius:6px;padding:16px 20px;margin-bottom:12px">
    <div style="font-size:11px;color:#8b949e;margin-bottom:8px">
      Confiança: <b style="color:{conf_color}">{conf_label} ({conf:.0%})</b> &nbsp;|&nbsp;
      tokens: entrada <b>{tok_in}</b> · saída <b>{tok_out}</b> &nbsp;|&nbsp; custo: <b>gratuito (OpenRouter)</b>
    </div>
    <div style="font-size:15px;line-height:1.8;color:#e6edf3;white-space:pre-wrap">{answer}</div>
  </div>
  <details open><summary style="cursor:pointer;color:#58a6ff;font-size:13px;margin-bottom:8px">📄 {len(results)} fonte(s)</summary>"""
        for i, r in enumerate(results, 1):
            c = r['chunk']; m = META.get(c['doc_id'], {})
            snip = best_snippet(c['text'], q, 120)
            html += f"""<div style="background:#161b22;border:1px solid #21262d;border-left:4px solid #ffd166;border-radius:6px;padding:10px 14px;margin-bottom:6px">
      <div style="color:#58a6ff;font-weight:600;font-size:12px">[{i}] {c['filename']}</div>
      <div style="color:#6e7681;font-size:11px;font-family:monospace;margin:3px 0">v{m.get('version','N/A')} | {c['section'][:55]} | {int(r['rel']*100)}% relevância</div>
      <div style="color:#c9d1d9;font-size:13px;line-height:1.6">{snip}…</div></div>"""
        html += "</details>"
        if conflicts:
            html += f'<details><summary style="cursor:pointer;color:#ff4757;font-size:13px">⚠️ {len(conflicts)} conflito(s)</summary>'
            for c in conflicts:
                col = {'Alta':'#ff4757','Média':'#ffd166','Baixa':'#6e7681'}[c['sev']]
                html += f'<div style="border-left:3px solid {col};padding:8px 12px;margin:4px 0;background:#161b22;border-radius:4px;font-size:12px;color:#c9d1d9">{c["desc"]}</div>'
            html += '</details>'
        else:
            html += '<div style="color:#2ecc71;font-size:12px;margin-top:6px">✓ Nenhum conflito detectado</div>'
        html += '</div>'
        display(HTML(html))

ask_btn.on_click(on_ask)
query_box.on_submit(on_ask)
display(widgets.VBox([
    widgets.HTML('<h3 style="color:#e6edf3">🧪 LabDocs Brasil — Interface de Consulta</h3>'),
    widgets.HBox([query_box, ask_btn]),
    topk_slider,
    output
]))

