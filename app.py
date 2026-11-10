import os, re, glob
import numpy as np
import requests
import gradio as gr
from rank_bm25 import BM25Okapi

# Query embeddings. Locally this uses sentence-transformers. On small servers
# (Render free = 512 MB RAM) torch does not fit, so it falls back to fastembed,
# which runs the same all-MiniLM-L6-v2 model with ONNX and much less memory.
try:
    from sentence_transformers import SentenceTransformer
    _st = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(texts):
        return _st.encode(texts, normalize_embeddings=True)
except ImportError:
    from fastembed import TextEmbedding
    _fe = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def encode(texts):
        v = np.array(list(_fe.embed(texts)), dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)


NAME = "Zoha Shafique"
CHUNK_WORDS, OVERLAP = 250, 40

# Free LLM through Groq. The key is read from the GROQ_API_KEY secret.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
PREFERRED = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile",
             "openai/gpt-oss-20b", "openai/gpt-oss-120b",
             "qwen/qwen3-32b",
             "meta-llama/llama-4-scout-17b-16e-instruct"]
SKIP = ("whisper", "guard", "tts", "orpheus", "playai", "embed", "safeguard")
_models_cache = []


def available_models(key):
    """Ask Groq which models this key can actually use, best choices first."""
    if _models_cache:
        return _models_cache
    r = requests.get(GROQ_MODELS_URL,
                     headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Could not list models: HTTP {r.status_code} {r.text[:150]}")
    ids = [m["id"] for m in r.json().get("data", [])
           if not any(s in m["id"].lower() for s in SKIP)]
    ordered = [m for m in PREFERRED if m in ids] + [m for m in ids if m not in PREFERRED]
    _models_cache.extend(ordered)
    return _models_cache


# 1. Load and chunk (keeps source file name and chunk id)
def load_chunks():
    chunks = []
    for path in sorted(glob.glob("data/*.md") + glob.glob("data/*.txt")):
        words = open(path, encoding="utf-8").read().split()
        step = CHUNK_WORDS - OVERLAP
        for i, start in enumerate(range(0, max(len(words), 1), step)):
            piece = words[start:start + CHUNK_WORDS]
            if len(piece) < 30 and i > 0:
                continue
            chunks.append({"source": os.path.basename(path),
                           "chunk_id": i, "text": " ".join(piece)})
    return chunks


chunks = load_chunks()
texts = [c["text"] for c in chunks]

# 2. Index: BM25 + dense embeddings
tok = lambda s: re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok(t) for t in texts])

if os.path.exists("embeddings.npy") and len(np.load("embeddings.npy")) == len(texts):
    emb = np.load("embeddings.npy")
else:
    emb = np.array(encode(texts))
    np.save("embeddings.npy", emb)


# 3. Retrievers: return list of (chunk index, score)
def bm25_search(q, k=5):
    s = bm25.get_scores(tok(q))
    top = np.argsort(s)[::-1][:k]
    return [(int(i), float(s[i])) for i in top]


def dense_search(q, k=5):
    qv = encode([q])[0]
    s = emb @ qv  # cosine similarity (embeddings are normalized)
    top = np.argsort(s)[::-1][:k]
    return [(int(i), float(s[i])) for i in top]


# 4. Hybrid: Reciprocal Rank Fusion, score = sum of 1 / (60 + rank)
def hybrid_search(q, k=5):
    scores = {}
    for results in (bm25_search(q, 20), dense_search(q, 20)):
        for rank, (i, _) in enumerate(results, start=1):
            scores[i] = scores.get(i, 0) + 1 / (60 + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


# 5. Generation (persona prompt, grounded in retrieved context only)
PROMPT = """You are the AI clone of {name}, a 7th semester BS Software Engineering student.
Speak in the first person, in a friendly, simple tone.

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say exactly: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, say you are an AI clone of {name}.

CONTEXT:
{context}

QUESTION:
{question}"""


def generate(question, results):
    context = "\n\n".join(
        f"[{chunks[i]['source']}] {chunks[i]['text']}" for i, _ in results)
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return ("(No GROQ_API_KEY set, so no AI answer. "
                "Showing the best retrieved text instead.)\n\n"
                + chunks[results[0][0]]["text"][:400])
    prompt = PROMPT.format(name=NAME, context=context, question=question)
    errors = []
    try:
        models_to_try = available_models(key)[:4]
    except Exception as e:
        return f"Generation error: {e}"
    if not models_to_try:
        return "Generation error: this Groq key has no chat models available."
    for m in models_to_try:
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": m,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 400, "temperature": 0.3},
                timeout=60)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"] or ""
                return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            errors.append(f"{m}: HTTP {r.status_code} {r.text[:150]}")
        except Exception as e:
            errors.append(f"{m}: {str(e)[:150]}")
    return "Generation error:\n" + "\n".join(errors)


# 6. UI
def ask(question, mode):
    if not question.strip():
        return "Please type a question.", ""
    fn = {"BM25 only": bm25_search,
          "Dense only": dense_search,
          "Hybrid": hybrid_search}[mode]
    results = fn(question, 4)
    answer = generate(question, results)
    panel = ""
    for i, s in results:
        c = chunks[i]
        panel += (f"**{c['source']}** (chunk {c['chunk_id']}) | score: {s:.4f}\n\n"
                  f"{c['text'][:500]}...\n\n---\n")
    return answer, panel


with gr.Blocks(title=f"{NAME}'s Digital Clone") as demo:
    gr.Markdown("# ⚠️ This is an AI clone, not the real person.")
    gr.Markdown(f"Ask me anything about {NAME}.")
    q = gr.Textbox(label="Your question")
    mode = gr.Radio(["BM25 only", "Dense only", "Hybrid"],
                    value="Hybrid", label="Retrieval mode")
    btn = gr.Button("Ask")
    ans = gr.Textbox(label="Answer", lines=6)
    ctx = gr.Markdown(label="Retrieved chunks")
    btn.click(ask, [q, mode], [ans, ctx])
    q.submit(ask, [q, mode], [ans, ctx])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)))