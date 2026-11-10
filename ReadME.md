---
title: Zoha Shafique Digital Clone
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Zoha Shafique: Digital Clone (Hybrid RAG)

**This is an AI clone, not the real person.** Ask it questions about Zoha Shafique. It retrieves passages from her own documents using BM25, dense embeddings, or both combined (hybrid), and answers in the first person using only those passages.

## How it works

1. **Chunking:** each document is split into overlapping chunks of about 250 words (40 words overlap). The source file name and chunk ID are stored with every chunk.
2. **BM25:** `rank_bm25` over tokenized chunks.
3. **Dense:** `all-MiniLM-L6-v2` sentence embeddings, cosine similarity with numpy, embeddings precomputed and saved in `embeddings.npy`. No vector database.
4. **Hybrid:** Reciprocal Rank Fusion, `score = sum of 1 / (60 + rank)` over the BM25 and dense top 20 lists. (The other common option is a weighted score `alpha * dense + (1 - alpha) * bm25` after min-max normalizing each score. I did not use it.)
5. **Generation:** the top 4 chunks and the question go to a free Llama model on Groq. The prompt makes it answer in the first person, only from the context, and say "I don't have that in my notes." when the answer is missing. The API key is stored as a Space secret (`GROQ_API_KEY`), never in the code.
6. **UI:** Gradio app with a question box, answer box, mode selector (BM25 only, Dense only, Hybrid), and a panel with the retrieved chunks, sources and scores.

## Knowledge base

Eight files in `data/`, all written by me about myself:

- `bio.md` and `education.md`: factual background (degree, semester, Matric and Pre-Medical history).
- `experience.md`: my internship at Big Brains (Android app testing, test cases, OWASP ZAP, Burp Suite).
- `skills.md`: languages, tools and technologies.
- `projects.md` and `project_writeup.md`: my four academic projects, the second one as a long-form write-up.
- `personality.md`: interests, values, learning style, opinions and goals.
- `faq.md`: 30 first-person questions answered the way I would answer them.

I chose these because I can judge immediately whether an answer is right. Rare keywords such as "Burp Suite", "OWASP ZAP", "Big Brains" and "Plagiarism Detection System" give BM25 something to win on, and descriptive sentences like "I break the concept into smaller parts" let dense search handle paraphrased questions. No phone numbers, addresses, ID numbers or private details about other people are included.

## Test questions

Each cell shows the top chunk (file, chunk number) and its score. BM25 scores and dense cosine scores are on different scales, and hybrid RRF scores are at most about 0.033.

| # | Question | BM25 result | Dense result | Hybrid result | Best mode and why |
|---|---|---|---|---|---|
| 1 | What is Burp Suite? (exact rare keyword) | faq.md c1 (3.53) | faq.md c1 (0.12) | faq.md c1 (0.0328) | All three found the same chunk (the generated answer for this question is discussed in the failure case below). BM25 is the clearest win: "Burp Suite" is an exact rare term, so its score is high, while dense gave a low score (0.12) because a tool name carries little meaning for the embedding model. |
| 2 | How do you handle things you find hard to understand? (paraphrase) | faq.md c2 (9.24) | project_writeup.md c2 (0.30) | faq.md c1 (0.0320) | **Dense.** The answer says "when a technical concept is difficult, I break it into smaller parts", which shares almost no words with the question. Dense matched the meaning. BM25 matched common words like "you" and "do" and returned a chunk (Q21 to Q30) that does not contain the answer. Hybrid recovered by pulling in the FAQ chunk containing Q19. Note: the final BM25-mode answer was still good, because the LLM receives the top 4 chunks and the right text was inside one of them even though it was not ranked first. |
| 3 | What did you do at Big Brains? | faq.md c1 (9.38) | experience.md c0 (0.33) | experience.md c0 (0.0320) | **Hybrid (tied with dense).** BM25 returned the short FAQ answer. Dense and hybrid returned experience.md, which has the full internship description (Android testing, test cases, authentication testing, OWASP ZAP, Burp Suite). |
| 4 | Tell me about your plagiarism project | faq.md c1 (4.49) | projects.md c0 (0.42) | faq.md c1 (0.0320) | **Dense.** projects.md has the complete project section (SRE course, text similarity, text processing). BM25 and hybrid put the short FAQ answer first. |
| 5 | What is your favorite food? (not in my documents) | faq.md c0 (3.85) | faq.md c2 (0.09) | faq.md c0 (0.0323) | No retriever can win here because the answer does not exist. Dense gave the lowest score (0.09), a useful warning sign. The LLM correctly answered "I don't have that in my notes." instead of inventing a food. |

## Failure case

**Question 1, "What is Burp Suite?", in Hybrid mode.** Retrieval worked: the top chunk was `faq.md` chunk 1, which contains my answer to "Which testing tools have you explored?" ("I have explored OWASP ZAP and Burp Suite concepts during my learning and internship experience"). But the clone answered only "I don't have that in my notes." That is honest, but it is a weak answer, because my notes do say something relevant: I explored Burp Suite during my internship.

Why it happened: the question asks for a definition, while my notes only mention Burp Suite in passing as a tool I used and never define it. My prompt tells the model to answer ONLY from the context and otherwise say it does not know, so the model treated "no definition" as "no answer" instead of sharing the partial information. The retriever was not the problem. The prompt rule is all-or-nothing, and my knowledge base is thin on explanations of tools.

## Improvement I would make next

Change the prompt so that when the context only partly answers a question, the clone says what my notes do contain (for example, "I explored Burp Suite during my internship") and clearly states what is missing, instead of replying with a flat "I don't have that in my notes." I would also add a short description of each tool I used to `skills.md`, split the FAQ into one chunk per question, add a cross-encoder reranker over the hybrid top 10, and remove stopwords from the BM25 tokenizer.