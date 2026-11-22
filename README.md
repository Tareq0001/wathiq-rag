# وَثِيق (Wathiq) — Enterprise Document ETL & Hybrid RAG Engine

**Live Interactive Demo (GitHub Pages):** [https://tareq0001.github.io/wathiq-rag/](https://tareq0001.github.io/wathiq-rag/)  
**Mirror (Surge Edge):** [https://wathiq-rag.surge.sh](https://wathiq-rag.surge.sh)

> **Dual-Mode Workbench Architecture:** The static demo hosted on GitHub Pages (and Surge mirror) runs an **In-Browser Edge Engine** (`MODE: SURGE EDGE ENGINE`) implementing Saudi PDPL PII masking, prompt/SQL injection guardrails, Mozilla `pdf.js` page-by-page extraction, Arabic normalization, SHA-256 deduplication (`crypto.subtle`), 256-dim n-gram vector embeddings, and Okapi BM25 + Dense Cosine RRF (`k=60`) directly in the browser. Running the repository locally or via Docker connects the identical UI (`API: FASTAPI ONLINE`) to the **FastAPI + PyMuPDF + SQLite WAL / PostgreSQL 16 `pgvector` Backend**.

**Wathiq** is a production-structured bilingual (Arabic + English) Document ETL & Hybrid Retrieval-Augmented Generation (RAG) engine built with **FastAPI**, **PyMuPDF**, **NumPy**, **SQLite WAL**, and a ready-to-deploy **PostgreSQL 16 + pgvector (HNSW)** schema.

It combines **Saudi PDPL PII masking**, **prompt/SQL injection guardrails**, **SHA-256 deduplication**, **Arabic text normalization**, **Okapi BM25 + Dense Cosine Similarity fused via Reciprocal Rank Fusion (RRF)**, **grounded citation synthesis**, and **immutable audit telemetry**.

---

## Architecture

```text
  PDF / Raw Text ──▶ [PDPL Guardrails] ──▶ [SHA-256 Deduplication]
                            │
                            ▼
                    [PyMuPDF + Arabic ETL]
             (Tatweel/Alef/Teh Marbuta Normalization)
                            │
                            ▼
         [Word-Boundary Sliding Chunks (450 / 90 overlap)]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   [Okapi BM25 Lexical]        [256-d Dense Embeddings]
              │                           │
              └─────────────┬─────────────┘
                            ▼
           [Reciprocal Rank Fusion (RRF, k=60)]
                            │
                            ▼
        [Grounded Synthesis + Faithfulness + Audit Log]
```

---

## Quickstart (Local Python)

```bash
cd C:\Users\abuos\.gemini\antigravity\scratch\wathiq-rag
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open the live RTL/LTR Workbench at `http://127.0.0.1:8000/`.

Run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

*(Optional External LLM)*: By default, Wathiq uses deterministic extractive synthesis so it runs offline with zero API keys. To enable Gemini synthesis, set `GEMINI_API_KEY` and `USE_EXTERNAL_LLM=1`.

---

## Docker & `pgvector` PostgreSQL 16 Deployment

```bash
docker compose up --build -d
```

- **`db`**: `pgvector/pgvector:pg16` initialized via `sql/init_pgvector.sql` (`vector(256)` column with `HNSW` cosine index + GIN full-text index).
- **`wathiq-api`**: FastAPI service exposed on port `8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Live RTL/LTR Hybrid RAG Workbench UI |
| `GET` | `/api/v1/health` | Service health check |
| `POST` | `/api/v1/documents/ingest` | Ingest raw text with PDPL PII masking & SHA-256 deduplication |
| `POST` | `/api/v1/documents/upload-pdf` | Multipart PDF upload parsed page-by-page via PyMuPDF |
| `GET` | `/api/v1/documents` | List indexed documents and sensitivity classifications |
| `POST` | `/api/v1/query` | Security guardrail check + Hybrid BM25/Dense RRF + grounded answer |
| `GET` | `/api/v1/metrics` | Corpus stats, PII redactions, guardrail blocks, & recent audit logs |

---

## دليل إجابات المقابلة الشخصية للمهندس طارق (Interview Defense Guide)

### 1. لماذا استخدمت التقطيع المتداخل (Chunking with Overlap: `450` / `90`) مع مراعاة حدود الكلمات؟
- **السبب:** تقطيع النصوص الطويلة يحافظ على دقة الاسترجاع لأن المتجهات تفقد التركيز عند ضغط صفحات كاملة في متجه واحد. التداخل (`90` حرفاً) يمنع ضياع المعنى إذا انقطعت الجملة بين مقطعين، ومراعاة حدود الكلمات تمنع قص الكلمات العربية أو الإنجليزية من المنتصف.

### 2. ما أهمية تطبيع النصوص العربية (Arabic Normalization) في محركات البحث؟
- **السبب:** الكاتب قد يكتب (الإدارة / الادارة) أو (مدرسة / مدرسه) أو (مستشفى / مستشفي) أو يضيف التطويل (`ـ`) والتشكيل. توحيد الألف (`أ إ آ -> ا`) والتاء المربوطة (`ة -> ه`) والألف المقصورة (`ى -> ي`) وإزالة التطويل يرفع نسبة استدعاء النتائج (`Recall`) في البحث المعجمي والدلالي دون تشويه النص الأصلي المعروض للمستخدم.

### 3. لماذا دمجت `BM25` مع `Dense Embeddings` عبر خوارزمية `RRF` (`k=60`)؟
- **Okapi BM25:** متفوق في المطابقة الحرفية الدقيقة للمصطلحات والأكواد والأرقام (مثل `PDPL`، `HNSW`، `85 C`).
- **Dense Cosine Similarity:** متفوق في التقاط السياق والتشابه اللفظي الجزئي.
- **Reciprocal Rank Fusion (RRF):** تدمج ترتيب القائمتين بدون الحاجة لمعايرة أوزان الدرجات المختلفة (`1 / (60 + rank)`), مما يمنح ترتيباً مستقراً ومتوازناً.

### 4. كيف يطبق المشروع ضوابط نظام حماية البيانات الشخصية السعودي (`PDPL`)؟
- تمر جميع المستندات والاستعلامات على طبقة `guardrails.py` قبل التخزين أو الفهرسة، فتكتشف وتُخفي أرقام الهوية الوطنية/الإقامة، والآيبان السعودي (`SA...`)، وأرقام الجوال (`05...` / `+9665...`)، والبريد الإلكتروني، وترفع تصنيف المستند تلقائياً إلى `CONFIDENTIAL_PII`، مع حظر محاولات حقن الأوامر (`Prompt/SQL Injection`) وتسجيل كل عملية في `query_audit_logs`.

### 5. كيف يمنع النظام التكرار وما دور `faithfulness_score`؟
- يُحسب بصمة `SHA-256` للمحتوى لتجاوز إعادة فهرسة نفس المستند (`deduplicated: true`)، وتُقاس درجة الموثوقية (`faithfulness_score`) بحساب نسبة الكلمات الجوهرية في الإجابة الموثّقة `[DocTitle - p.X]` الموجودة فعلياً في المقاطع المسترجعة للحد من الهلوسة.
