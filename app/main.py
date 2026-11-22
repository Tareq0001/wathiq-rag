"""FastAPI entrypoint, ETL ingestion endpoints, hybrid RAG API, and RTL/LTR workbench UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.etl import chunk_document, compute_embedding, extract_pdf_pages, sha256_hash
from app.guardrails import inspect_document
from app.rag import GuardrailBlockedError, WathiqRAGService
from app.retriever import HybridRetriever
from app.storage import WathiqStorage

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "wathiq.db")

_SEED_DOCUMENTS: tuple[dict[str, str], ...] = (
    {
        "title": "Saudi PDPL & Enterprise Data Governance Policy",
        "source_type": "policy",
        "text": (
            "سياسة حماية البيانات الشخصية (نظام حماية البيانات الشخصية السعودي PDPL) وحوكمة البيانات المؤسسية: "
            "تلتزم المنشأة بحماية البيانات الشخصية وفق نظام حماية البيانات الشخصية الصادر عن سدايا (SDAIA). "
            "يجب تطبيق تقنيات إخفاء الهوية والتنقيح التلقائي (PII Redaction) على أرقام الهوية الوطنية والإقامة، "
            "وأرقام الحسابات المصرفية الدولية (IBAN)، وأرقام الجوال، وعناوين البريد الإلكتروني قبل الفهرسة أو العرض. "
            "Enterprise Data Governance requires strict classification into INTERNAL and CONFIDENTIAL_PII tiers, "
            "SHA-256 content deduplication, and immutable query audit logging for every retrieval transaction."
        ),
    },
    {
        "title": "PostgreSQL pgvector & Hybrid ETL Pipeline Standard",
        "source_type": "standard",
        "text": (
            "معيار خط أنابيب الاستخراج والتحويل والتحميل (ETL) وفهرسة المتجهات عبر PostgreSQL pgvector: "
            "يعتمد المحرك على استخراج نصوص ملفات PDF صفحة بصفحة باستخدام PyMuPDF ثم تطبيع النصوص العربية "
            "(إزالة التطويل وتوحيد الهمزات والتاء المربوطة والألف المقصورة). "
            "Chunking uses a 450-character sliding window with 90-character overlap aligned to word boundaries. "
            "Hybrid retrieval combines Okapi BM25 lexical ranking with 256-dimensional L2-normalized dense embeddings "
            "using pgvector HNSW cosine distance and Reciprocal Rank Fusion (RRF, k=60) for balanced precision and recall."
        ),
    },
    {
        "title": "Industrial Telemetry & Predictive Maintenance SOP",
        "source_type": "sop",
        "text": (
            "إجراء التشغيل القياسي للقياس الصناعي عن بعد والصيانة التنبؤية (Predictive Maintenance SOP): "
            "تراقب المستشعرات الصناعية الاهتزازات ودرجة حرارة المحامل ومعدل استهلاك الطاقة الكهربائية لحظيا. "
            "When vibration RMS exceeds 4.5 mm/s or bearing temperature rises above 85 degrees Celsius, "
            "an automated maintenance ticket is triggered before unplanned downtime occurs. "
            "All telemetry logs and anomaly reports are archived with page citations for root-cause engineering audits."
        ),
    },
)


class IngestTextRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    text: str = Field(..., min_length=1)
    source_type: str = Field(default="text", max_length=50)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


def _ingest_pages(
    storage: WathiqStorage,
    title: str,
    source_type: str,
    pages: list[dict],
) -> dict:
    """Apply PDPL guardrails, SHA-256 deduplication, chunking, embeddings, and storage."""
    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text found in input.")

    redacted_pages: list[dict] = []
    total_pii = 0
    sensitivity = "INTERNAL"

    for page in pages:
        report = inspect_document(str(page.get("text", "")))
        if report.redacted_text.strip():
            redacted_pages.append(
                {
                    "page_number": int(page.get("page_number", 1)),
                    "text": report.redacted_text,
                }
            )
        total_pii += report.pii_redacted_count
        if report.sensitivity_level == "CONFIDENTIAL_PII":
            sensitivity = "CONFIDENTIAL_PII"

    if not redacted_pages:
        raise HTTPException(status_code=400, detail="Document contains no indexable text.")

    canonical_text = "\n\n".join(p["text"] for p in redacted_pages)
    content_hash = sha256_hash(canonical_text)

    existing = storage.document_exists_by_hash(content_hash)
    if existing is not None:
        return {
            "status": "duplicate",
            "deduplicated": True,
            "document": existing,
        }

    chunks = chunk_document(redacted_pages, chunk_size=450, overlap=90)
    if not chunks:
        raise HTTPException(status_code=400, detail="Failed to produce chunks from document.")

    embeddings = [compute_embedding(c["content"]) for c in chunks]
    saved_doc = storage.save_document_with_chunks(
        title=title.strip(),
        source_type=source_type.strip() or "text",
        content_hash=content_hash,
        sensitivity=sensitivity,
        pii_redacted=total_pii,
        page_count=len(redacted_pages),
        chunks=chunks,
        embeddings=embeddings,
    )
    return {
        "status": "ingested",
        "deduplicated": False,
        "document": saved_doc,
    }


def _seed_initial_corpus(storage: WathiqStorage) -> None:
    """Populate 3 enterprise bilingual documents if the corpus is empty."""
    if storage.list_documents():
        return
    for seed in _SEED_DOCUMENTS:
        _ingest_pages(
            storage=storage,
            title=seed["title"],
            source_type=seed["source_type"],
            pages=[{"page_number": 1, "text": seed["text"]}],
        )


def create_app(db_path: str | None = None) -> FastAPI:
    """Create and configure the Wathiq RAG FastAPI application."""
    resolved_db = db_path or os.getenv("WATHIQ_DB_PATH") or DEFAULT_DB_PATH

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        storage = WathiqStorage(resolved_db)
        storage.init_schema()
        _seed_initial_corpus(storage)
        rag_service = WathiqRAGService(storage=storage, retriever=HybridRetriever())
        application.state.storage = storage
        application.state.rag = rag_service
        try:
            yield
        finally:
            storage.close()

    application = FastAPI(
        title="Wathiq Enterprise Document ETL & Hybrid RAG Engine",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/api/v1/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "wathiq-rag",
            "version": "1.0.0",
        }

    @application.post("/api/v1/documents/ingest")
    def ingest_text(payload: IngestTextRequest) -> dict:
        storage: WathiqStorage = application.state.storage
        return _ingest_pages(
            storage=storage,
            title=payload.title,
            source_type=payload.source_type,
            pages=[{"page_number": 1, "text": payload.text}],
        )

    @application.post("/api/v1/documents/upload-pdf")
    async def upload_pdf(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
    ) -> dict:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        try:
            pages = extract_pdf_pages(pdf_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {exc}") from exc

        doc_title = (title or "").strip() or (file.filename or "Uploaded PDF").strip()
        storage: WathiqStorage = application.state.storage
        return _ingest_pages(
            storage=storage,
            title=doc_title,
            source_type="pdf",
            pages=pages,
        )

    @application.get("/api/v1/documents")
    def list_documents() -> dict:
        storage: WathiqStorage = application.state.storage
        return {"documents": storage.list_documents()}

    @application.post("/api/v1/query")
    def query_rag(payload: QueryRequest) -> dict:
        rag: WathiqRAGService = application.state.rag
        try:
            return rag.answer_query(query=payload.query, top_k=payload.top_k)
        except GuardrailBlockedError as exc:
            raise HTTPException(status_code=400, detail=exc.report.blocked_reason) from exc

    @application.get("/api/v1/metrics")
    def get_metrics() -> dict:
        storage: WathiqStorage = application.state.storage
        return storage.get_metrics()

    @application.get("/", response_class=HTMLResponse)
    def workbench() -> str:
        return _WORKBENCH_HTML

    return application


_WORKBENCH_HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>وَثِيق — Wathiq Enterprise Hybrid RAG Workbench</title>
  <style>
    :root {
      --bg: #0b1120;
      --panel: #111827;
      --panel-alt: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #0ea5e9;
      --accent-2: #10b981;
      --warn: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      background: #0f172a;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    header h1 { margin: 0; font-size: 1.3rem; }
    header p { margin: 2px 0 0; color: var(--muted); font-size: 0.9rem; }
    .container {
      max-width: 1260px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 1fr 1.2fr;
      gap: 20px;
    }
    @media (max-width: 960px) { .container { grid-template-columns: 1fr; } }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .card h2 { margin-top: 0; font-size: 1.05rem; color: var(--accent); }
    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 16px;
    }
    .stat {
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .stat .val { font-size: 1.25rem; font-weight: 700; direction: ltr; display: inline-block; }
    .stat .lbl { font-size: 0.78rem; color: var(--muted); }
    label { display: block; font-size: 0.85rem; color: var(--muted); margin: 8px 0 4px; }
    input[type="text"], input[type="number"], textarea {
      width: 100%;
      padding: 9px 11px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #0f172a;
      color: var(--text);
      font-family: inherit;
    }
    button {
      margin-top: 10px;
      padding: 9px 16px;
      border: none;
      border-radius: 6px;
      background: var(--accent);
      color: #082f49;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary { background: var(--accent-2); color: #052e16; }
    .status-msg { margin-top: 8px; font-size: 0.88rem; }
    .err { color: var(--danger); }
    .ok { color: var(--accent-2); }
    .answer-box {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      margin-top: 10px;
      white-space: pre-wrap;
    }
    .chunk-item, .doc-item, .log-item {
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      margin-top: 8px;
      font-size: 0.88rem;
    }
    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 0.78rem;
      color: var(--muted);
      margin-bottom: 4px;
      direction: ltr;
    }
    .badge {
      background: #0f172a;
      border: 1px solid var(--border);
      padding: 2px 7px;
      border-radius: 999px;
      font-size: 0.75rem;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>وَثِيق (Wathiq) — Enterprise Document ETL & Hybrid RAG Engine</h1>
      <p>PyMuPDF ETL • Arabic Normalization • Saudi PDPL Guardrails • BM25 + Cosine RRF</p>
    </div>
    <span class="badge" id="healthBadge">Checking API...</span>
  </header>

  <main class="container">
    <section>
      <div class="stats">
        <div class="stat"><div class="val" id="mDocs">0</div><div class="lbl">Documents</div></div>
        <div class="stat"><div class="val" id="mChunks">0</div><div class="lbl">Chunks</div></div>
        <div class="stat"><div class="val" id="mQueries">0</div><div class="lbl">Queries</div></div>
        <div class="stat"><div class="val" id="mFaith">0.0</div><div class="lbl">Avg Faithfulness</div></div>
        <div class="stat"><div class="val" id="mPii">0</div><div class="lbl">PII Redacted</div></div>
        <div class="stat"><div class="val" id="mBlocked">0</div><div class="lbl">Guardrail Blocks</div></div>
      </div>

      <div class="card">
        <h2>رفع مستند PDF / Upload PDF</h2>
        <form id="pdfForm">
          <label>عنوان اختياري / Optional Title</label>
          <input type="text" id="pdfTitle" placeholder="مثال: لائحة الموارد البشرية" />
          <label>ملف PDF</label>
          <input type="file" id="pdfFile" accept=".pdf,application/pdf" required />
          <button type="submit">تشغيل خط ETL وحفظ المستند</button>
        </form>
        <div id="pdfStatus" class="status-msg"></div>
      </div>

      <div class="card">
        <h2>إدخال نص مباشر / Ingest Raw Text</h2>
        <form id="textForm">
          <label>عنوان المستند / Title</label>
          <input type="text" id="txtTitle" required placeholder="مثال: سياسة الأمن السيبراني" />
          <label>المحتوى / Text (يدعم اكتشاف وتنقيح الهوية والآيبان والجوال)</label>
          <textarea id="txtBody" rows="4" required placeholder="أدخل نص المستند هنا..."></textarea>
          <button type="submit" class="secondary">فهرسة النص</button>
        </form>
        <div id="textStatus" class="status-msg"></div>
      </div>

      <div class="card">
        <h2>المستندات المفهرسة / Indexed Corpus</h2>
        <div id="docList"></div>
      </div>
    </section>

    <section>
      <div class="card">
        <h2>الاستعلام الهجين / Hybrid RAG Query (BM25 + Dense Cosine RRF)</h2>
        <form id="queryForm">
          <label>السؤال / Query (عربي أو إنجليزي)</label>
          <input type="text" id="queryInput" required placeholder="مثال: ما هي معايير حماية البيانات الشخصية PDPL؟" />
          <label>Top-K Chunks</label>
          <input type="number" id="topK" value="5" min="1" max="15" />
          <button type="submit">تنفيذ الاسترجاع الهجين والتوليد الموثق</button>
        </form>
        <div id="queryStatus" class="status-msg"></div>
        <div id="answerContainer" style="display:none;">
          <div class="meta-row" id="answerMeta"></div>
          <div class="answer-box" id="answerText"></div>
          <h3>المقاطع المسترجعة ودرجات RRF / Retrieved Chunks</h3>
          <div id="chunkResults"></div>
        </div>
      </div>

      <div class="card">
        <h2>سجل التدقيق الأمني / Query Audit Logs</h2>
        <div id="auditLogs"></div>
      </div>
    </section>
  </main>

  <script>
    async function refreshAll() {
      const [hRes, dRes, mRes] = await Promise.all([
        fetch('/api/v1/health'),
        fetch('/api/v1/documents'),
        fetch('/api/v1/metrics')
      ]);
      if (hRes.ok) {
        const h = await hRes.json();
        document.getElementById('healthBadge').textContent = 'API: ' + h.status.toUpperCase();
      }
      if (dRes.ok) {
        const d = await dRes.json();
        const docList = document.getElementById('docList');
        docList.innerHTML = d.documents.map(doc => `
          <div class="doc-item">
            <strong>${escapeHtml(doc.title)}</strong>
            <div class="meta-row">
              <span>type: ${escapeHtml(doc.source_type)}</span>
              <span>pages: ${doc.page_count}</span>
              <span>chunks: ${doc.chunk_count}</span>
              <span>sensitivity: ${escapeHtml(doc.sensitivity)}</span>
              <span>pii_redacted: ${doc.pii_redacted}</span>
            </div>
          </div>
        `).join('');
      }
      if (mRes.ok) {
        const m = await mRes.json();
        document.getElementById('mDocs').textContent = m.total_documents;
        document.getElementById('mChunks').textContent = m.total_chunks;
        document.getElementById('mQueries').textContent = m.total_queries;
        document.getElementById('mFaith').textContent = m.avg_faithfulness;
        document.getElementById('mPii').textContent = m.pii_redactions;
        document.getElementById('mBlocked').textContent = m.guardrail_blocks;
        document.getElementById('auditLogs').innerHTML = m.recent_queries.map(q => `
          <div class="log-item">
            <div>${escapeHtml(q.query_text)}</div>
            <div class="meta-row">
              <span>blocked: ${q.guardrail_blocked}</span>
              <span>pii: ${q.pii_redacted}</span>
              <span>faithfulness: ${q.faithfulness_score}</span>
              <span>latency: ${q.latency_ms} ms</span>
            </div>
          </div>
        `).join('') || '<div class="meta-row">No queries yet.</div>';
      }
    }

    function escapeHtml(str) {
      return String(str).replace(/[&<>"']/g, s => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[s]));
    }

    document.getElementById('pdfForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById('pdfFile');
      const status = document.getElementById('pdfStatus');
      if (!fileInput.files.length) return;
      const fd = new FormData();
      fd.append('file', fileInput.files[0]);
      const title = document.getElementById('pdfTitle').value.trim();
      if (title) fd.append('title', title);
      const res = await fetch('/api/v1/documents/upload-pdf', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) {
        status.className = 'status-msg err';
        status.textContent = data.detail || 'Upload failed';
      } else {
        status.className = 'status-msg ok';
        status.textContent = `Status: ${data.status} | chunks: ${data.document.chunk_count} | pii_redacted: ${data.document.pii_redacted}`;
        await refreshAll();
      }
    });

    document.getElementById('textForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const status = document.getElementById('textStatus');
      const res = await fetch('/api/v1/documents/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: document.getElementById('txtTitle').value,
          text: document.getElementById('txtBody').value,
          source_type: 'text'
        })
      });
      const data = await res.json();
      if (!res.ok) {
        status.className = 'status-msg err';
        status.textContent = data.detail || 'Ingest failed';
      } else {
        status.className = 'status-msg ok';
        status.textContent = `Status: ${data.status} | chunks: ${data.document.chunk_count} | pii_redacted: ${data.document.pii_redacted}`;
        await refreshAll();
      }
    });

    document.getElementById('queryForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const status = document.getElementById('queryStatus');
      const container = document.getElementById('answerContainer');
      const res = await fetch('/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: document.getElementById('queryInput').value,
          top_k: Number(document.getElementById('topK').value || 5)
        })
      });
      const data = await res.json();
      if (!res.ok) {
        status.className = 'status-msg err';
        status.textContent = data.detail || 'Query blocked';
        container.style.display = 'none';
        await refreshAll();
        return;
      }
      status.textContent = '';
      container.style.display = 'block';
      document.getElementById('answerMeta').textContent =
        `Latency: ${data.latency_ms} ms | Faithfulness: ${data.faithfulness_score} | PII Redacted: ${data.pii_redacted}`;
      document.getElementById('answerText').textContent = data.answer;
      document.getElementById('chunkResults').innerHTML = data.retrieved_chunks.map(c => `
        <div class="chunk-item">
          <div class="meta-row">
            <strong>${escapeHtml(c.doc_title)} (p.${c.page_number})</strong>
            <span>RRF: ${c.rrf_score}</span>
            <span>Dense: ${c.dense_score} (#${c.dense_rank})</span>
            <span>BM25: ${c.bm25_score} (#${c.bm25_rank})</span>
          </div>
          <div>${escapeHtml(c.content)}</div>
        </div>
      `).join('');
      await refreshAll();
    });

    refreshAll();
  </script>
</body>
</html>
"""

app = create_app()
