"""FastAPI service: /answer, /search, a thin UI, and /metrics.

Routes depend on the lazily-built Retriever and Answerer via FastAPI dependency
injection, so tests override them with fakes and never touch a backend.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel

from app import __version__
from app.generation.answerer import Answerer
from app.retrieval.retriever import Retriever
from app.serving.deps import get_answerer, get_retriever
from app.serving.tools import answer_to_dict, chunks_to_dicts

_ANSWERS = Counter("sec_rag_answers_total", "Answers produced", ["refused"])

_UI = """<!doctype html>
<html><head><meta charset="utf-8"><title>SEC RAG</title>
<style>
 body{font:16px system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}
 input,textarea,button{font:inherit} textarea{width:100%;height:5rem}
 .cite{color:#555;font-size:.85rem;margin:.25rem 0}
 #out{white-space:pre-wrap;margin-top:1rem;border-top:1px solid #ddd;padding-top:1rem}
</style></head><body>
<h1>SEC RAG</h1>
<p>Grounded, span-cited answers over SEC filings.</p>
<textarea id="q" placeholder="e.g. what are Apple's supply chain risks?"></textarea>
<p><input id="t" placeholder="ticker (optional), e.g. AAPL">
<button onclick="ask()">Ask</button></p>
<div id="out"></div>
<script>
async function ask(){
  const out=document.getElementById('out'); out.textContent='...';
  const r=await fetch('/answer',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({question:document.getElementById('q').value,
                         ticker:document.getElementById('t').value||null})});
  const a=await r.json();
  if(a.refused){out.textContent='Not supported by any filing. ('+a.reason+')';return;}
  let h=a.text+'\\n\\nConfidence: '+a.confidence;
  h+='\\n\\nCitations:';
  for(const c of a.citations){h+='\\n['+c.marker+'] '+c.ticker+' '+c.form+' '+c.section+
    ' ('+c.accession+(c.as_of?(', as of '+c.as_of):'')+')';}
  out.textContent=h;
}
</script></body></html>
"""


class AnswerRequest(BaseModel):
    question: str
    ticker: str | None = None
    form: str | None = None
    section: str | None = None


class SearchRequest(BaseModel):
    question: str
    ticker: str | None = None
    form: str | None = None
    section: str | None = None
    limit: int | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="SEC RAG", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _UI

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/search")
    def search(
        request: SearchRequest,
        retriever: Annotated[Retriever, Depends(get_retriever)],
    ) -> dict[str, object]:
        chunks = retriever.retrieve(
            request.question,
            ticker=request.ticker,
            form=request.form,
            section=request.section,
            limit=request.limit,
        )
        return {"results": chunks_to_dicts(chunks)}

    @app.post("/answer")
    def answer(
        request: AnswerRequest,
        answerer: Annotated[Answerer, Depends(get_answerer)],
    ) -> dict[str, object]:
        result = answerer.answer(
            request.question,
            ticker=request.ticker,
            form=request.form,
            section=request.section,
        )
        _ANSWERS.labels(refused=str(result.refused).lower()).inc()
        return answer_to_dict(result)

    return app


app = create_app()
