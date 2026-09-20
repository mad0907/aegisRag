# ADR-002: Docling's structural+semantic HybridChunker, not fixed-size splitting

## Status
Accepted — implemented (`src/aegisrag/ingestion/pipeline.py`).

## Context
The simplest chunking strategy is fixed-size character/token windows with overlap. It's trivial
to implement and is what most tutorial-grade RAG pipelines use.

## Decision
Use Docling's `HybridChunker`, which chunks along document structure (headings, paragraphs,
list items, table boundaries) and packs chunks up to a token budget matched to the embedding
model's tokenizer, rather than cutting at arbitrary character offsets.

## Alternatives considered
- **Fixed-size character chunks with overlap**: simple, but routinely splits a sentence or a
  table row in half, and carries no heading/section context — a chunk that says "the three
  criteria are..." with no indication of *which* three criteria, from *which* section.
- **One chunk per page**: avoids mid-sentence splits but produces wildly uneven chunk sizes and
  loses within-page structure entirely.

## Consequences
- Every chunk retains its section heading path (`chunk.meta.headings`) and page number
  (`chunk.meta.doc_items[].prov[].page_no`), stored in `chunks.section_path` / `chunks.page_no` —
  this is *why* citations in this system can say "EthicsofAI.pdf, p.6, §3. Machines with Moral
  Status" instead of just a filename.
- `chunker.contextualize(chunk)` prepends heading context to the text actually sent for
  embedding, so the embedding reflects what section a chunk belongs to, not just its raw content.
- Cost: Docling's layout model is meaningfully slower than a regex-based splitter, and — as
  observed directly during this build — can be *very* slow on image/graphics-heavy PDFs (a
  second corpus PDF was dropped from this pass after Docling's OCR/layout stage took 30+ minutes
  on an 8-page slide-style document; see the README's Status section). This is a real, measured
  cost of the decision, not a hypothetical one.
