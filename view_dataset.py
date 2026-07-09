#!/usr/bin/env python3
"""
Browse a curated TriSearch dataset (from generate_datasets.py).

  python3 view_dataset.py
  python3 view_dataset.py --dataset-dir models/data/trisearch-v1 --port 7861

Shows image, domain tag, captions, query, and unrelated_query with paging.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from trisearch_data_format import DEFAULT_DATASET_ROOT, load_dataset_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--max-load", type=int, default=None,
                   help="Only load first N rows into memory (default: all).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def _card_html(rec: dict, index: int, total: int) -> str:
    caps = "".join(
        f"<li>{html.escape(str(c))}</li>" for c in rec.get("captions") or []
    )
    domain = html.escape(str(rec.get("domain", "")))
    source = html.escape(str(rec.get("source", "")))
    q = html.escape(str(rec.get("query", "")))
    uq = html.escape(str(rec.get("unrelated_query", "")))
    rid = html.escape(str(rec.get("id", "")))
    return f"""
    <div style="font-family:system-ui,sans-serif;max-width:720px">
      <div style="color:#666;margin-bottom:8px">
        Item <b>{index + 1}</b> / {total} · id <code>{rid}</code>
      </div>
      <div style="margin-bottom:8px">
        <span style="background:#e8f0fe;padding:2px 8px;border-radius:4px">{domain}</span>
        <span style="color:#666;margin-left:8px">{source}</span>
      </div>
      <h3 style="margin:12px 0 4px">Captions</h3>
      <ol style="margin-top:0">{caps}</ol>
      <h3 style="margin:12px 0 4px">Query (should find this image)</h3>
      <p style="background:#f0fff4;padding:8px;border-radius:6px">{q}</p>
      <h3 style="margin:12px 0 4px">Unrelated query (distractor)</h3>
      <p style="background:#fff5f5;padding:8px;border-radius:6px">{uq}</p>
    </div>
    """


def build_viewer(records: list[dict]):
    """Build Gradio Blocks UI. Avoid gr.State constants as event inputs (Gradio 5 bug)."""
    import gradio as gr

    n = len(records)

    def show(index):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        index = max(0, min(index, n - 1))
        rec = records[index]
        return rec["image"], _card_html(rec, index, n), index

    def go_prev(index):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        return show(index - 1)

    def go_next(index):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        return show(index + 1)

    with gr.Blocks(title="TriSearch dataset viewer") as demo:
        gr.Markdown(
            f"## TriSearch dataset viewer\n"
            f"**{n:,}** items loaded"
        )
        with gr.Row():
            prev_btn = gr.Button("← Prev")
            next_btn = gr.Button("Next →")
            index_in = gr.Number(value=0, label="Index", precision=0)
            go_btn = gr.Button("Go")
        image = gr.Image(type="pil", label="Image (1024×1024)")
        meta = gr.HTML()
        # Keep index only as a Number (visible); do not pass gr.State(const)
        # into .click() — Gradio 5.12 get_api_info crashes on that pattern.
        index_state = gr.Number(value=0, visible=False, precision=0)

        outputs = [image, meta, index_state, index_in]

        def show_sync(index):
            img, html_card, idx = show(index)
            return img, html_card, idx, idx

        def prev_sync(index):
            img, html_card, idx = go_prev(index)
            return img, html_card, idx, idx

        def next_sync(index):
            img, html_card, idx = go_next(index)
            return img, html_card, idx, idx

        # api_name=False: Gradio 5.12 get_api_info() crashes on some event
        # schemas (TypeError: argument of type 'bool' is not iterable), which
        # breaks the browser homepage. Keep events UI-only.
        _evt = dict(api_name=False)
        demo.load(lambda: show_sync(0), outputs=outputs, **_evt)
        go_btn.click(show_sync, inputs=[index_in], outputs=outputs, **_evt)
        index_in.submit(show_sync, inputs=[index_in], outputs=outputs, **_evt)
        prev_btn.click(prev_sync, inputs=[index_state], outputs=outputs, **_evt)
        next_btn.click(next_sync, inputs=[index_state], outputs=outputs, **_evt)

    return demo


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.exists():
        raise SystemExit(
            f"Dataset not found at {args.dataset_dir}. "
            f"Run: python3 generate_datasets.py --preview --skip-query-generation"
        )

    print(f"Loading {args.dataset_dir} ...", flush=True)
    records = load_dataset_records(args.dataset_dir, max_samples=args.max_load)
    if not records:
        raise SystemExit("Dataset is empty.")
    print(f"  {len(records):,} records", flush=True)

    demo = build_viewer(records)

    # Fail fast if Gradio cannot build API metadata (common with bad State wiring).
    try:
        demo.get_api_info()
    except Exception as exc:
        raise SystemExit(
            f"Gradio UI failed API introspection (viewer would be broken in browser): {exc}"
        ) from exc

    print(f"Open viewer at http://{args.host}:{args.port}", flush=True)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
