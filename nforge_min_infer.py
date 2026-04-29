#!/usr/bin/env python3
"""Same 5-line inference as the docs; argparse only fills paths and saves .npz.

Docs core::

    from nforge import NForgeModel
    model = NForgeModel.from_pretrained("facebook/tribev2")
    events = model.get_events_dataframe(video_path="clip.mp4")
    preds, segments = model.predict(events)

If ``from_pretrained("facebook/tribev2")`` fails on config validation, use
``tribe_v2.py`` (Hub yaml patch) instead.

``--audio-only`` is extra (Whisper skip); not in the 5-line snippet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="NForge: docs 5-line flow + optional .npz")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--audio", type=Path)
    src.add_argument("--text", type=Path)
    p.add_argument("--checkpoint", default="facebook/tribev2")
    p.add_argument("--cache-dir", type=Path, default=Path("./cache_nforge_min"))
    p.add_argument("--output", type=Path, default=Path("nforge_preds.npz"))
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--audio-only", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- inference in 5 lines (NForge docs) ---
    from nforge import NForgeModel

    model = NForgeModel.from_pretrained(
        args.checkpoint, cache_folder=str(args.cache_dir), device=args.device
    )
    if args.audio_only:
        if args.text is not None:
            raise SystemExit("--audio-only does not apply to --text")
        from nforge.inference.predictor import get_audio_and_text_events

        path = args.video if args.video is not None else args.audio
        typ = "Video" if args.video is not None else "Audio"
        events = get_audio_and_text_events(
            pd.DataFrame(
                [
                    {
                        "type": typ,
                        "filepath": str(path.resolve()),
                        "start": 0,
                        "timeline": "default",
                        "subject": "default",
                    }
                ]
            ),
            audio_only=True,
        )
    elif args.video is not None:
        events = model.get_events_dataframe(video_path=str(args.video))
    elif args.audio is not None:
        events = model.get_events_dataframe(audio_path=str(args.audio))
    else:
        events = model.get_events_dataframe(text_path=str(args.text))

    preds, segments = model.predict(events, verbose=not args.quiet)
    # --- end docs flow ---

    meta = {"checkpoint": args.checkpoint, "preds_shape": list(preds.shape), "n_segments": len(segments)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, preds=preds, meta_json=np.array([json.dumps(meta)]))
    print(f"Wrote {args.output}  preds={preds.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
