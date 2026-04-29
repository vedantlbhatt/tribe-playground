#!/usr/bin/env python3
"""Literally the docs: inference in 5 lines (+ print).

    python five_lines_infer.py path/to/clip.mp4

Snippet (same as NForge “Get Started”)::

    from nforge import NForgeModel
    model = NForgeModel.from_pretrained("facebook/tribev2")
    events = model.get_events_dataframe(video_path="clip.mp4")
    preds, segments = model.predict(events)
"""

import sys

from nforge import NForgeModel

video_path = sys.argv[1] if len(sys.argv) > 1 else "clip.mp4"
model = NForgeModel.from_pretrained("facebook/tribev2")
events = model.get_events_dataframe(video_path=video_path)
preds, segments = model.predict(events)

print(f"Predicted {preds.shape[0]} segments x {preds.shape[1]} vertices")
