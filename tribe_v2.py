#!/usr/bin/env python3
"""Multimodal brain-encoding inference (TRIBE v2 weights on Hugging Face).

You can run the stack through **NForge** (recommended extras: ROI attention,
streaming, attribution) or the upstream **tribev2** package — same checkpoint
and same ``from_pretrained`` / ``get_events_dataframe`` / ``predict`` flow.

PACE Phoenix notes
------------------
- Needs **Python 3.11+** and a **recent PyTorch** (see ``nforge`` / ``tribev2`` on PyPI/GitHub).
  The module ``pytorch/2.1.0`` may be too old; use a conda env with matching versions.
- **Do not** ``pip install nforge`` from PyPI — that name is a different project.
  Brain NForge: ``pip install 'git+https://github.com/kairowandev/nforge.git'``
  If import fails with ``No module named 'lightning'``: ``pip install lightning``
  Or Meta TRIBE: ``pip install 'git+https://github.com/facebookresearch/tribev2.git'``
- Hugging Face weights download on first run; set ``HF_TOKEN`` if the hub requires it.

Example (after scp to cluster)::

    python tribe_v2.py --video ./clip.mp4 --output ./preds.npz
    python tribe_v2.py --backend nforge --video ./clip.mp4
    python tribe_v2.py --backend tribe --audio ./audio.wav --cache-dir ./cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _import_nforge_model():
    """Import brain-encoding NForge (kairowandev). PyPI package *nforge* is unrelated."""

    try:
        import nforge as _nf
    except ImportError as e:
        logger.error(
            "nforge import failed (%s: %s). Remove ./nforge.py if it shadows the package.",
            type(e).__name__,
            e,
        )
        raise ImportError(
            f"Brain NForge did not import ({type(e).__name__}: {e}). "
            "Fix: pip uninstall nforge -y; pip install 'git+https://github.com/kairowandev/nforge.git' "
            "inside the same conda env. Or use --backend tribe."
        ) from e
    cls = getattr(_nf, "NForgeModel", None)
    if cls is None:
        raise ImportError(
            "Installed 'nforge' has no NForgeModel (wrong PyPI package?). "
            "Fix: pip uninstall nforge -y && pip install 'git+https://github.com/kairowandev/nforge.git'"
        )
    return cls


def _load_model_class(backend: str) -> tuple[type, str]:
    """Return (ModelClass, resolved_backend_label). *backend* is auto|nforge|tribe."""

    if backend == "nforge":
        return _import_nforge_model(), "nforge"

    if backend == "tribe":
        try:
            from tribev2 import TribeModel
        except ImportError as e:
            raise ImportError(
                "Backend 'tribe' requires: "
                "pip install 'git+https://github.com/facebookresearch/tribev2.git'"
            ) from e
        return TribeModel, "tribe"

    # auto: prefer brain NForge when installed (same weights as tribev2 HF checkpoint)
    try:
        return _import_nforge_model(), "nforge"
    except ImportError:
        pass
    try:
        from tribev2 import TribeModel

        return TribeModel, "tribe"
    except ImportError as e:
        raise ImportError(
            "Install one of: "
            "pip install 'git+https://github.com/kairowandev/nforge.git'   OR   "
            "pip install 'git+https://github.com/facebookresearch/tribev2.git'"
        ) from e


def _patch_tribev2_yaml_for_nforge(yaml_text: str) -> tuple[str, bool]:
    """Meta HF config uses TribeSurfaceProjector; NForge only registers NforgeSurfaceProjector."""

    if "TribeSurfaceProjector" not in yaml_text:
        return yaml_text, False
    patched = yaml_text.replace("name: TribeSurfaceProjector", "name: NforgeSurfaceProjector")
    return patched, True


def _nforge_from_pretrained(
    ModelClass: type,
    checkpoint: str,
    *,
    cache_folder: str,
    device: str,
    checkpoint_name: str = "best.ckpt",
):
    """Load NForge with Hub/local TRIBE configs that reference Meta-only projector names."""

    ck = Path(checkpoint)
    fp_kw = {"cache_folder": cache_folder, "device": device}

    if ck.is_dir() and (ck / "config.yaml").is_file():
        raw = (ck / "config.yaml").read_text(encoding="utf-8")
        fixed, did = _patch_tribev2_yaml_for_nforge(raw)
        if not did:
            return ModelClass.from_pretrained(checkpoint, **fp_kw)
        td = Path(tempfile.mkdtemp(prefix="tribe_nforge_cfg_"))
        (td / "config.yaml").write_text(fixed, encoding="utf-8")
        src_ckpt = ck / checkpoint_name
        if not src_ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {src_ckpt}")
        os.symlink(src_ckpt.resolve(), td / checkpoint_name, target_is_directory=False)
        logger.info("NForge: patched config (TribeSurfaceProjector → NforgeSurfaceProjector), loading from %s", td)
        return ModelClass.from_pretrained(str(td), **fp_kw)

    if not ck.exists():
        from huggingface_hub import hf_hub_download

        repo_id = str(checkpoint)
        config_path = Path(hf_hub_download(repo_id, "config.yaml"))
        raw = config_path.read_text(encoding="utf-8")
        fixed, did = _patch_tribev2_yaml_for_nforge(raw)
        if not did:
            return ModelClass.from_pretrained(repo_id, **fp_kw)
        td = Path(tempfile.mkdtemp(prefix="tribe_nforge_cfg_"))
        (td / "config.yaml").write_text(fixed, encoding="utf-8")
        ckpt_path = Path(hf_hub_download(repo_id, checkpoint_name))
        os.symlink(ckpt_path.resolve(), td / checkpoint_name, target_is_directory=False)
        logger.info("NForge: patched HF tribev2 config for this stack; loading from %s", td)
        return ModelClass.from_pretrained(str(td), **fp_kw)

    return ModelClass.from_pretrained(checkpoint, **fp_kw)


def _run_nvidia_smi() -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.stdout:
            print(out.stdout, end="" if out.stdout.endswith("\n") else "\n")
        if out.stderr:
            print(out.stderr, file=sys.stderr, end="" if out.stderr.endswith("\n") else "\n")
        if out.returncode != 0:
            logger.warning("nvidia-smi exited with code %s", out.returncode)
    except FileNotFoundError:
        logger.warning("nvidia-smi not found on PATH")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")


def _log_torch_device() -> None:
    import torch

    logger.info("torch %s", torch.__version__)
    logger.info("torch.cuda.is_available() = %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("torch.cuda.device_count() = %s", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            logger.info("  [%s] %s", i, torch.cuda.get_device_name(i))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRIBE v2 inference (multimodal → cortical predictions)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Video file (.mp4, .avi, .mkv, .mov, .webm)")
    src.add_argument("--audio", type=Path, help="Audio file (.wav, .mp3, .flac, .ogg)")
    src.add_argument("--text", type=Path, help="Text file (.txt); TTS + transcription pipeline")

    p.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "nforge", "tribe"),
        help="Inference package: nforge (pip install nforge), tribe (Meta repo), or auto (nforge if installed)",
    )
    p.add_argument(
        "--checkpoint",
        default="facebook/tribev2",
        help="Hugging Face repo id or local checkpoint directory",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache"),
        help="Feature cache directory (created if missing)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tribe_predictions.npz"),
        help="Output .npz path for predictions array",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Device for the loaded checkpoint",
    )
    p.add_argument(
        "--no-nvidia-smi",
        action="store_true",
        help="Skip printing nvidia-smi at startup",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm in predict()",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.perf_counter()

    _log_torch_device()
    if not args.no_nvidia_smi:
        _run_nvidia_smi()

    try:
        import numpy as np
        ModelClass, resolved = _load_model_class(args.backend)
    except ImportError as e:
        logger.error("%s", e)
        if e.__cause__ is not None:
            logger.error("Caused by: %s: %s", type(e.__cause__).__name__, e.__cause__)
        return 1

    logger.info("Using backend: %s", resolved)

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {"cache_folder": str(args.cache_dir), "device": args.device}
    if args.video is not None:
        kwargs["video_path"] = str(args.video.resolve())
    elif args.audio is not None:
        kwargs["audio_path"] = str(args.audio.resolve())
    else:
        kwargs["text_path"] = str(args.text.resolve())

    logger.info("Loading model from %s …", args.checkpoint)
    if resolved == "nforge":
        model = _nforge_from_pretrained(
            ModelClass,
            args.checkpoint,
            cache_folder=str(args.cache_dir),
            device=args.device,
        )
    else:
        model = ModelClass.from_pretrained(
            args.checkpoint,
            cache_folder=str(args.cache_dir),
            device=args.device,
        )

    logger.info("Building events dataframe …")
    df = model.get_events_dataframe(**{k: v for k, v in kwargs.items() if k.endswith("_path")})

    logger.info("Running predict …")
    preds, segments = model.predict(df, verbose=not args.quiet)

    meta = {
        "backend": resolved,
        "checkpoint": args.checkpoint,
        "input": {k: str(v) for k, v in kwargs.items() if k.endswith("_path")},
        "preds_shape": list(preds.shape),
        "n_segments": len(segments),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        preds=preds,
        meta_json=np.array([json.dumps(meta)]),
    )
    logger.info("Wrote %s  preds=%s  segments=%s", args.output, preds.shape, len(segments))
    logger.info("Done in %.2fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
