"""Wrapper around SemGuard's trained 1.3B classifier for line-level rollback.

SemGuard's classifier is a frozen-then-fine-tuned DeepSeek-Coder-1.3B-base
backbone (`AutoModel`) with a single linear head (`fc: Linear(2048, 1)`)
producing a logit on top of the mean-pooled (mask-weighted) hidden states of
the LAST hidden layer. Trained with `BCEWithLogitsLoss`; decision threshold
is `sigmoid(logit) >= 0.5` by default (see
`reproduce/algo/semguard/upstream/critic/models.py::DsClassBCE`).

This module exposes `SemGuardEvaluator`, which:

  1. Reconstructs that architecture in the main project's torch (so we don't
     depend on the SemGuard venv at runtime).
  2. Loads the saved state dict (full backbone + head — see report from the
     investigation step).
  3. Provides `async def score_prefix(prefix: str) -> float` returning the
     sigmoid probability that the prefix is on the right path.

Async surface: the heavy forward pass is dispatched to a thread via
`asyncio.to_thread` so the caller's event loop stays responsive. A single
inference lock (`asyncio.Lock`) serialises calls; the underlying model is not
re-entrant on one GPU.

Input format: matches SemGuard's training-time prompt template
(`DatasetCA.tokenize`):

    "\nQUESTION:\n{nl}\nINCOMPLETE CODE:\n{code}\nResult:\n"

We don't have the original NL prompt at evaluator-call time (the LM in our
demo doesn't expose it cleanly), so callers pass the full text directly. The
template is constructed via `format_prefix()` if both `nl` and `code` are
supplied; otherwise the prefix is fed as-is. This is the only intentional
deviation from the upstream input pipeline — the evaluator was trained on
template-formatted text so accuracy on unformatted prefixes will be lower,
but it still produces a calibrated [0, 1] probability.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Mirror upstream's input template so callers can format their (nl, code)
# pair the same way the model was trained on. The leading newline is part of
# the upstream template — do not strip.
PROMPT_TEMPLATE = "\nQUESTION:\n{nl}\nINCOMPLETE CODE:\n{code}\nResult:\n"

# Upstream's max_source_len. The classifier's training-time inputs were
# right-truncated to this length, so we mirror it at inference. The training
# code passes `max_length=MAX_LEN` (not //2) to the `ids_2` branch which is
# the only one the deepseekca/Ds model uses; we follow that.
DEFAULT_MAX_LEN = 600


def format_prefix(nl: str, code: str) -> str:
    """Render the (natural-language description, partial code) pair into the
    exact prompt template the classifier was trained on. Use this when you
    have both pieces; otherwise feed your raw prefix directly to
    `score_prefix`."""
    return PROMPT_TEMPLATE.format(nl=nl, code=code)


class SemGuardEvaluator:
    """Wraps SemGuard's trained DeepSeek-Coder-1.3B classifier.

    Lifecycle:
      1. `__init__` records paths/config but does NOT load weights — that
         would block the event loop on import.
      2. `await warmup()` loads the backbone + head onto `device` and sets
         the model to `eval()`. Call this once at startup.
      3. `await score_prefix(text)` returns the sigmoid probability that the
         prefix is "on the correct path" (the positive class during
         training). Caller decides what to do with values < threshold.
      4. `await close()` frees GPU memory by dropping the model and calling
         `torch.cuda.empty_cache()`.

    Threading: PyTorch inference is not async-friendly; we offload via
    `asyncio.to_thread`. An internal `asyncio.Lock` serialises requests so
    two concurrent callers don't race on the same GPU. For sub-second prefix
    scoring on a 1.3B model this is fine; if it becomes a bottleneck the lock
    can be removed and the model placed behind a `torch.cuda.Stream` per
    worker.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        backbone_path: Path | str,
        device: str = "cuda",
        threshold: float = 0.5,
        max_source_len: int = DEFAULT_MAX_LEN,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.backbone_path = Path(backbone_path)
        self.device = device
        self.threshold = float(threshold)
        self.max_source_len = int(max_source_len)

        # All heavy state — populated by warmup(). Kept as Any to avoid
        # importing torch at module import time (the project's other modules
        # tolerate missing torch in unit tests).
        self._model: Any = None
        self._tokenizer: Any = None
        self._pad_token_id: int | None = None
        self._lock = asyncio.Lock()
        self._loaded = False

    # ── public API ──────────────────────────────────────────────────────

    async def warmup(self) -> None:
        """Load backbone + classification head onto `self.device`. Idempotent:
        safe to call more than once (subsequent calls are no-ops)."""
        if self._loaded:
            return
        await asyncio.to_thread(self._load_sync)
        self._loaded = True

    async def score_prefix(self, prefix: str) -> float:
        """Return sigmoid(logit) in [0, 1] for `prefix` under the classifier.

        Raises `RuntimeError` if `warmup()` hasn't been called. The forward
        pass runs in a worker thread; concurrent callers are serialised via
        an internal lock."""
        if not self._loaded:
            raise RuntimeError(
                "SemGuardEvaluator.score_prefix called before warmup(). "
                "Call `await evaluator.warmup()` once at startup."
            )
        async with self._lock:
            return await asyncio.to_thread(self._score_sync, prefix)

    async def close(self) -> None:
        """Drop the model and free GPU memory. After close(), `score_prefix`
        will raise — call `warmup()` again to re-enable."""
        if not self._loaded:
            return
        async with self._lock:
            await asyncio.to_thread(self._close_sync)
        self._loaded = False

    # ── sync internals (always run in a worker thread) ──────────────────

    def _load_sync(self) -> None:
        # Imported lazily so import-time failures (no torch / no transformers)
        # don't break unrelated modules that import `soundcode.semguard_eval`
        # just to type-check the class.
        try:
            import torch  # noqa: F401
            import torch.nn as nn
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "SemGuardEvaluator requires torch + transformers in the main "
                f"project env (uv add transformers torch). Original error: {e!r}"
            ) from e

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SemGuard checkpoint not found: {self.checkpoint_path}. "
                "Has the training run finished? Expected an epoch-N "
                "model_<N>.bin file (e.g. "
                "reproduce/results/semguard/train_*/.../checkpoinss/0/model_0.bin)."
            )
        if not self.backbone_path.exists():
            raise FileNotFoundError(
                f"Backbone directory not found: {self.backbone_path}. "
                "Pass `backbone_path` pointing at a local DeepSeek-Coder-1.3B "
                "HF snapshot (e.g. ~/hf-models/deepseek-coder-1.3b-base)."
            )

        backbone_str = str(self.backbone_path)
        log.info("SemGuardEvaluator: loading backbone from %s", backbone_str)
        # `trust_remote_code=True` mirrors upstream's call in
        # `build_or_load_gen_model`. DeepSeek-Coder-1.3B-base doesn't actually
        # use any remote code, but the upstream model class was constructed
        # with this flag so we match it byte-for-byte.
        config = AutoConfig.from_pretrained(backbone_str, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(backbone_str, trust_remote_code=True)
        if tokenizer.pad_token is None:
            # Mirrors upstream's `tokenizer.pad_token = tokenizer.eos_token`
            # set in `build_or_load_gen_model` — without this the right-padded
            # input attention mask will mis-compute and the mean pool will be
            # off by a few tokens.
            tokenizer.pad_token = tokenizer.eos_token

        backbone = AutoModel.from_pretrained(backbone_str, trust_remote_code=True)
        dim_text = config.hidden_size

        # Reconstruct the DsClassBCE architecture verbatim.
        class _DsClassifier(nn.Module):
            def __init__(self, backbone: nn.Module, dim: int) -> None:
                super().__init__()
                self.model = backbone
                self.fc = nn.Linear(dim, 1)

            def forward(self, ids, mask):
                # Right-pads handled by `mask`. Mean-pool the last hidden
                # state over the unmasked positions, then project to 1 logit.
                enc = self.model(ids, attention_mask=mask).last_hidden_state
                mask_sum = mask.sum(dim=-1, keepdims=True)
                pooled = enc.masked_fill(mask.unsqueeze(-1) == 0, 0).sum(dim=1) / mask_sum
                return self.fc(pooled)

        model = _DsClassifier(backbone, dim_text)

        log.info("SemGuardEvaluator: loading checkpoint from %s",
                 self.checkpoint_path)
        # weights_only=False because the upstream checkpoint is a plain
        # `model.state_dict()` dump and might (depending on torch version)
        # default to a stricter loader.
        state_dict = torch.load(
            str(self.checkpoint_path), map_location="cpu", weights_only=False,
        )
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Expected a state_dict in {self.checkpoint_path}, got "
                f"{type(state_dict).__name__}. The upstream training script "
                "saves `model.state_dict()` — was this checkpoint produced "
                "by a different pipeline (e.g. PEFT / accelerate)?"
            )

        # Sanity-check: the checkpoint must contain BOTH the backbone keys
        # (`model.*`) and the classification head (`fc.weight`, `fc.bias`).
        has_backbone = any(k.startswith("model.") for k in state_dict)
        has_head = "fc.weight" in state_dict and "fc.bias" in state_dict
        if not has_head:
            raise RuntimeError(
                f"Checkpoint {self.checkpoint_path} is missing the "
                "classification head (`fc.weight`/`fc.bias`). Was this saved "
                "from a different model class? Keys sample: "
                f"{list(state_dict.keys())[:5]}"
            )
        if not has_backbone:
            warnings.warn(
                f"Checkpoint {self.checkpoint_path} has no `model.*` keys — "
                "loading head only; backbone weights will stay at the "
                "AutoModel.from_pretrained init. Scores will be unreliable.",
                stacklevel=2,
            )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Log up to 5 to keep output sane.
            log.warning("SemGuard: missing %d keys when loading state_dict; "
                        "first 5: %s", len(missing), missing[:5])
        if unexpected:
            log.warning("SemGuard: %d unexpected keys when loading; first 5: %s",
                        len(unexpected), unexpected[:5])

        model.eval()
        # Move to device. The backbone (loaded via AutoModel.from_pretrained
        # without an explicit dtype) inherits the config.torch_dtype — for
        # deepseek-coder-1.3b this is bfloat16. The freshly-constructed
        # `fc: nn.Linear` defaults to float32, which causes a dtype mismatch
        # at the matmul. Cast the WHOLE model uniformly to the backbone's
        # dtype after loading weights, so head + body agree. (The state_dict
        # also stores fc weights at float32; .to(dtype) downcasts in place,
        # which is fine for a single Linear with ~2k params.)
        backbone_dtype = next(backbone.parameters()).dtype
        model.to(device=self.device, dtype=backbone_dtype)

        self._model = model
        self._tokenizer = tokenizer
        self._pad_token_id = tokenizer.pad_token_id

        log.info("SemGuardEvaluator: ready on %s (dim=%d, threshold=%.2f)",
                 self.device, dim_text, self.threshold)

    def _score_sync(self, prefix: str) -> float:
        import torch

        if self._model is None or self._tokenizer is None:
            # Defensive — `score_prefix` already checks `_loaded`, but
            # `_loaded` could conceivably be flipped externally.
            raise RuntimeError("SemGuard model not loaded")

        # Match the upstream `ids_2` tokenization path: right-padded up to
        # max_source_len. Truncation at the END (which is the upstream
        # default) loses the most-recent prefix — for a partial program
        # that's a problem since the line we want to judge IS the most recent
        # text. We override the tokenizer's truncation side to LEFT so the
        # last `max_source_len` tokens are kept. This is an intentional
        # deviation from training-time tokenization (training used right-
        # truncation) but is the correct behaviour for online prefix scoring.
        # Note: upstream calls `tokenizer.encode_plus(...)` which is
        # deprecated in newer transformers; calling the tokenizer directly
        # is the supported replacement and does the same thing.
        prev_trunc_side = getattr(self._tokenizer, "truncation_side", "right")
        self._tokenizer.truncation_side = "left"
        try:
            enc = self._tokenizer(
                prefix,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
                max_length=self.max_source_len,
                return_token_type_ids=False,
                return_tensors="pt",
            )
        finally:
            self._tokenizer.truncation_side = prev_trunc_side
        ids = enc["input_ids"].to(self.device, dtype=torch.long)
        # Upstream computes mask as `ids.ne(pad_token_id)`. We do the same
        # (rather than trust `attention_mask` from the tokenizer) so the
        # behaviour matches training.
        mask = ids.ne(self._pad_token_id)

        with torch.no_grad():
            logit = self._model(ids, mask)
        # Single-example forward → shape (1, 1) (batch, 1). Squeeze to scalar.
        prob = torch.sigmoid(logit.view(-1)).item()
        # Clamp to [0, 1] defensively against fp drift.
        return max(0.0, min(1.0, float(prob)))

    def _close_sync(self) -> None:
        import torch

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._pad_token_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
