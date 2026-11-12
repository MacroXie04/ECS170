"""Classify a review or continue a three-word prompt with a saved Stage 4 model.

Examples (from the repository root):
    python script/stage_4_script/predict_stage4.py RUN/checkpoints/best.pt --text "A wonderful movie."
    python script/stage_4_script/predict_stage4.py RUN/checkpoints/best.pt --text "What did the" --temperature 0.8 --seed 1704

Classification requires the run's unmodified config.json beside the checkpoint
or in its parent run directory, to reuse the training sequence-length limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from local_code.stage_4_code.Dataset_Loader import BOS, EOS, UNK, Vocabulary, tokenize
from local_code.stage_4_code.Method_RNN import SequenceModel, generate


def _device(value: str) -> torch.device:
    if value not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    value = ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available; use --device cpu")
    return torch.device(value)


def _run_config(checkpoint: Path, saved: dict) -> dict | None:
    candidates = (checkpoint.parent / "config.json", checkpoint.parent.parent / "config.json")
    config_path = next((p for p in candidates if p.is_file()), None)
    if config_path is None:
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "config_hash" in saved:
        canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False)
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if actual != saved["config_hash"]:
            raise ValueError("config.json does not match this checkpoint; restore its original run configuration")
    if config.get("model") != saved["model_kwargs"]:
        raise ValueError("Model settings in config.json do not match the checkpoint")
    return config


def predict(checkpoint, text: str, *, device: str = "auto", temperature: float = 0.0,
            top_k: int = 20, seed: int = 1704, max_new_tokens: int = 40) -> dict:
    """Return JSON-serializable inference results without changing saved files.

    Generation preserves the supplied prefix's spelling, capitalization, and
    punctuation in its displayed output. Unknown prompt tokens are rejected,
    rather than silently pretending the model understood them. Classification
    reports OOV tokens and truncation explicitly.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide nonempty input text")
    checkpoint = Path(checkpoint).expanduser().resolve()
    selected_device = _device(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required = {"model", "model_kwargs", "vocab"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError("Expected a Stage 4 best.pt checkpoint containing model, model_kwargs, and vocab")
    vocab = Vocabulary.from_dict(saved["vocab"])
    kwargs = saved["model_kwargs"]
    if kwargs.get("vocab_size") != len(vocab):
        raise ValueError("Checkpoint vocabulary length does not match the model")
    task = kwargs.get("task")
    if task not in ("classification", "generation"):
        raise ValueError("Checkpoint must identify its classification or generation task")
    config = _run_config(checkpoint, saved)
    words = tokenize(text)
    if not words:
        raise ValueError("Input has no tokens after text cleaning")
    # Constructing a module should not consume the caller's global CPU RNG.
    with torch.random.fork_rng(devices=[]):
        model = SequenceModel(**kwargs)
    model.load_state_dict(saved["model"], strict=True)
    model.to(selected_device).eval()
    common = {"task": task, "cell": kwargs["cell"], "device": str(selected_device),
              "checkpoint_epoch": saved.get("epoch")}

    if task == "classification":
        if config is None:
            raise ValueError("Classification needs the original config.json next to best.pt or its checkpoints directory")
        max_length = config.get("classification_max_length")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("config.json must contain a positive classification_max_length")
        all_ids = vocab.encode(text)
        used = all_ids[:max_length]
        ids = torch.tensor([used], dtype=torch.long, device=selected_device)
        with torch.inference_mode():
            probabilities = model(ids, torch.tensor([len(used)], device=selected_device))[0].float().softmax(-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Model returned non-finite probabilities")
        negative, positive = probabilities.cpu().tolist()
        label = int(positive > negative)
        oov_words = sorted({word for word in words[:max_length] if word not in vocab.stoi})
        return {**common, "label_id": label, "label": "positive" if label else "negative",
                "probabilities": {"negative": negative, "positive": positive},
                "input_token_count": len(all_ids), "used_token_count": len(used),
                "max_length": max_length, "truncated": len(all_ids) > max_length,
                "unknown_token_count": used.count(UNK), "unknown_tokens": oov_words}

    # Tokenizer punctuation is not counted as a natural-language word.
    word_count = sum(any(char.isalnum() for char in word) for word in words)
    if word_count != 3:
        raise ValueError(f"Generation requires exactly three words (punctuation does not count); received {word_count}")
    unknown = sorted({word for word in words if word not in vocab.stoi})
    if unknown:
        raise ValueError("Prompt contains tokens outside the saved vocabulary: " + ", ".join(unknown))
    prefix = text.strip()
    prefix_ids = vocab.encode(prefix, bos=True)
    output_ids = generate(model, prefix_ids, eos_id=EOS, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_k=top_k, seed=seed)
    continuation = vocab.decode(output_ids[len(prefix_ids):])
    separator = "" if not continuation or continuation[0] in ".,!?;:%)]}" else " "
    output_text = prefix + separator + continuation
    return {**common, "prompt": prefix, "text": output_text, "continuation": continuation,
            "prompt_word_count": word_count, "prompt_token_count": len(prefix_ids) - 1,
            "generated_token_count": len(output_ids) - len(prefix_ids),
            "ended_with_eos": output_ids[-1] == EOS, "unknown_tokens": [],
            "sampling": {"temperature": temperature, "top_k": top_k, "seed": seed,
                         "max_new_tokens": max_new_tokens}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Path to a run's checkpoints/best.pt")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="Review text or exactly three prompt words")
    inputs.add_argument("--text-file", type=Path, help="UTF-8 text file containing the input")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 selects greedy generation")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1704)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args(argv)
    try:
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        result = predict(args.checkpoint, text, device=args.device, temperature=args.temperature,
                         top_k=args.top_k, seed=args.seed, max_new_tokens=args.max_new_tokens)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
