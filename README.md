# tiny-lora-finetune-demo

> **OSS / learning demo only.** This is a personal open-source teaching project by [vk-ai](https://github.com/vk-ai). It is **not** employer production software, is **not** affiliated with any employer, and must not be described as production fine-tuning infrastructure.

CPU-friendly **toy LoRA** on a tiny numpy classifier: frozen base weights, trainable low-rank adapters (`A`, `B`), a short SGD loop, and a **before vs after** eval harness (loss + accuracy). No Hugging Face downloads, no GPU, no `peft` — the adapter math is ~50 lines you can read.

## Why

LoRA papers and PEFT stacks are great, but for learning and CI you often want:

- **No multi-GB model downloads**
- **Seconds on CPU**, not minutes on GPU
- Explicit **rank / alpha** config and a pytest gate that after ≥ before

This repo is that slice.

## What’s inside

| Piece | Role |
|---|---|
| `src/tiny_lora/lora.py` | Toy `LoRALinear` + `merge_and_unload()` → `MergedLinear` (classic `α/r` or rsLoRA `α/√r`) |
| `src/tiny_lora/model.py` | Tiny frozen MLP + LoRA classification head |
| `src/tiny_lora/train.py` | SGD over `A`/`B` only |
| `src/tiny_lora/eval.py` | Before/after loss + accuracy harness |
| `configs/default.yaml` | LoRA `rank` / `alpha` / `scaling: classic\|rslora` |
| `evals/runner.py` | CI-friendly eval CLI + JSON report |
| `tests/` | Unit + train + harness (pytest, seconds on CPU) |
| `ci/github-actions.yml` | GitHub Actions workflow (copy to `.github/workflows/ci.yml` to enable CI) |

## Quickstart

```bash
git clone https://github.com/vk-ai/tiny-lora-finetune-demo.git
cd tiny-lora-finetune-demo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python examples/quickstart.py
python evals/runner.py
python -m tiny_lora
```

Example output:

```text
Tiny LoRA before/after eval
  LoRA rank=4  alpha=8.0
  trainable=140  frozen=643
  before  loss=1.10xx  acc=0.3x
  after   loss=0.2xxx  acc=0.9x
  delta   loss=-0.8xxx  acc=+0.5x
```

## LoRA math (toy)

For frozen `W ∈ R^{out×in}` and adapters `A ∈ R^{r×in}`, `B ∈ R^{out×r}`:

```text
classic:  scale = α / r
rslora:   scale = α / √r     # Rank-Stabilized LoRA (teaching stub, not peft)
ΔW = scale · B @ A
y  = x @ Wᵀ + x @ ΔWᵀ + b
```

- `A` ~ N(0, 1/√in), `B = 0` → adapter starts as a **no-op**
- Only `A` and `B` receive gradients; base `W` / hidden layer stay frozen
- Config: `configs/default.yaml` → `lora.rank`, `lora.alpha`, `lora.scaling`
- Eval JSON records `scale_mode` (+ `scaling_value`). Optional: `python evals/runner.py --sweep`
- **`merge_and_unload()`** folds `W ← W + scale·(B@A)` and returns a plain `MergedLinear` / classifier (**must assign** the return value — same footgun as [peft#2032](https://github.com/huggingface/peft/issues/2032)). Eval JSON adds `mode: adapter|merged` and `adapter_vs_merged_max_abs_logit`.
- **Not** Hugging Face `peft` / transformers — numpy toy only.

## Eval harness

`run_before_after(cfg)` builds the model, scores the held-out synthetic set, trains LoRA, scores again, then (by default) **merges** the adapter and re-scores:

1. **loss** — mean cross-entropy (before vs after)
2. **accuracy** — argmax vs labels (before vs after)
3. **merged parity** — `model = model.merge_and_unload()`; adapter vs merged logits within atol; report `mode: merged`

```bash
pytest tests/test_eval.py tests/test_train.py -q
```

CI fails if after accuracy regresses vs before on this toy task.

## Design notes

- **Numpy only** — no torch / transformers / peft required; installs and runs on CPU in seconds
- **Synthetic blobs** — sklearn-sized multi-class Gaussians; zero network I/O
- **Explicit adapters** — learning the low-rank update, not wrapping a 7B checkpoint
- **Not production** — no distributed training, no checkpoint formats, no HF Hub push


## CI

The workflow YAML lives at [`ci/github-actions.yml`](ci/github-actions.yml) (mirrored for OSS demos). To enable GitHub Actions on this repo, copy it to `.github/workflows/ci.yml` (requires a token with the `workflow` scope). Pytest + eval CLI are the gate.

## License

MIT — see [LICENSE](LICENSE).
