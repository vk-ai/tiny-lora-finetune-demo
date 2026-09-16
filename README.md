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
| `src/tiny_lora/lora.py` | Toy `LoRALinear`: `y = Wx + (α/r) BAx + b` |
| `src/tiny_lora/model.py` | Tiny frozen MLP + LoRA classification head |
| `src/tiny_lora/train.py` | SGD over `A`/`B` only |
| `src/tiny_lora/eval.py` | Before/after loss + accuracy harness |
| `configs/default.yaml` | LoRA `rank` / `alpha`, data & train knobs |
| `evals/runner.py` | CI-friendly eval CLI + JSON report |
| `tests/` | Unit + train + harness (pytest, seconds on CPU) |
| `ci/` + `.github/workflows/` | Mirrored GitHub Actions CI |

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
  trainable=144  frozen=608
  before  loss=1.10xx  acc=0.3x
  after   loss=0.2xxx  acc=0.9x
  delta   loss=-0.8xxx  acc=+0.5x
```

## LoRA math (toy)

For frozen `W ∈ R^{out×in}` and adapters `A ∈ R^{r×in}`, `B ∈ R^{out×r}`:

```text
ΔW = (α / r) · B @ A
y  = x @ Wᵀ + x @ ΔWᵀ + b
```

- `A` ~ N(0, 1/√in), `B = 0` → adapter starts as a **no-op**
- Only `A` and `B` receive gradients; base `W` / hidden layer stay frozen
- Config: `configs/default.yaml` → `lora.rank`, `lora.alpha`

## Eval harness

`run_before_after(cfg)` builds the model, scores the held-out synthetic set, trains LoRA, scores again:

1. **loss** — mean cross-entropy (before vs after)
2. **accuracy** — argmax vs labels (before vs after)

```bash
pytest tests/test_eval.py tests/test_train.py -q
```

CI fails if after accuracy regresses vs before on this toy task.

## Design notes

- **Numpy only** — no torch / transformers / peft required; installs and runs on CPU in seconds
- **Synthetic blobs** — sklearn-sized multi-class Gaussians; zero network I/O
- **Explicit adapters** — learning the low-rank update, not wrapping a 7B checkpoint
- **Not production** — no distributed training, no checkpoint formats, no HF Hub push

## License

MIT — see [LICENSE](LICENSE).
