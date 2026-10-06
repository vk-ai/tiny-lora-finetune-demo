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
| `src/tiny_lora/dora.py` | DoRA stub: magnitude vector `m` + normalized direction (peft `use_dora=True` analogue) |
| `src/tiny_lora/qlora.py` | Optional QLoRA *concept* (bitsandbytes import-guarded; numpy fake-4bit always offline-safe) |
| `src/tiny_lora/multi_adapter.py` | `MultiAdapterHead`: named adapters + `add_weighted_adapter` (linear / cat / TIES-lite) (round 4) |
| `src/tiny_lora/merge_eval.py` | Two-task A/B adapter demo + merge eval table + single-adapter invariant |
| `src/tiny_lora/forgetting.py` | Forgetting/retention check: task A → task B per variant + intruder dimensions (round 5) |
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


## DoRA + optional QLoRA (round 3)

**DoRA** (weight-decomposed LoRA) keeps a learnable **magnitude** vector `m` and applies LoRA on the **normalized direction** of `W+ΔW`:

```text
W′ = m · (W + scale·B@A) / ||W + scale·B@A||_row
```

Config: `lora.use_dora: true` (or peft production one-liner `LoraConfig(use_dora=True)` — **not** used here; numpy toy only).

**QLoRA concept:** `lora.qlora: true` fake-quantizes the frozen base to 4-bit-shaped storage for the memory story. Real `bitsandbytes` is **optional** — if absent, CI still runs the numpy fake path and `qlora_available()` is False.

```python
from tiny_lora import compare_adapters, format_comparison  # format via eval
from tiny_lora.eval import compare_adapters, format_comparison
rows = compare_adapters()  # LoRA vs DoRA vs fake-QLoRA table
print(format_comparison(rows))
```

Community decision axes: start LoRA → QLoRA if memory-bound → DoRA if quality-bound at low rank  
([data-dynamics explainer](https://www.data-dynamics.io/en/blog/lora-qlora-dora), [peft DoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora_variant_dora)).

## Eval harness

`run_before_after(cfg)` builds the model, scores the held-out synthetic set, trains LoRA, scores again, then (by default) **merges** the adapter and re-scores:

1. **loss** — mean cross-entropy (before vs after)
2. **accuracy** — argmax vs labels (before vs after)
3. **merged parity** — `model = model.merge_and_unload()`; adapter vs merged logits within atol; report `mode: merged`

```bash
pytest tests/test_eval.py tests/test_train.py -q
```

CI fails if after accuracy regresses vs before on this toy task.

## Multiple adapters + weighted merging (round 4)

`MultiAdapterHead` keeps **one frozen base** `W, b` and a dict of **named** LoRA/DoRA adapters that share it (the peft `add_adapter` / `set_adapter` / `add_weighted_adapter` shape, in numpy):

```python
import numpy as np
from tiny_lora import MultiAdapterHead

rng = np.random.default_rng(0)
head = MultiAdapterHead.from_base(W, b)                      # or .from_head(model.head)
head.add_adapter("A", rank=4, alpha=8.0, rng=rng, scale_mode="rslora")
head.add_adapter("B", rank=4, alpha=8.0, rng=rng, scale_mode="rslora")
# ... train each (see merge_eval.build_two_task) ...
head.add_weighted_adapter(["A", "B"], [1.0, 1.0], "AB", combination_type="linear")
head.add_weighted_adapter(["A", "B"], [1.0, 1.0], "AB_cat", combination_type="cat")
head.add_weighted_adapter(["A", "B"], [1.0, 1.0], "AB_ties", combination_type="ties", density=0.5)
head.add_weighted_adapter(["AB", "B"], [1.0, -1.0], "forget_B")  # negative weight = task negation
head.set_adapter("AB")                                         # None → base only
```

| `combination_type` | What it computes | Rank of result | Exact? |
|---|---|---|---|
| `linear` | task arithmetic `ΔW = Σ wᵢ·ΔWᵢ` (**negative weights OK**), then SVD re-factorization | `min(Σ rᵢ, out, in)` or `svd_rank` | yes (unless `svd_rank` truncates; `svd_rel_error` reported) |
| `cat` | stack factors `A=[A₁;A₂]`, `B=[w₁s₁B₁ \| w₂s₂B₂]` | `Σ rᵢ` | yes, no SVD |
| `ties` | TIES-lite on `τᵢ = wᵢ·ΔWᵢ`: top-`density` trim, then sign election (Σ τ), then disjoint mean | as `linear` | no, by design |

**Invariant (tested for classic LoRA, rsLoRA and DoRA, with every combination type):** `add_weighted_adapter([a], [1.0])` reproduces adapter `a`'s logits (max |Δlogit| about 1e-15). Two details make that hold. Both correspond to the bug report in [peft#3761](https://github.com/huggingface/peft/issues/3761), where one adapter at weight 1.0 came back *different* for rsLoRA and DoRA, silently:

- **Scaling.** Each ΔWᵢ uses *its own* scale (α/r or α/√r). The merged adapter gets **effective scaling 1**, and its factors carry the scale. Re-deriving α/√r from the new rank is the bug, and `test_naive_rescaling_breaks_rslora_invariant` shows it.
- **DoRA magnitude.** It has no canonical combination. This toy *defines* it as magnitude task arithmetic, `m = m₀ + Σ wᵢ·(mᵢ − m₀)` with `m₀ = ‖W‖_row`, which is exact for one adapter at weight 1.0. The upstream plan is to raise for DoRA, and `dora_policy="raise"` does the same here. Mixing LoRA and DoRA in one merge raises.

**Two-task demo:** adapter `A` learns labels {0,1} and `B` learns {2,3} (4-class blobs, chance 0.25):

```bash
python evals/runner.py --multi-adapter    # classic / rsLoRA × LoRA / DoRA tables → evals/multi_adapter.json
```

```text
adapter          rank  task_A  task_B     all      (classic LoRA, seed 42)
base                0   0.172   0.109   0.141
A                   4   1.000   0.000   0.500
B                   4   0.000   1.000   0.500
linear              4   1.000   1.000   1.000
cat                 8   1.000   1.000   1.000
ties                4   1.000   1.000   1.000
linear_minus_B      4   1.000   0.000   0.500
```

The linear, cat and TIES merges solve **both** tasks. Subtracting `B` again (negative weight, cf. [peft#2796](https://github.com/huggingface/peft/issues/2796)) forgets task B. TIES is lossy by design and scores slightly lower on DoRA and rsLoRA runs. Background: [peft model-merging guide](https://huggingface.co/docs/peft/v0.18.0/developer_guides/model_merging) · [TIES, arXiv 2306.01708](https://arxiv.org/abs/2306.01708) · [LoRAX adapter merging](https://loraexchange.ai/guides/merging_adapters/).

**Honesty:** this is a numpy toy of peft `add_weighted_adapter` and TIES. It is not peft, mergekit, or LoRAX, has no DARE, and the numbers come from synthetic blobs, not LLM benchmarks.

## Forgetting / retention check (round 5)

This answers "if I fine-tune again on task B, how much of task A do I lose, per adapter variant?" It reuses the round-4 two-task data: task A is labels {0,1} and task B is labels {2,3}, scored with a 4-way argmax (chance 0.25).

1. **Stage 1:** fine-tune the full head on task A. The result becomes the frozen base `W_A, b_A`.
2. **Stage 2:** train each variant on task B **only**, from the same base, with the same lr and batch size:
   - `full_head`: `W` and `b`
   - `full_W`: `W` only, with the bias frozen as LoRA does
   - `lora_r{2,4,8}`
   - `rslora_r8`
   - `dora_r4`
3. **Report:** task-A accuracy before and after, forgetting, retention, task-B accuracy after, ‖ΔW‖_F, and an **intruder-dimension** count. That count is the number of singular vectors of `W'` with max |cos| < 0.5 against `W_A` ([arXiv 2410.21228](https://arxiv.org/abs/2410.21228)).

```bash
python evals/runner.py --forgetting          # 5 seeds, fixed budget + matched learning → evals/forgetting.json
python evals/runner.py --forgetting --variants full_head lora_r2 lora_r8 --target-b-acc 0.95
```

Results are means over seeds 42–46 (`A_min` is the worst seed):

```text
fixed budget (50 epochs)          matched learning (stop at task-B train acc ≥ 0.90)
variant    A_after  A_min B_after  | epochs A_after  A_min B_after
full_head    0.010  0.000   0.997  |   10.2   0.863  0.458   0.918
full_W       0.871  0.609   0.985  |   31.0   0.975  0.922   0.877
lora_r2      0.212  0.000   1.000  |    8.4   0.648  0.458   0.981
lora_r4      0.256  0.047   1.000  |    7.4   0.778  0.609   0.972
lora_r8      0.341  0.172   1.000  |   15.2   0.791  0.623   0.963
rslora_r8    0.222  0.094   0.997  |    4.6   0.743  0.375   0.956
dora_r4      0.572  0.422   0.709  |   45.4   0.663  0.426   0.767
```

What this toy shows. Treat it as small-scale evidence, not a general law:

- **Over-training on B is the main source of forgetting.** Every variant keeps much more of task A when it stops once B is learned. Monitoring the stage-1 metric while you train stage 2 is what the peft maintainers suggest in [peft#2873](https://github.com/huggingface/peft/issues/2873).
- **"LoRA forgets less" is not automatic here.** With the bias frozen, the full fine-tune (`full_W`) retained the most. At this lr and α/r scaling, LoRA takes much larger weight steps (‖ΔW‖ about 10–24 vs 6–8). Under matched learning, higher rank forgot less (r2 < r4 ≈ r8). Compare the "illusion of equivalence" discussion in [peft#2907](https://github.com/huggingface/peft/issues/2907) and [arXiv 2410.21228](https://arxiv.org/abs/2410.21228). [peft#3542](https://github.com/huggingface/peft/issues/3542) (CLoRA) and [The Neural Base's forgetting walkthrough](https://theneuralbase.com/lora-fundamentals/learn/advanced/catastrophic-forgetting-investigation/) cover mitigations.
- **The bias matters:** moving `b` is the cheapest way to push classes 0/1 down, and `full_head` uses it.

**Honesty:** the head has only 4 outputs, so rank ≥ 4 is already a full-rank ΔW. The differences come from optimisation dynamics and what is frozen, not from extra capacity. These are synthetic blobs, 5 seeds, and one lr, not LLM benchmarks, and the numbers move with lr and α. CI checks only that stage 1 learns A and every variant learns B above chance. It does not gate on which variant "wins".

## Design notes

- **Numpy only** — no torch / transformers / peft required; installs and runs on CPU in seconds
- **Synthetic blobs** — sklearn-sized multi-class Gaussians; zero network I/O
- **Explicit adapters** — learning the low-rank update, not wrapping a 7B checkpoint
- **Not production** — no distributed training, no checkpoint formats, no HF Hub push


## CI

The workflow YAML lives at [`ci/github-actions.yml`](ci/github-actions.yml) (mirrored for OSS demos). To enable GitHub Actions on this repo, copy it to `.github/workflows/ci.yml` (requires a token with the `workflow` scope). Pytest + eval CLI are the gate.

## License

MIT — see [LICENSE](LICENSE).
