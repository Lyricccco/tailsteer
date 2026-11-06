# Evaluation

Generation, scoring and reporting are separate stages. Each benchmark retains its official sample IDs, scoring rubric and aggregation. API/parse failures are recorded as missing scores, never silently filled with zeros. A report is complete only when the expected cases have valid outputs and scores. Resume reuses successful records and retries missing ones.

## Official benchmark data and code

```bash
for bench in qwenimagebench bizgeneval gedit rise imgedit-basic mse edival; do
  tailsteer prepare "$bench"
done
```

Source repositories and revisions are listed in `configs/evaluation/sources.lock.yaml`. Assets are downloaded into `data/evaluation/`; no private benchmark fork is required. For an existing dataset, use `scripts/prepare_evaluation.py --bench NAME --source-dir PATH`. Preparation converts the official data to a shared manifest without changing instructions or case IDs.

Qwen-Image-Bench contains 1,000 English and 1,000 Chinese prompt cases, using the public `prompt_en` and `prompt_cn` fields. BizGenEval contains 400 cases and keeps its prescribed aspect ratios. ImgEdit uses the official **nine-category macro average**, not a pooled sample average.

## API judges

Configure credentials in the environment, not in a committed YAML:

```bash
export OPENAI_API_KEY='your-key'
# Optional, for a compatible provider:
export OPENAI_BASE_URL='https://your-provider.example/v1'
```

| Benchmark | Default scorer | Execution |
|---|---|---|
| GEdit | GPT-4.1 | API, CPU |
| RISE | GPT-4.1-2025-04-14 | API, CPU |
| ImgEdit-basic | GPT-4o | API, CPU |
| MSE-Bench | GPT-5-nano, Responses API | API, CPU |
| BizGenEval | Gemini-3-Flash-Preview | OpenAI-compatible API, CPU |
| Qwen-Image-Bench | Official Q-Judger | Local GPU |
| EdiVal | Official local detector stack | Local GPU |
| Add & Preserve COA | GPT-6.1-sol | API, CPU |

The provider must expose the configured judge model and image input API. A model identifier alone does not provision access. For BizGen, a separate compatible Gemini endpoint can be configured with `BIZGEN_API_KEY` and `BIZGEN_BASE_URL`. Other API judges use `OPENAI_*`. No credentials are bundled.

The COA judge settings are in `configs/evaluation/add_preserve_objects.yaml`; changing its model or prompt defines a different judging protocol. Preserve the recorded settings when comparing models.

## Local GPU judges

Q-Judger needs its own environment and checkpoint:

```bash
python3.10 -m venv .venv-qjudger
.venv-qjudger/bin/pip install -r requirements/qwen-judge.txt
.venv-qjudger/bin/pip install -e '.[evaluation]'
hf download Qwen/Qwen-Image-Bench --local-dir models/Q-Judger
export TAILSTEER_QIB_PYTHON="$PWD/.venv-qjudger/bin/python"
export TAILSTEER_QIB_MODEL="$PWD/models/Q-Judger"
```

The official judge uses thinking mode and checklist-level scoring. Responses missing dimensions are retried with a JSON-completeness instruction, without changing the rubric or inventing missing answers.

EdiVal has a separate dependency stack (vLLM and detection/vision libraries). Follow the downloaded pinned EdiVal source's `env_setup/` instructions in an isolated environment. Prepare its model assets with:

```bash
python scripts/prepare_evaluation.py --bench edival --edival-judge-models
# Alternatively register already downloaded assets:
python scripts/prepare_evaluation.py --bench edival --edival-local-models /path/to/judges
```

Its DINOv3 model may require accepting upstream access terms. Use `TAILSTEER_EDIVAL_PYTHON` to select the environment when running the public CLI. EdiVal scoring needs at least two GPUs. Local judge failures do not stop later suites.

## Run and resume

```bash
tailsteer infer --run-dir runs/klein-main --inference-suite paper-benchmarks \
  --eval-root data/evaluation --gpus 8 --resume

# An isolated generation smoke test:
tailsteer infer --run-dir runs/klein-main --inference-suite gedit \
  --limit 8 --stage generate --gpus 8
```

Generation-only and score-only invocations must use the same suite, checkpoint, policies, seeds, area cap and dataset selection so their result identities match. External benchmark results are in `data/evaluation/results/`. Summaries retain coverage, per-case scores, failures, judge settings and input fingerprints.

## Add & Preserve

```bash
tailsteer prepare add-preserve
tailsteer infer --run-dir runs/klein-main --inference-suite add-preserve --gpus 8
```

A CPU worker computes PSNR, SSIM and LPIPS as generated rounds become available. LPIPS downloads its AlexNet weights on first use; set `TORCH_HOME` to a shared cache if desired. Each output is compared with the original decorated source image using the frozen outside-box mask. No generated image is resized to make it fit the mask.

- PSNR: RGB mean squared error over outside pixels, capped at 120 dB.
- SSIM: Gaussian 11-pixel window, sigma 1.5, population covariance; average the outside similarity map.
- LPIPS: AlexNet v0.1, spatial score map at the original resolution; average outside the box.

Before SSIM and LPIPS, the allowed interior is replaced with the reference interior to suppress its contribution to neighborhoods and features. Each turn reports the equal-image mean and population standard deviation. Keep the per-turn coverage alongside the scores.

**Cumulative Object Accuracy (COA)** is enabled by default. The judge receives only the fixed allowed interior crop, excluding the red border, plus all object requests through that turn. It labels each object `yes`, `no` or `unsure`, with brief evidence. `yes` receives one point; the other two receive zero. Each turn's score is the fraction of cumulative requested objects present, and COA averages over turns, then equally over images. Missing API judgements remain missing, rather than counting as failed edits.

The default short prompt checks recognizable object presence and specified color/material, ignores size, shadows and background quality, and requires each object body to lie fully within the crop. COA measures edit completion; the outside-box metrics measure preservation. Neither substitutes for the other.

To rescore saved Add & Preserve images, pass their resolved inference config to `scripts/score_add_preserve.py`. `--objects-only` runs only COA; `--preservation-only` runs only the objective metrics.
