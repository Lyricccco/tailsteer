# Release validation

Validation was performed on eight NVIDIA H100 80 GB GPUs with Python 3.10 and the backbone-specific environments. These checks compare the public implementation with the research implementation; they are not new full-benchmark scores.

## Numerical and training checks

- The four default training configurations match their 2,000-update paper runs, including the optimizer schedule, sigma-Beta sampler, trainable scope, and Qwen's final-step loss weight. Paths are portable, and automatic inference export is enabled.
- One 1024 × 1024 local-edit cache trajectory per backbone was generated with both implementations. All 20 compared state/schedule tensor arrays were exactly equal.
- For each backbone, a non-square source was edited for two rounds using both `base` and `tailsteer`, with full native denoising schedules and image feedback. All 16 public/reference PNG pairs were pixel-identical.
- All four backbones generated the first English and Chinese Qwen-Image-Bench cases under both policies (16 images). FLUX.2, Qwen and SenseNova also completed one multi-image composition turn under both policies (six images), retaining all four static references. These runs produced 11 comparison panels. A checkpoint-free Kontext `base` run completed two editing rounds and two comparison panels.
- Each backbone completed one real eight-GPU FSDP2 optimizer update using its freshly generated cache and the paper learning-rate horizon. All 32 rank reports confirmed finite gradients, finite updated parameters, and changed parameter values. The validation harness substituted small rank receipts for large checkpoint writes; this does not constitute an eight-GPU checkpoint-I/O or full 2,000-update run.
- Real PSNR, SSIM, and AlexNet LPIPS computations matched the research implementation exactly. Four objective-score records were also generated, aggregated, and reused on resume.

## CPU and packaging checks

The main CPU test environment passed 256 tests, with 20 environment-dependent tests skipped. The separate Kontext environment check passed six tests; the Qwen environment passed its 22-test inference suite. Real LPIPS was checked separately using installed weights. Tests cover target construction, gradients, schedule conventions, non-square geometry, multi-reference routing, policy selection, dataset identities, resume, and scoring aggregation.

The editable package and CLI entry point were installed in an isolated environment. All five help commands and four eight-GPU training dry runs succeeded. Static checks found no unused imports or undefined names.

Qwen-Image-Bench's 2,000 EN/ZH prompt records and BizGenEval's 400 records matched the research manifests. Twenty editing-benchmark launch plans (five benchmarks across four backbones) were checked with real input assets.

The public Q-Judger worker also completed one actual GPU judgement of a generated English sample, using the official Quality checklist and explicitly enabled thinking mode. The parsed checklist passed completeness validation without a retry. This is a scorer integration check, not a complete Qwen-Image-Bench score.

## Scope

The full model stacks were validated in existing pinned environments, rather than by reinstalling every GPU dependency from scratch. API judges were tested with protocol fixtures; no paid API requests or complete benchmark reruns were performed. External model access terms and API credentials must be supplied by the user. Dataset publication checks are documented with the dataset releases.

For a new installation, start with a small cache using `--limit 8 --disable-filter`, then a transient one-update training run using `--stop-at-update 1`. Use separate smoke output directories. Remove `--disable-filter` and use the full source manifest for a paper reproduction.

## Complete model repositories

The FLUX.2, Qwen-Image-2.1 and SenseNova TailSteer releases retain their official model layouts. The public suite loader was checked against the original inference exports on two image-feedback editing rounds using both `base` and `tailsteer`. All 12 generated image pairs were pixel-identical. A further six images from actual TailSteer-only CLI runs, loading every model component from the complete released repositories, were also pixel-identical to the inference exports. Existing native-pipeline validation also covers text-to-image generation and two-round editing.

The Hub download route, sharded tensor loading, actor switching and TailSteer-only model selection have CPU regression coverage. Hub files were checked against the local release inventories; model-card updates leave every weight shard unchanged.

## Published dataset checks

Both dataset releases were uploaded, checked against the Hub file hashes, and downloaded into fresh directories for checksum verification. The 47,788 source requests preserve the research manifest's IDs, grouping, prompts, seeds and category metadata. The 100 benchmark images, masks and all 1,000 editing rounds preserve the frozen inputs.

All four backbone cache/inference plans were checked against the downloaded releases with image feedback and COA enabled. The Hugging Face JSON loader read both manifests successfully, and 25 publication/public-workflow tests passed. Default preparation commands pin the verified Hub commit IDs; Version tags provide human-readable release references.
