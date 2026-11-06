# Cache and training

## Cache generation

```bash
tailsteer cache --backbone flux2-klein-base-9b \
  --sources data/training/sources.jsonl --image-root data/training \
  --output-dir cache/flux2-klein-base-9b --gpus 8
```

Each rank generates independent source trajectories. The writer stores grouped states and schedule metadata, and resumes completed candidates. Use the same output directory to continue an interrupted capture. For a smoke test, `--limit 8` selects eight candidates; `--disable-filter` is for checking execution only and must not be used for the paper cache.

Cache generation uses the training preprocessing (1024 × 1024). Multi-turn inference preserves non-square image geometry using backbone-specific alignment. These serve different purposes and must not be conflated.

The frozen base model supplies the cached trajectory and final state. Local-edit examples use a change mask; identity examples use the full keep region. The preservation endpoint combines the source state in the keep region with the base terminal state in the edited region. FLUX/Qwen states are in their native VAE latent space; SenseNova states are packed RGB pixel tokens.

## Training

```bash
tailsteer train --backbone flux2-klein-base-9b \
  --cache-dir cache/flux2-klein-base-9b \
  --output-dir runs/klein-main --gpus 8 \
  --inference-suite cafe-20turn --inference-suite add-preserve
```

Supply another complete YAML with `--config configs/train/flux2-klein-base-9b.yaml`, or use the short overrides `--steering-onset`, `--learning-rate` and `--max-updates`. `--dry-run` writes the resolved plan and prints the launch command without allocating GPUs. FSDP2 distributes training parameters across the requested GPUs.

Only source identities in the training partition are optimized. Each iteration samples a denoising step and rolls out the current model without gradient tracking from a cached anchor. Both the base velocity and preservation-directed velocity are evaluated at the reached state; gradients flow through the trained velocity prediction at the sampled step. The optimizer updates the full selected denoiser, not just its last blocks.

| Backbone | Trainable scope | Frozen components |
|---|---|---|
| FLUX.1 Kontext | Full denoiser | VAE and text encoders |
| FLUX.2 Klein | Full transformer | VAE and text encoder |
| Qwen-Image-2.1 | Full denoiser | VAE and Qwen-VL conditioner |
| SenseNova-U1.5 | Full configured MoT, including trainable conditioning path | Frozen teacher and reference vision encoder |

SenseNova uses `conditioning_gradients: true` in its full-model setting. Detaching that path or selecting only generation experts changes the experiment.

## Paper defaults

| Backbone | Steering onset | Denoising steps | Learning rate | Final-step loss weight |
|---|---:|---:|---:|---:|
| FLUX.1 Kontext dev | 25 | 50 | 1e-5 | 1.0 |
| FLUX.2 Klein Base 9B | 46 | 50 | 1e-5 | 1.0 |
| Qwen-Image-2.1 | 30 | 40 | 1e-6 | 0.1 |
| SenseNova-U1.5-8B-MoT | 42 | 50 | 1e-5 | 1.0 |

All four configurations use 2,000 optimizer updates, 100 warmup updates and a 10,000-update cosine decay horizon. The effective batch size on eight GPUs is 16 (one sample per GPU, two accumulation steps). Sigma-Beta sampling uses `rho: 0.8` and `beta: 3`; the blending schedule uses `odds_ratio: 1`. The YAML files in [configs/train](../configs/train) are the complete configurations.

## Paper terminology

| Public configuration | Paper meaning | Internal numerical representation |
|---|---|---|
| `steering_onset` | k_s: start of preservation-oriented steering | `transition.start_step` |
| `last_denoising_step` | Final supervised model forward | `transition.supervision_step` |
| `blending_schedule: odds` | Base/preservation velocity blending | `alpha_schedule.name` |
| `odds_ratio` | Odds adjustment in the blending schedule | `alpha_schedule.odds_ratio` |
| `timestep_sampling.rho` | Uniform density mixture coefficient | `uniform_mix` |
| `timestep_sampling.beta` | Beta density shape | `beta` |
| `final_step_loss_weight` | Final-forward loss multiplier (Qwen: 0.1) | `endpoint_loss_weight` |
| `max_updates` | Optimizer-update budget | `max_steps` |
| `warmup_updates` | Learning-rate warmup | `warmup_steps` |
| `lr_decay_updates` | Cosine horizon | `lr_schedule_steps` |

The implementation's internal `alpha` is the **preservation** coefficient. The paper's lambda weights the **base** velocity, so lambda = 1 − alpha. This naming translation does not reverse the blending calculation.

Sigma-Beta sampling integrates `rho + (1-rho) * beta * (1-sigma)^(beta-1)` over each native denoising interval. It is not uniform sampling of discrete step indices. Index labels remain zero based for FLUX/Qwen and one based for SenseNova, preserving the paper's selected onsets.

## Checkpoints and continuation

```bash
tailsteer train --config configs/train/flux2-klein-base-9b.yaml \
  --output-dir runs/klein-main --resume auto --gpus 8
```

`checkpoints/step-XXXXXXXX/` contains the distributed model, optimizer and scheduler state for resuming training. `exports/step-XXXXXXXX/` contains the inference weights and manifest; it excludes optimizer state. Inference exports are written at configured checkpoint boundaries, including the last update.

To stop a smoke run after one update without changing the intended learning-rate horizon, use `--stop-at-update 1`. This skips post-training suites. Resume with the same configuration afterwards.

Post-training suites run in the supplied order. A failed suite is recorded in `suite-status.json`, and later suites are attempted. Completed results can be reused with `tailsteer infer --run-dir runs/klein-main --resume ...`. Inference failures do not invalidate the completed training checkpoint.
