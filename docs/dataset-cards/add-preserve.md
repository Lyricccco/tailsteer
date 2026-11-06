---
pretty_name: Add & Preserve
license: other
license_name: mixed-source-licenses
license_link: LICENSE.md
language:
- en
task_categories:
- image-to-image
size_categories:
- n<1K
tags:
- tailsteer
- multi-turn-image-editing
- benchmark
configs:
- config_name: default
  data_files:
  - split: test
    path: manifest.json
  field: items
---

# Add & Preserve

**Can your editor add new objects for ten turns without changing the rest of the image?**

100 images · 10 English editing instructions per image · object completion + background preservation.

![Ten-turn editing comparison](media/ten-turn-editing.gif)

*One input, ten successive small-object additions. Each model feeds its own previous output into the next turn.*

## A simple test for cumulative drift

Add objects **inside the red box**. Keep everything **outside it** unchanged. Retain the objects added in earlier turns. This separates successful editing from preservation, so simply ignoring an instruction cannot count as doing well on both.

Images come from **[Open Images](https://storage.googleapis.com/openimages/web/index.html)** and **[MONET](https://huggingface.co/datasets/jasperai/monet)**. We manually selected images and box positions; English addition prompts were annotated with **GPT-6.1-sol**.

![Example benchmark inputs](media/benchmark-examples.jpg)

**COA — Cumulative Object Accuracy** checks the requested objects inside the box using GPT-6.1-sol. **PSNR, SSIM and LPIPS** measure changes outside the box relative to the initial input. The evaluation reports these metrics after each turn.

## Results from the TailSteer paper

**Preservation curves across ten turns**

![Preservation across turns](media/drift-curves.png)

| Model | COA (%) ↑ | PSNR@10 ↑ | SSIM@10 ↑ | LPIPS@10 ↓ |
|---|---:|---:|---:|---:|
| FLUX.1 Kontext dev | 34.77 | 18.64 | 0.585 | 0.388 |
| FLUX.1 Kontext dev after TailSteer | 34.84 | 21.02 | 0.672 | 0.257 |
| FLUX.2 [klein] Base 9B | 31.02 | 13.85 | 0.289 | 0.524 |
| FLUX.2 [klein] Base 9B after TailSteer | 31.84 | 19.96 | 0.621 | 0.290 |
| SenseNova-U1.5-8B-MoT | 94.93 | 11.23 | 0.166 | 0.607 |
| SenseNova-U1.5-8B-MoT after TailSteer | 94.78 | 18.22 | 0.489 | 0.417 |
| Qwen-Image-2.1 | 67.36 | 16.93 | 0.438 | 0.270 |
| Qwen-Image-2.1 after TailSteer | 66.94 | 18.15 | 0.513 | 0.239 |
| GPT-Image-2.5-Sunburst (API) | 98.29 | 18.55 | 0.540 | 0.378 |

TailSteer rows use the paper's reference steering onsets: Kontext 25, Klein 46, SenseNova 42 and Qwen 30. These are image-feedback results; GPT API scores use the paper's available-output protocol.

![Qualitative results after ten editing turns](media/turn10-paper.png)

*Turn-10 comparisons from the paper. The separately labeled “w/o VAE” columns illustrate the latent-feedback ablation.*

## Evaluate your editor

With TailSteer installed, prepare your backbone and configure the COA judge API, then run:

```bash
tailsteer prepare add-preserve
tailsteer infer --backbone flux2-klein-base-9b --policies base \
  --inference-suite add-preserve --gpus 8
```

For a trained run, use `--run-dir YOUR_TRAINING_RUN` instead of `--backbone ... --policies base` to compare the base model and TailSteer.

To download the inputs independently:

```python
from huggingface_hub import snapshot_download
snapshot_download("Lyricccco/Add-and-Preserve", repo_type="dataset",
                  revision="v1.0.1", local_dir="data/add-preserve")
```

`tasks.json` contains the prompts and seeds; `images/` contains the initial inputs; `masks/` and `manifest.json` define the evaluation regions. Feed each generated image into the next turn and keep the supplied geometry unchanged.

Per-image attribution and licenses are in `provenance.jsonl` and [LICENSE.md](LICENSE.md). Our training source collection is available as **[TailSteer-Source](https://huggingface.co/datasets/Lyricccco/TailSteer-Source)**.
