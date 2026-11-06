---
pretty_name: TailSteer-Source
license: other
license_name: mixed-source-licenses
license_link: LICENSE.md
language:
- en
task_categories:
- image-to-image
size_categories:
- 10K<n<100K
tags:
- tailsteer
- image-editing
- trajectory-cache
configs:
- config_name: default
  data_files:
  - split: train
    path: sources.jsonl
---

# TailSteer-Source

**Source images and local-edit instructions for learning drift-resistant image editing.**

11,947 images · 47,788 trajectory requests · photographs, anime, illustrations, 3D renders and graphic designs.

![Source images and local edits](media/source-to-edit.gif)

*Selected source/edit pairs from real FLUX.2 [klein] Base 9B rollouts. Each instruction edits one local feature while keeping the rest of the image unchanged.*

## From images to preservation training

Images come from **[Open Images](https://storage.googleapis.com/openimages/web/index.html)** and **[MONET](https://huggingface.co/datasets/jasperai/monet)**. Local-edit prompts were annotated with **GPT-6.1-sol**. Each image has **three independent local-edit instructions and one identity instruction**.

Run a base editor on these requests to build a **Preservation-Aware Trajectory Cache**, then train TailSteer from that cache. The download provides source images and prompts; the cache is generated with your chosen backbone.

![What a trajectory cache contains](media/cache-contents.png)

The cache retains the encoded source, terminal output, selected denoising states, preservation masks, and the conditioning/schedule metadata needed for training. The colored state panels above are PCA visualizations, not decoded images.

## Get started

With TailSteer installed and the base model prepared:

```bash
tailsteer prepare training-source
tailsteer cache --backbone flux2-klein-base-9b --gpus 8
tailsteer train --backbone flux2-klein-base-9b --gpus 8
```

Also supports **FLUX.1 Kontext dev, Qwen-Image-2.1 and SenseNova-U1.5-8B-MoT**. Change `--backbone` to select the model.

To download the dataset independently:

```python
from huggingface_hub import snapshot_download
snapshot_download("Lyricccco/TailSteer-Source", repo_type="dataset",
                  revision="v1.0", local_dir="data/training")
```

`sources.jsonl` pairs relative image paths with prompts, source IDs and seeds. Images are in `images/`; per-image attribution is in `provenance.jsonl`. Original image licenses are retained; see [LICENSE.md](LICENSE.md).

Evaluate multi-turn editing with our companion benchmark: **[Add & Preserve](https://huggingface.co/datasets/Lyricccco/Add-and-Preserve)**.

From *TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing*.
