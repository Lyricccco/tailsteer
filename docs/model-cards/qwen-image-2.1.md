---
license: other
license_name: qwen-research
license_link: LICENSE
library_name: diffusers
pipeline_tag: any-to-any
base_model: Qwen/Qwen-Image-2.1
base_model_relation: finetune
tags:
- tailsteer
- image-editing
- text-to-image
---

# Qwen-Image-2.1 · TailSteer

### Join the TailSteer Adventure! ✨

We'd love to see what you create with TailSteer! Share your results, ideas, or feedback with us—every little creation makes our day! (≧▽≦) ♡

If you'd like to support our work, a ⭐ on [**GitHub**](https://github.com/Lyricccco/tailsteer) or a ❤️ on [**Hugging Face**](https://huggingface.co/Lyricccco/Qwen-Image-2.1-TailSteer) would mean a lot—and stay tuned to our GitHub repo for future TailSteer-trained model releases! **(ง •̀ᴗ•́)ง ♡**

### Keep creating. Keep the details.

**Preserve the generation and editing strengths of Qwen-Image-2.1. Unlock greater stability across repeated edits.** TailSteer is a final training stage that largely retains the base model's learned capabilities while reducing the drift that builds up when each edit becomes the next input.

[Try the model](#use-your-existing-pipeline) · [Single-turn results](#keep-the-foundation-models-strengths) · [Add & Preserve](https://huggingface.co/datasets/Lyricccco/Add-and-Preserve) · [Training data](https://huggingface.co/datasets/Lyricccco/TailSteer-Source)

**+1.22 dB PSNR · 11.5% lower LPIPS** outside the edit region at turn 10 on Add & Preserve.

## Keep editing the scene. Keep the people.

![Twenty successive edits: input, base model and TailSteer](../../website/assets/readme/qwen/twenty-turns.gif)

Twenty successive stage edits put faces, clothing and background details under repeated generation. TailSteer reduces the accumulated texture drift while continuing to carry out the edits. Each branch feeds its own previous output into the next turn. [View the final comparison](../../website/assets/readme/qwen/twenty-turns-last.jpg) · [Prompts and frames](../../website/assets/readme/qwen/examples/editing.json).

## Use your existing pipeline

**Same architecture. Same official model layout. Same loading API.** These are complete model weights: replace your original model ID or local path with `Lyricccco/Qwen-Image-2.1-TailSteer`. No adapters, merging, extra inference modules or TailSteer-specific runtime are required.

Use the same official environment as the base model. Only the model ID changes.

```python
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Lyricccco/Qwen-Image-2.1-TailSteer", torch_dtype=torch.bfloat16,
).to("cuda")

edited = pipe(
    image=Image.open("input.png").convert("RGB"),
    prompt="Change the jacket to blue.",
    num_inference_steps=40, true_cfg_scale=1.0, use_kv_cache=True,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
edited.save("output.png")
```


For the next edit, pass the edited image back into the same pipeline. Use TailSteer throughout the original denoising trajectory; no model switching or preservation mask is needed at inference.

## Keep the foundation model's strengths

![Text-to-image comparisons with identical prompts and seeds](../../website/assets/readme/qwen/text-to-image.jpg)

*Same prompt and seed in each pair. [Full prompts and images](../../website/assets/readme/qwen/examples/text-to-image.json).*

The same checkpoint supports text-to-image generation and single-turn editing. Across Qwen-Image-Bench, BizGenEval, GEdit and ImgEdit, scores stay close to the base model. The table includes RISE as well, where accuracy is lower in this setting.

| Benchmark | Base model | After TailSteer |
|---|---:|---:|
| Qwen-IB EN ↑ | 53.39 | 53.06 |
| Qwen-IB ZH ↑ | 53.34 | 53.20 |
| BizGen ES (%) ↑ | 57.03 | 57.38 |
| GEdit EN ↑ | 8.11 | 8.04 |
| GEdit ZH ↑ | 7.80 | 7.72 |
| RISE accuracy (%) ↑ | 20.56 | 18.06 |
| ImgEdit ↑ | 4.40 | 4.42 |

## The same multi-image interface

![Multiple-reference editing with the original pipeline](../../website/assets/readme/qwen/multi-input.gif)

A person and a pose guide remain ordinary inputs to Qwen. This sequence shows successive reference-guided edits with the base model and TailSteer. [View the final comparison](../../website/assets/readme/qwen/multi-input-last.jpg).

## Measure what stays unchanged

[**Add & Preserve**](https://huggingface.co/datasets/Lyricccco/Add-and-Preserve) makes drift visible: add a small object inside the red box at each turn, retain previous objects, and preserve everything outside. It tests both object completion and background preservation across 100 images and ten turns.

![Preservation over ten turns on all 100 benchmark images](../../website/assets/readme/qwen/preservation-curves.png)

| Model | COA (%) ↑ | PSNR@10 ↑ | SSIM@10 ↑ | LPIPS@10 ↓ |
|---|---:|---:|---:|---:|
| Base model | 67.36 | 16.93 | 0.438 | 0.270 |
| After TailSteer | 66.94 | 18.15 | 0.513 | 0.239 |

COA (Cumulative Object Accuracy) averages requested-object presence across ten turns. PSNR, SSIM and LPIPS measure the unchanged region. These are standard image-feedback results; curves use all 100 images. [Metric data](../../website/assets/readme/qwen/results.json).

## Run demos and benchmarks

With the TailSteer codebase installed in the official backbone environment:

```bash
tailsteer infer --backbone qwen-image-2.1 \
  --checkpoint Lyricccco/Qwen-Image-2.1-TailSteer \
  --policies tailsteer --inference-suite cafe-20turn --gpus 1
```

Prepare the original base weights and omit `--policies tailsteer` for side-by-side comparisons. The same suite interface runs text-to-image, single-turn editing, multi-turn editing and Add & Preserve benchmarks after their datasets and judges are configured.

## How TailSteer works

**Preserve early dynamics; steer the trajectory tail.** TailSteer learns from the editor's own denoising rollouts, using unchanged source content as a preservation reference. It improves long-horizon stability using single-edit trajectories, without requiring curated ground-truth editing pairs or training on multi-turn editing chains.

This is the paper's **2,000-update checkpoint**, with training steering onset **30 / 40**. The onset controls training supervision; inference uses the trained model at every step. The packaged model was [validated against the original inference export](../validation.md).

From *TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing*.

The [base model](https://huggingface.co/Qwen/Qwen-Image-2.1) license and attribution apply: [LICENSE](https://huggingface.co/Lyricccco/Qwen-Image-2.1-TailSteer/blob/main/LICENSE) · [NOTICE](https://huggingface.co/Lyricccco/Qwen-Image-2.1-TailSteer/blob/main/NOTICE). Improved using Qwen.
