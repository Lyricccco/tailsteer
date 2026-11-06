# Upstream components

TailSteer loads backbone implementations and official benchmark scorers from their original repositories. Source revisions are recorded in `upstream.lock` and `configs/evaluation/sources.lock.yaml`.

| Component | Source |
|---|---|
| FLUX.1 Kontext | https://github.com/black-forest-labs/flux |
| FLUX.2 / Qwen pipelines | https://github.com/huggingface/diffusers |
| Qwen-Image-2.1 | https://github.com/QwenLM/Qwen-Image-2.1 |
| SenseNova-U1 | https://github.com/OpenSenseNova/SenseNova-U1 |
| Qwen-Image-Bench | https://github.com/QwenLM/Qwen-Image-Bench |
| BizGenEval | https://github.com/microsoft/BizGenEval |
| GEdit | https://github.com/stepfun-ai/Step1X-Edit |
| RISE | https://github.com/PhoenixZ810/RISEBench |
| ImgEdit | https://github.com/PKU-YuanGroup/ImgEdit |
| MSE-Bench | https://github.com/ByteDance-Seed/VINCIE |
| EdiVal | https://github.com/TianyuCodings/EdiVal |

Downloaded implementations retain their upstream LICENSE files. Model weights and datasets are downloaded separately and retain their own terms. The demo folder contains example inputs and model-generated reference outputs; a code license does not relicense third-party input images or model weights.

The figures in `website/assets/readme/` reproduce TailSteer paper results and selected project-site demonstrations. They include original inputs and generated outputs, with prompts and quantitative data alongside the figures. These presentation assets retain the same underlying image and model terms as the demos. Plot and comparison labels use [Inter](website/assets/INTER-LICENSE.txt); the two-line slogan uses [Caveat Brush](website/assets/CAVEAT-BRUSH-OFL.txt). Both fonts are licensed under the SIL Open Font License.

The interactive page uses the same shared images and fonts. Mathematical glyphs
use DejaVu Serif, with its [license notice](website/assets/DEJAVU-LICENSE.txt).
Resource and model links use the organizations' identifying marks:
[arXiv](https://info.arxiv.org/brand/logos.html),
[GitHub](https://brand.github.com/foundations/logo),
[Hugging Face](https://huggingface.co/datasets/huggingface/brand-assets),
[Black Forest Labs](https://huggingface.co/black-forest-labs),
[SenseNova](https://huggingface.co/sensenova) and
[Qwen](https://huggingface.co/Qwen). These marks retain their respective owners'
rights and are not covered by this repository's code license.
