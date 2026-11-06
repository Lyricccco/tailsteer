# Readme figures and examples

These figures are assembled from saved TailSteer experiment outputs used by the project page and paper. Images retain their full frames and aspect ratios; no enhancement or retouching is applied. Animation captions summarize instructions. Original prompts, seeds where recorded, and displayed frames are included in each model's `examples/` directory.

- FLUX.2: step 2000, steering onset 46, sigma-Beta rho 0.8 / beta 3.
- SenseNova: step 2000, steering onset 42, sigma-Beta rho 0.8 / beta 3.
- Qwen-Image-2.1: step 2000, steering onset 30, sigma-Beta rho 0.8 / beta 3, final-step weight 0.1.

`results.json` stores the paper's reported scores and full-100-image Add & Preserve curves. Standard image feedback is used. Text-to-image pairs share their prompt and seed; these selected examples do not replace aggregate benchmark evaluation.

The trajectory mark and training diagram come from the TailSteer project page and paper. Inter labels and the Caveat Brush slogan use the project page's SIL Open Font License fonts; their licenses are included in the parent assets directory. The two-line slogan is stored as SVG outlines so its lettering does not depend on the reader's installed fonts. Experimental input images retain the original dataset terms. Images shared with the interactive project page are stored once in `../results/`; the example JSON files reference them by relative path. The readme metrics use all 100 images, while the project page’s API-comparison curves use the shared 97 complete chains.
