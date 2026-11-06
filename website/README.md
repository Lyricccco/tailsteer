# TailSteer project page

Published at **https://lyricccco.github.io/tailsteer/** from this directory.
The page is plain HTML, CSS and JavaScript. It needs no application server,
package installation or model weights.

## GitHub Pages

1. Upload this repository to `Lyricccco/tailsteer`, including `.github/workflows/pages.yml`.
2. In **Settings → Pages → Build and deployment**, select **GitHub Actions** as the source.
3. Push to `main`, or run **Actions → Project page → Run workflow** on `main`.

The workflow validates local references, checks JavaScript syntax and publishes
only `website/`. Pull requests run the checks without deploying. No custom domain
is required. Relative resource URLs support the `/tailsteer/` project path.
See [GitHub's publishing-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Files and shared assets

- `index.html`, `styles.css`, `app.js`: page layout and trajectory animations.
- `results.js`, `results.css`, `image-loader.js`: comparisons and prioritized image loading.
- `assets/`: fonts, icons, measured curves and displayed image sequences.
- `assets/readme/`: documentation figures and example manifests. Images already
  used by the page are referenced from `assets/results/` instead of being copied.

The repository README and model cards link to these same files. When replacing
an asset, update its references in both the page and documentation. Run
`python3 scripts/check_website.py` from the repository root before publishing.
When changing CSS or JavaScript, update the corresponding `?v=` value in
`index.html` to the first 12 characters of the file's SHA-256 checksum.

The frequency animation retains all 51 measured states (steps 0–50). The
API-comparison curves use the 97 shared complete chains, while the README's
reported local-model scores use all 100 samples. These are separate evaluation
cohorts and intentionally retain separate numerical records.

Font licenses ship with the assets. Image and model terms are described in
[`THIRD_PARTY.md`](../THIRD_PARTY.md).
