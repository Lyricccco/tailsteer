# Dataset licensing and attribution

TailSteer-authored instructions and benchmark annotations are provided under the Apache License 2.0; the full license is in `LICENSE-APACHE-2.0.txt`.

This does not relicense the underlying images. Refer to `provenance.jsonl` for each image's source, reported upstream license and available attribution.

- **Open Images:** the release retains the upstream image identifier, author, title, landing URL and license URL from the official image metadata. See the [Open Images download page](https://storage.googleapis.com/openimages/web/download_v7.html) and the corresponding [CC BY 2.0 terms](https://creativecommons.org/licenses/by/2.0/) where specified by the record.
- **MONET:** the release retains the source repository, revision, shard, source group and reported license. MONET's dataset card is [available here](https://huggingface.co/datasets/jasperai/monet). Its repository license and constituent-source metadata are distinct from an independent verification of copyright in every underlying image.

Training source images are copied from the research source collection without re-encoding. Add & Preserve additionally applies RGB/orientation preprocessing, alignment cropping and a white rectangle with an inward red border. Its manifest records the geometry and hashes of the distributed assets.
