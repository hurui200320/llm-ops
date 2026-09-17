# llm-ops

My private repo for managing local LLM deployment

## Model storage

Models are stored locally on my NAS's `archive` share.

With a gnome managed SMB mount, it's located at `/var/run/user/<uid>/gvfs/smb-share:server=<tailscale ip>,share=archive/LLM`.

Under the `LLM` folder, models are stored by `LLM/<Model Family>/<Variant>/`.

Models are grouped by model family, for example: `gemma-4`, `muse-glimmer` and `ornith-1.5`. Model family should written in a-z0-9 and `-` only.

Under each family, each variant is a dedicated folder: `<owner>-<size>-<attribution>`.
For example, under the `gemma-4` family, we might have: 

+ `google-26B-A4B-it`: Google's official Gemma 4 26B A4B it
+ `google-31B-it-qat-q40-unsloth`: Google's official Gemma 4 31B it QAT version made by unsloth
+ `coder3101-31B-it-heretic`: coder3101's decensored version of Gemma 4 31B it using heretic

Under each variant's folder, we will have a `README.md` and the GGUFs and the sha256 checksum files.
See [scripts](./scripts/README.md) for more info.

## Toolbox

The maintenance scripts ship as a public container image built by GitHub Actions:

```text
ghcr.io/hurui200320/llm-ops-toolbox:latest
```

Run an interactive maintenance shell on the NAS (models assumed under `/mnt/user/archive/LLM`):

```bash
docker run -it --rm --pull=always -v /mnt/user/archive/LLM:/data \
    ghcr.io/hurui200320/llm-ops-toolbox:latest
```

Inside the container, `check_updates`, `verify_checksums`, and `download_hf` are available on `PATH`, the library root defaults to `/data` (the mounted library), and downloads persist in the mounted library after the container is removed. A one-shot update check:

```bash
docker run --rm --pull=always -v /mnt/user/archive/LLM:/data:ro \
    ghcr.io/hurui200320/llm-ops-toolbox:latest check_updates
```

Set `HF_TOKEN` at runtime (`-e HF_TOKEN=...`) for gated models; it is never part of the image. See [scripts/README.md](./scripts/README.md) for full usage details.
