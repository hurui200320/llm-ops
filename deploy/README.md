# Deploy

Running on fedora-tuf, which has two AMD 7900XTX.

```
/mnt/unenc-xfs/llama-swap -config /mnt/unenc-xfs/llm-ops/deploy/llama-swap.config.yaml -watch-config
```

+ `/mnt/unenc-xfs/models`: copy the archive/LLM content from NAS
+ `/mnt/unenc-xfs/llm-ops`: git clone of this repo

To download llama-swap (download to /mnt/unenc-xfs/llama-swap):

```
curl -sL https://api.github.com/repos/mostlygeek/llama-swap/releases/latest | jq -r '.assets[] | select(.name | test("linux_amd64")).browser_download_url' | xargs curl -sL | tar xz -C /mnt/unenc-xfs/ llama-swap
chmod +x /mnt/unenc-xfs/llama-swap
```

To update docker images:

```
docker pull ghcr.io/ggml-org/llama.cpp:full-vulkan
docker pull ghcr.io/ggml-org/llama.cpp:full-rocm
```

Note that llamacpp's docker image is not the latest, it will build docker image on fixed interval.
If latest docker doesn't work, build our own image.

## Security

This config and deployment setup is meant to be used in LAN, more specifically, my home LAN.

So it's wide open with no API keys configured, as I am the only living human in my house.
And I know I will be the only user.