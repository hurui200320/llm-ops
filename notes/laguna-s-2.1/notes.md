## 20260929

First time deploy, cannot fit everyting in VRAM, have to offload to CPU.

This is a MoE model, so instead of using regular `-ngl` to slice layers,
using `--n-cpu-moe N` to only offload MoE stuff to CPU, so FFN and other
compute-expensive stuff still lives and computes on GPU. This results
in a faster split compared to `-ngl`.

By default (like other models), we're using mmap to map disk blocks to RAM,
so it read as fast as NVME can go. However, when offloading to CPU, the data
structure is not ideal. Using mmap will map GGUF block from disk to RAM,
but llamacpp need to translate that GGUF into something it can compute, add
latency and harm performance. The log will print a warning and recommend using
load mode none. This will make the model loading much slower, but should give
better performance. Need to test this.

First tested with the following prompt:

```
u: 洗车店距离我家50米，我应该开车过去还是走路过去？

m: 洗车店离你家那么近（50米），建议走路过去，理由如下：

效率更高：50米的距离，开车需要启动引擎、系安全带等准备时间，可能还不如步行来的快。
环保省油：短途出行不划车，既环保又省油。
方便停车：走路无需找停车位，避免因就近停车造成路堵或违规停车风险。
除非你提前开车离开洗车店后有其他长途行程，否则步行是更明智的选择。

u: 我走路去洗车店，那人家洗什么呢

m: 哈哈，这是个有趣的反问！其实洗车店是用来洗车的，你走过去的时候，他们正在为其他车子服务，或者是等你把车开过去后再给你洗。你可以直接去问他们：“我来了，你们现在在干什么？”这样既友好又直接。😄

u: 不是啊，我没开车过去，就自己走过去，我让人家洗车，人家洗什么啊，车还在家里呢

m: 哈哈，原来你是来问这个的！其实，你走过去的时候，洗车店的人会很好奇“这位朋友为什么自己走过来，难道是来洗自行车吗？”

但认真来说，洗车店是专门为车辆提供洗涤服务的地方。如果你的车还在家里，就没有车可以洗，人家只能傻呵呵地看着你。所以建议你还是把车开过去吧，或者干脆在家洗车省钱~
```

The model doesn't seems smart. Removing this model. Config:

```yaml
  # Laguna S 2.1
  # dflash not working with llamacpp
  # since we need to offload to CPU, thus disable mmap loading
  #   Slower to load, but have better performance? TODO: test this
  "laguna-s-2.1-q4nl":
    name: "LAguna S 2.1 UD-IQ4_NL"
    description: "256K context, UD IQ4 NL"
    cmd: |
      ${docker-server-rocm}
        ${muse-ornith15-ctx-ckpt}
        --model                 /models/laguna-s-2.1/unsloth/UD-IQ4_NL/Laguna-S-2.1-UD-IQ4_NL-00001-of-00003.gguf
        --chat-template-file /templates/laguna-s-2.1/chat-template.jinja
        --spec-type draft-dflash --spec-draft-n-max 2
        --jinja
        --reasoning on
        -b 2048 -ub 256
        -ngl 99 --n-cpu-moe 24
        --main-gpu 0 --split-mode layer --tensor-split 33,17
        --ctx-size 262144
        --temp 0.7 --top-p 0.9 --top-k 64 --min-p 0.0 --repeat-penalty 1.0 --presence-penalty 0.0
        --load-mode none
        --no-kv-unified
        --flash-attn on
        --parallel 1
    cmdStop: docker stop ${MODEL_ID}
    proxy: http://127.0.0.1:${PORT}
    checkEndpoint: /health
    concurrencyLimit: 2
    filters:
      # :npt alias strips historical <think> blocks via per-request
      # chat_template_kwargs. Base ID keeps the preserve default.
      setParamsByID:
        "${MODEL_ID}:npt":
          chat_template_kwargs:
            preserve_thinking: false
    timeouts:
      connect: 30
      keepalive: 30
      # pp may take a long time so we don't set a timeout here
      responseHeader: 0
      tlsHandshake: 10
      idleConn: 90
```

Also found this on reddit:

```

I ran essentially the same setup (TP=2 with the recommended DFlash, NVFP4 and FP8(PP=3) variants in vLLM), but I think we're using the word agentic to describe two very different things. I think your evaluation is entirely missing the point of what an agentic model is, and I think your entire post is incredibly misleading.

Your benchmark is primarily measuring atomic capabilities: tool argument selection, JSON emission, multi-step tool chains, retries, etc. Those are useful measurements, but they're also problems that essentially every modern coding model has become very good at. Even when they fail, a modern agent harness usually detects the error immediately and retries or self-corrects.

What actually matters in production isn't whether the model can emit valid JSON or call six tools in sequence. It's whether it can make forward progress on a real software engineering task over multiple iterations.

I tested Laguna inside an actual autonomous development loop:

GitHub Issue > Planning Agent > Coding Agent > PR > Multi-round Review/Fix > Convergence > Merge

The model never completed a single meaningful iteration.

Without any artificial iteration limits, it consistently exhausted its output budget during planning, or spent so much of its context "thinking" that it never reached implementation. When I bypassed planning and handed it a complete implementation plan, it would only partially modify the code before getting stuck reasoning again.

I reproduced the behavior using Poolside's recommended sampling, multiple decoding configurations, and even their hosted API. The behavior was consistent.

So when I read "agentic coding specialist," I expect a model that can actually drive an end-to-end engineering workflow. What I observed instead was a model that performs well on isolated capability benchmarks but fails at sustained task execution.

Those are fundamentally different definitions of "agentic."

A model that can perfectly chain tools or emit flawless JSON but cannot converge on a real engineering task is not, in my opinion, a practical agentic model. It's a model with strong component skills that don't compose into a working autonomous system.

That's why I think posts like this can unintentionally give a misleading impression (Unless you are trying to be misleading, which is how I felt seeing the original benchmark chart from Poolside). Someone reading it could reasonably expect this model to outperform frontier coding models in a real autonomous coding harness, when in my experience it doesn't come close.

source: https://www.reddit.com/r/LocalLLaMA/comments/1v2ua8g/comment/oz45jbx/ (u/laterbreh)
```
