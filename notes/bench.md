# Benchmark

Date: 2026 Sep 18
llama.cpp (docker): `version: 0.4.1-dev (build 11028, commit 972d2313b)`

## Synbad

```bash
time ./bench/run_synbad.sh
```

### All pass models

The following models passed all tests for both plain and streaming.

+ `google-gemma-4-31b-q5km-text`
+ `ornith-1.5-35b-a3b-q80-vision`
+ `ornith-1.5-35b-a3b-abliterated-q80-vision`

### `google-gemma-4-26b-a4b-q80-vision` and `google-gemma-4-31b-qat-q40-vision`

Both plain and streaming failed at `tools/octo-list-no-optional-args` and the failure flaky.
For 26B model, plain test failed at 1/20 while streaming failed at 7/20.
For 31B QAT model, both plain and stream test failed at 1/20.

The failed response is identical (excluding tool call id) on the two models.

```
Response:
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "list",
        "arguments": "{\"dirPath\":\".\"}"
      },
      "id": "skVRg7B4vl9CtyB6TejTuA5MpGQRa2w8"
    }
  ]
}
AssertionError [ERR_ASSERTION]: Tried multiple asserts, but they all failed.
AssertionError [ERR_ASSERTION]: Expected a null or undefined value

AssertionError [ERR_ASSERTION]: Tried multiple asserts, but they all failed.
AssertionError [ERR_ASSERTION]: Expected values to be loosely deep-equal:

{
  dirPath: '.'
}

should loosely deep-equal

{}

AssertionError [ERR_ASSERTION]: Tried multiple asserts, but they all failed.
AssertionError [ERR_ASSERTION]: Expected a null or undefined value

AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:

'.' !== ''

    at new AssertionError (node:internal/assert/assertion_error:380:5)
    at Module.or (node/v24.21.0/lib/node_modules/@syntheticlab/synbad/dist/source/asserts.js:14:19)
    at Module.test (node/v24.21.0/lib/node_modules/@syntheticlab/synbad/dist/evals/tools/octo-list-no-optional-args.js:8:12)
    at Command.<anonymous> (node/v24.21.0/lib/node_modules/@syntheticlab/synbad/dist/source/index.js:141:26)
    at process.processTicksAndRejections (node:internal/process/task_queues:104:5) {
  generatedMessage: false,
  code: 'ERR_ASSERTION',
  actual: undefined,
  expected: undefined,
  operator: undefined,
  diff: 'simple'
}
❌ tools/octo-list-no-optional-args failed
```

The test expects the tool call to have no parameter, either it should be null, or empty object, or empty string.
But the model adds a default and harmless `.` parameter.

This doesn't mean llama.cpp is broken. It might suggest the quant is damaging the model, but it's hard to say with confidence.
For both 26B Q8 model and 31B QAT model, other tests are perfectly fine, only failed due to this harmless default value.

I think it's more like the model's character, it just tends to add a harmless default value.
Considering the synbad test is focused on testing llamacpp's parsing, I'd dismiss this error without rerunning the 40 round test.
We should use `tool-eval-bench` for the real tool call test.

### `meta-muse-glimmer-30b-kquant-vision`

Both plain and streaming test failed at `tools/parallel-tool`:

```
Response:
{
  "role": "assistant",
  "content": "",
  "reasoning_content": "We need to get weather for Paris and London. Use get_weather tool. The tool takes location city name. Need to call twice? Probably one at a time. We can call for Paris first, then London. The instruction says we have access to tools. We need to call get_weather for Paris, then for London. Since we can only do one function call per turn, we need to do sequential.\n\nFirst call get_weather location \"Paris\". Then later London.\n\nWe should respond with weather. Use tool.\n\nProbably need to format. Let's call get_weather for Paris.",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\":\"Paris\"}"
      },
      "id": "hWl1XBPKj76axRefsru0FQMqqQ9iT2LF"
    }
  ]
}
AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:

1 !== 2

    at Module.test (node/v24.21.0/lib/node_modules/@syntheticlab/synbad/dist/evals/tools/parallel-tool.js:5:12)
    at Command.<anonymous> (node/v24.21.0/lib/node_modules/@syntheticlab/synbad/dist/source/index.js:141:26)
    at process.processTicksAndRejections (node:internal/process/task_queues:104:5) {
  generatedMessage: true,
  code: 'ERR_ASSERTION',
  actual: 1,
  expected: 2,
  operator: 'strictEqual',
  diff: 'simple'
}
❌ tools/parallel-tool failed
```

It failed to produce parallel tool calls. After digging some documents:

+ https://dev.meta.ai/docs/muse-glimmer/prompting#tool-calling

> Muse Glimmer supports one tool call per turn. It does not support parallel tool calls;
> return each tool result before asking the model to select the next tool.

So it's the model's limitation, not llamacpp's.

### Timing

+ `google-gemma-4-31b-q5km-text`:              plain 3531 s, streaming 3533 s
+ `google-gemma-4-26b-a4b-q80-vision`:         plain 1625 s, streaming 1597 s
+ `google-gemma-4-31b-qat-q40-vision`:         plain 2968 s, streaming 2954 s
+ `meta-muse-glimmer-30b-kquant-vision`:       plain 2060 s, streaming 1978 s
+ `ornith-1.5-35b-a3b-q80-vision`:             plain 1854 s, streaming 1997 s
+ `ornith-1.5-35b-a3b-abliterated-q80-vision`: plain 1773 s, streaming 1825 s

Script duration: 462m49s

## Speed

```
time ./bench/run_longctx.sh
```

+ `google-gemma-4-31b-q5km-text`:              pp  355.33 t/s  tg 14.77 t/s
+ `google-gemma-4-26b-a4b-q80-vision`:         pp 1338.54 t/s  tg 36.80 t/s
+ `google-gemma-4-31b-qat-q40-vision`:         pp  300.87 t/s  tg 15.65 t/s
+ `meta-muse-glimmer-30b-kquant-vision`:       pp  954.41 t/s  tg 27.74 t/s  accept_rate 0.4957
+ `ornith-1.5-35b-a3b-q80-vision`:             pp 1996.68 t/s  tg 31.08 t/s
+ `ornith-1.5-35b-a3b-abliterated-q80-vision`: pp 1992.81 t/s  tg 30.86 t/s

Total time: 122m14s
