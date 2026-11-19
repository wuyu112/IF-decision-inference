# IF Decision inference package

This repository is the exact inference package used for IF-decision-27b v1.
It implements System One /v1/systemone typed decisions, calibrated option
probabilities, and the vLLM prompt-bidirectional attention plugin required by
the model.

The implementation is derived from StartLuxLabs/StartLux-Decision 1.1.0 under
Apache-2.0. The public upstream checked during release was commit
a99977d9332decc9fb27a2488b4a83b533b6cd5f. The runtime copy differs in four
files: cli.py, launch.py, server.py, and serving.py. Those differences add a
request deadline, aiohttp handler cancellation, lazy safetensors loading, and a
configurable vLLM max-num-seqs. The full resulting source is included here.

## Model and tested configuration

- Model: https://huggingface.co/peng12345/IF-decision-27b
- Tested weights: 49cb2aea2b54cbaabf2344c5939d689d1d5178ce
- Base: Qwen/Qwen3.8-27B revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0
- Precision: bfloat16
- vLLM: 0.31.0
- Context: 32768 tokens
- GPU memory utilization: 0.90
- max-num-seqs: 8
- Prefix caching and chunked prefill: disabled
- Native option capacity: 26

Full-attention layers occur every fourth layer (zero-based 3, 7, …, 63).
The plugin makes the prompt bidirectional in those full-attention layers. It
does not make every model layer bidirectional. Decision probabilities are read
from the A–Z option logits and mapped reversibly to the original option keys.

Requests above 26 options must be rejected as Unsupported for the formal
Decision Index run. The release workflow provides a capacity proxy for that
policy; grouped multi-pass inference is not claimed as equivalent.

## Clean installation

Use Python 3.12 and an NVIDIA CUDA environment compatible with vLLM 0.31.0.

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e '.[vllm]'

Download the immutable model revision:

    hf download peng12345/IF-decision-27b \
      --revision 49cb2aea2b54cbaabf2344c5939d689d1d5178ce \
      --local-dir model

Start the exact configuration:

    bash scripts/serve_if_decision.sh \
      --model ./model --port 18080 --vllm-port 18081 --gpu 0

Check health and run the protocol test:

    curl -fsS http://127.0.0.1:18080/health
    python tests/reproduce_request.py --base-url http://127.0.0.1:18080

## API

POST /v1/systemone receives model, state, and questions. Choice answers contain
the selected original key and a finite normalized probability for every
original option. Noul answers contain the probability of Yes.

See docs/api.md and docs/vllm_bidirectional.md.

## Reproduction evidence

The release validation compares a fixed request against the original service
and a fresh install of this repository. It verifies response schema, original
key preservation, probability normalization, chosen argmax, and per-option
probabilities within the tolerance recorded in reproduction_report.json.

## Licence and attribution

Apache-2.0. See LICENSE and NOTICE. StartLux-Decision is Copyright StartLux and
its contributors. IF Decision model training and weights are separate from this
inference implementation.
