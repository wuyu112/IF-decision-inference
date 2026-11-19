# IF Decision inference package

This repository is the exact inference package used for IF-decision-27b v1.
It implements System One /v1/systemone typed decisions, calibrated option
probabilities, and the vLLM prompt-bidirectional attention plugin required by
the model.

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
- Native option capacity per model pass: 26

Full-attention layers occur every fourth layer (zero-based 3, 7, …, 63).
The plugin makes the prompt bidirectional in those full-attention layers. It
does not make every model layer bidirectional. Decision probabilities are read
from the A–Z option logits and mapped reversibly to the original option keys.

Choice requests above 26 options use a versioned wide-choice path: near-equal
groups of at most 25 options are all scored,
the best 3 per group advance, and at most 26 finalists are scored in a final
pass. Finalists share 0.999 probability mass according to the final pass. Every
non-finalist remains in the response and shares 0.001 in proportion to its
first-round probability. The response therefore contains one finite,
normalised probability for every original option key.

## Decision Index integration

Decision Index accepts custom inference through its built-in HTTP engine. Start
this server, then run the unmodified public kit against `/v1/systemone`:

    python -m decision_index pipeline \
      --engine http \
      --option base_url=http://127.0.0.1:18080 \
      --option model=IF-decision-27b-v1 \
      --suite /path/to/suite-0.3 \
      --out /path/to/run \
      --edition 0.3 --compact

The server receives only the kit's `state` and `questions`. It does not inspect
benchmark identifiers, source metadata, or expected answers. Original option
keys are mapped reversibly to A-Z for read-out and restored in the response.
The kit validates that every question is answered, every original option has a
finite probability, probabilities sum to one, and `choice` is an original key.

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

The protocol test covers ordinary choice/noul output and a 29-option request;
the latter must return all 29 original keys with a normalised distribution.

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

## References and attribution

The decision-head design and multi-pass inference strategy were informed by
the ideas described in [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision).
The referenced implementation is licensed under Apache-2.0; see `LICENSE` and
`NOTICE` for the required attribution and pinned reference revision.
