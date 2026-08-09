# Tests

```bash
scripts/run_tests.sh                       # pytest, or run_all_tests.py as fallback
scripts/run_tests.sh tests/test_model.py -k forward
```

```
tests/
├── test_text_transform.py   char ↔ int round-trip, incl. Akan ɛ/ɔ
├── test_model.py            DeepSpeech forward pass shapes
├── run_all_tests.py         unittest suite covering the above plus metrics,
│                            audio transforms and greedy decoding
└── info.md                  this file
```

The suite is deliberately dependency-light: it exercises shapes, encodings and
metric maths on synthetic tensors, so it runs in seconds without any audio data
or trained checkpoint. It does **not** cover the training loop, HuggingFace
dataset loading or the Gradio app — those need real data and weights.
