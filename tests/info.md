# Tests

```bash
scripts/run_tests.sh                       # pytest, or run_all_tests.py as fallback
scripts/run_tests.sh tests/test_model.py -k forward
```

```
tests/
├── test_text_transform.py   char ↔ int round-trip, incl. Akan ɛ/ɔ and digits
├── test_model.py            DeepSpeech and ConformerCTC forward pass shapes & registry
├── run_all_tests.py         unittest suite covering DeepSpeech, Conformer, metrics,
│                            audio transforms, greedy decoding, and HF mock dataset
└── info.md                  this file
```

The suite is deliberately dependency-light: it exercises shapes, encodings and
metric maths on synthetic tensors, so it runs in seconds without any audio data
or trained checkpoint. It does **not** cover the training loop, HuggingFace
dataset loading or the Gradio app — those need real data and weights.
