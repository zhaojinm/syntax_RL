# Third-party components

This tree contains source snapshots for reproducibility:

| Component | Upstream | Revision | License file |
| --- | --- | --- | --- |
| verl | https://github.com/verl-project/verl | `3a394c9bd09c8bd321ae2070edaa3b5e84af201a` | `vendor/verl/LICENSE` |
| LLaMA-Factory | https://github.com/hiyouga/LlamaFactory | `92fa3df4c4e4e338ecc1cbdc6530b118337286e3` | `vendor/LlamaFactory/LICENSE` |
| self-attentive-parser EVALB copy | https://github.com/nikitakit/self-attentive-parser | `24435a156a64d7433829e9a500a81f46b7e58030` | `third_party/EVALB/LICENSE` |

`third_party/EVALB/nk.prm` matches the experiment copy (`MAX_ERROR 0`) rather
than the upstream revision's `MAX_ERROR 10000`. This does not change scores on
valid, token-aligned PTB predictions, but records the exact evaluation setup.

Syntax-RL's own top-level license has not yet been selected. The bundled
third-party components retain their respective licenses listed above.
