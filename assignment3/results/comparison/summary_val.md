# Model comparison — val split

Mean ± sample standard deviation over seeds (± 0 when only one run exists).

| Model | Runs | Accuracy | Macro-F1 | Trainable params | s/epoch | Peak VRAM (MiB) | Image gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| image_only | 1 | 0.4101 ± 0.0000 | 0.1779 ± 0.0000 | 2,565 | 50.1 | 1298 | — |
| text_only | 1 | 0.3598 ± 0.0000 | 0.1995 ± 0.0000 | 847,493 | 47.2 | 64 | — |
| concat | 1 | 0.3675 ± 0.0000 | 0.2298 ± 0.0000 | 1,045,893 | 48.4 | 1302 | — |

Differences of this size should be judged against the seed standard deviation.
