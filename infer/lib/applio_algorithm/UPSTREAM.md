# Applio inference architecture snapshot

This directory vendors `rvc/lib/algorithm` from the official Applio repository
at commit `085197e738ce9dd4c0bae1e0a74df5de25b89444` (2026-08-12 checkout).

Only import paths were changed from `rvc.lib.algorithm` to
`infer.lib.applio_algorithm` so the code can coexist with the vendored official
RVC runtime. The snapshot is used only when a checkpoint declares Applio's
`MRF HiFi-GAN` or `RefineGAN` vocoder. Standard HiFi-GAN RVC checkpoints keep
the existing official RVC inference implementation.

The upstream MIT license is preserved in `LICENSE`.
