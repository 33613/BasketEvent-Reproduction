# Local verification scripts

These scripts validate the BasketEvent implementation on the local Windows/
RTX 5060 Laptop environment without claiming reproduction of paper metrics.

## Scripts

- `verify_feature_pipeline.py`: verifies TimeSformer, ROIAlign, bbox/type
  embeddings, global tokens, local tokens, and actor-global attention with one
  clip and one player.
- `verify_model_chain.py`: verifies the complete tensor path from
  context-enhanced entity tokens through entity interaction, temporal pooling,
  cross-clip interaction, gated pooling, player representations, and logits.

Run from the repository root with the `sam3` Conda environment active:

```powershell
$env:PYTHONNOUSERSITE = "1"
python -B local_script/verify_feature_pipeline.py
python -B local_script/verify_model_chain.py
```

The scripts use the author's clean example trajectories. Only TimeSformer is
initialized from Kinetics-400 pretrained weights. Without the authors'
BasketEvent checkpoint, the project-specific embeddings, interaction modules,
pooling modules, and classifier are randomly initialized. The checks therefore
cover execution, shapes, masks, finite values, and resource usage—not learned
event semantics or accuracy.

## Ground-truth label limitation

Qwen associates an anonymous trajectory with a roster identity. Event labels
come separately from NBA play-by-play structured fields and deterministic
parsing of the ground-truth description, then must be joined to trajectories by
player identity. The authors did not release that acquisition, parsing, and
joining pipeline or the complete annotated JSON files. The checked-in example
JSON has no `event.actionType`, so it cannot provide a real training loss.

## Verified results (2026-08-13)

The first-stage test passed with one clip, eight frames, one player plus the
ball, and peak allocated CUDA memory of approximately 0.66 GiB.

The full-chain test also passed with two clips, eight frames per clip, eight
players plus the ball, and peak allocated CUDA memory of approximately 0.75
GiB. Verified shapes were:

```text
Z_tilde                         (2, 9, 8, 768)
Z_bar                           (2, 9, 8, 768)
c_mi                            (2, 9, 768)
C_hat                           (2, 9, 768)
h_i                             (8, 768)
alpha                           (2, 8, 1)
l_i                             (8, 11)
pi_i                            (8, 11)
```

Gated clip weights and class probabilities both summed to one for every valid
player.
