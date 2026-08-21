# Qwen runtime diagnostics

This directory stores opt-in Qwen audit runs made by
`tests/run_qwen_diagnostics.py`. Runtime folders are ignored by Git because
they contain video crops and verbose model output, but they remain available
on the server for inspection.

Each run contains:

- `manifest.json`: inputs, model path, crop settings, and selected SAM3 IDs;
- `<track_id>/crops/`: every image supplied to Qwen;
- `<track_id>/contact_sheet.jpg`: one visual overview of those inputs;
- `<track_id>/crop_manifest.json`: frame, box, brightness, and sharpness data;
- `<track_id>/legacy_*`: the current `recognize.py` prompt and response;
- `<track_id>/decomposed_*`: validity/jersey-only prompt and response;
- `<track_id>/result.json`: parsed output and deterministic roster lookup;
- `summary.json`: first failed stage for every tested trajectory.

Run from the repository root. Repeat `--track-id` to test selected tracks, or
omit it to diagnose every SAM3 player candidate:

```bash
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

python -u tests/run_qwen_diagnostics.py \
  --video /home/fangzilin/data/basket/bkn-vs-det-0022400861/video/100.mp4 \
  --bbox-json /home/fangzilin/data/basket_artifacts/bkn-vs-det-0022400861/tracks/raw/100.json \
  --roster-json /home/fangzilin/data/basket_artifacts/bkn-vs-det-0022400861/metadata/recognize_roster.json \
  --qwen-model /home/fangzilin/models/Qwen2.5-VL-7B-Instruct \
  --run-name bkn-det-100 \
  --mode both
```

`legacy_retained=false` with `decomposed_failure_stage=complete` indicates
that the current combined prompt rejected a track that the separated prompt
could identify. Other failure-stage values localize the first failure to JSON
parsing, on-court validation, jersey color, jersey number, or roster lookup.
