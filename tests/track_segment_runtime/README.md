# Track-segment validation runtime

This directory stores controlled experiments produced by
`tests/run_track_segment_validation.py`. Generated trajectory JSON, PlayNet
reports, manifests, and logs are ignored by Git.

The experiment keeps the source-video frame coordinate system intact. Boxes
outside a manually audited stable interval are replaced with `null`; the
selected ball is copied from the existing clean JSON. The boundary is recorded
as a manual assumption and must not be treated as an automatic split result.

For clip 100, the first validation isolates the white number 20 portion of raw
`player_8` and checks the canonical checkpoint label `ast` (Assist). See the
repository handoff message or `--help` for the complete server command.
