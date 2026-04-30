# TODO

- Review training and inference scripts outside `models/`.
- Fix training resume-state handling so missing checkpoints do not skip epochs.
- Fix learning-rate schedule stepping so it updates every iteration.
- Improve script ergonomics with clearer validation and runtime summaries.
- Fix inference utility imports so scripts run from the repo root.
- Document likely bottlenecks:
  - Per-sample CPU metric conversion inside the training loop.
  - Serial image and EXR reads during inference.
