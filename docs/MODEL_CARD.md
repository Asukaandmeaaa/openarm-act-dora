# ACT mixed real/simulation checkpoint

- Artifact: `model-v1/act-sim-mix-30k.tar.gz` in GitHub Releases.
- Runtime directory: `models/act_sim_mix_100_30k_01/pretrained_model`.
- Exact archive and six runtime file hashes: `artifacts/model.json`.
- Architecture: LeRobot ACT, two RGB cameras, 16-dimensional bimanual state/action, chunk size 100.
- Training provenance: `mix30k_metrics_01/checkpoint_030000`, 30,000 additional mixed-data updates initialized from `act_sim_mix_100_060000_init` (the existing real-expert model expanded to 100-step chunks).
- Data composition recorded in the original training manifest: 37 real expert episodes, 100 normal simulation episodes, 50 simulation recovery/correction episodes.
- Original training-manifest SHA256: `5f182bbba7fdba6e1c0d56c640084f38e46388c64d3df012c652620b98137780`.

The published checkpoint and normalization processors are byte-for-byte copies of those used by the original tested deployment. Processor statistics are part of the artifact; do not substitute new normalization values. CPU loading overrides only the saved device-processor placement. Skipping pretrained ResNet initialization does not remove backbone weights from the full checkpoint.

The deployment uses the right arm for policy motion; the left arm is held with gravity compensation. The model still expects both arms in right-then-left order and both camera images. Camera placement, scene arrangement, motor conventions, state normalization and gripper encoding must match the collection setup.

This release provides the trained checkpoint for deployment replication. It does not include the raw training datasets or the complete training pipeline. See `docs/EXPERIMENTS.md` for a single operator-reported physical session, not a generalized task-success benchmark.

The repository owner's trained artifact is distributed with the project's MIT terms; third-party software and backbone implementations retain their upstream licenses and attribution in NOTICE. No new ACT algorithm is claimed.
