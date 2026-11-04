# Experiment results

This directory contains saved outputs from the project runs.

## Main executed experiment

The headline results are under `experiment/test/`. This is the completed held-out gold-span classification run using data derived from the PhysioNet SNOMED CT Entity Linking Challenge v1.2.1:

- 272 notes
- 75,490 input annotation rows
- 75,442 annotations retained after validation
- note-level support/dev/test split (60/20/20)
- 1, 5, 10 and 20 shots per entity type
- 50 episodes per shot count
- identical paired episodes for point and covariance-aware heads

The top-level repository README contains the headline metric tables and explains the evaluation scope. Detailed outputs:

- `experiment/test/metrics/` — aggregate metrics, per-episode metrics and paired comparisons
- `experiment/test/plots/` — F1/ECE curves, confusion matrices, calibration and confidence plots
- `experiment/test/config.yaml` — saved configuration
- `experiment/test/splits.json` — saved note split membership
- `experiment/test/data_audit.json` — input counts and rejected records
- `experiment/test/COMPLETE.json` — completion metadata

## Other folders

- `experiment/dev_0/`, `dev_1/`, `dev_2/`: development/configuration runs, not the final held-out test result.
- `smoke/`: smoke-test artefacts, not headline results.
- `legacy_previous_dataset/`: earlier dataset outputs retained for traceability. Do not combine these with the current experiment when reporting results.

FP/FN CSV exports are intentionally excluded from this public-facing bundle because per-example error records may contain clinical text. The original clinical dataset files are also excluded; see `../data/README.md`.

**The completed results are already present.** You do not need to run Colab merely to inspect them.
