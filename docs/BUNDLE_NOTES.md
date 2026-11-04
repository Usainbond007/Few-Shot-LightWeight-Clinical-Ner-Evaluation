# Bundle notes

This repository combines the source project and saved experiment outputs. The main current run is `results/experiment/test/`; development and smoke outputs are retained. `results/legacy_previous_dataset/` contains earlier results and must not be presented as the current held-out run.

The final run is based on the PhysioNet SNOMED CT Entity Linking Challenge v1.2.1 (derived from MIMIC-IV-Note). Raw clinical CSVs and potentially text-bearing FP/FN exports are intentionally excluded from this GitHub bundle because redistribution/access restrictions and privacy must be respected. See `data/README.md`.

The methodology document records the broader proposed study. The executed run included here is the gold-span classification experiment described in the main README; native GLiNER extraction and separate external validation were not run for its headline metrics.
