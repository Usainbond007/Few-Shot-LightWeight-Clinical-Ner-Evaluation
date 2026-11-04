# Clinical entities research demo

From Windows PowerShell, run:

```powershell
cd C:\dev\Ml_Project_Clinical_NER\frontend
npm.cmd install
npm.cmd run dev
```

Open the local URL printed by Vite. To check TypeScript and produce a static bundle in `dist/`, run `npm.cmd run build`.

The frontend uses React, Vite, TypeScript, Tailwind CSS v4, shadcn/ui controls (Button, Native Select, Collapsible, Separator), Recharts, and Lucide. The palette is defined with CSS variables in `src/tailwind.css`; the research layout is in `src/style.css`.

The UI imports the archived held-out result JSON and plot images from `../results/experiment/test/`. It does not access clinical CSVs or run GLiNER inference. The illustrative note is synthetic. For the eight evaluation views, Micro-F1, Macro-F1, mean and standard deviation, win rate, ECE, and per-entity F1 are populated from saved data; the confusion matrix displays archived row-normalized plots. Raw confusion counts and example-level predictions with confidence need saved query-level labels, predictions, and probabilities. No such records are in the archived test metrics.

The saved `all_results.json` supplies aggregate means, SD, and ECE. Paired wins and ties are calculated from `episodes_*shot.json`; class F1 is averaged over each episode's `per_type` scores. The app validates episode counts and aggregate means when it loads.
