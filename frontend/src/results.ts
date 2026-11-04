import aggregate from "../../results/experiment/test/metrics/all_results.json";
import e1 from "../../results/experiment/test/metrics/episodes_1shot.json";
import e5 from "../../results/experiment/test/metrics/episodes_5shot.json";
import e10 from "../../results/experiment/test/metrics/episodes_10shot.json";
import e20 from "../../results/experiment/test/metrics/episodes_20shot.json";

export const shots = [1, 5, 10, 20] as const;
export type Shot = (typeof shots)[number];
export type Method = "point" | "gaussian";
export const classes = ["Problem", "Treatment", "Test"] as const;
type ClassName = (typeof classes)[number];
type Score = { mean: number; std: number };
type Episode = { f1: number; support: number; per_type: Record<ClassName, { f1: number }> };
type Episodes = { point: Episode[]; gaussian: Episode[] };
type Result = {
  point_prototype: {
    metrics: { f1: Score; macro_f1: Score; n_episodes: number };
    calibration: { ece: number };
  };
  gaussian_prototype: {
    metrics: { f1: Score; macro_f1: Score; n_episodes: number };
    calibration: { ece: number };
  };
};
const saved = aggregate as Record<string, Result>;
const episodes: Record<Shot, Episodes> = {
  1: e1,
  5: e5,
  10: e10,
  20: e20,
} as EpisodesRecord;
type EpisodesRecord = Record<Shot, Episodes>;
for (const shot of shots) {
  const run = saved[`${shot}_shot`],
    pair = episodes[shot];
  if (!run || pair.point.length !== 50 || pair.gaussian.length !== 50)
    throw new Error(`Incomplete ${shot}-shot experiment data`);
  for (const method of ["point", "gaussian"] as const) {
    const list = pair[method],
      summary =
        run[method === "point" ? "point_prototype" : "gaussian_prototype"]
          .metrics;
    if (
      summary.n_episodes !== list.length ||
      list.some(
        (e) =>
          !Number.isFinite(e.f1) ||
          classes.some((c) => !Number.isFinite(e.per_type[c]?.f1)),
      )
    )
      throw new Error(`Invalid ${shot}-shot ${method} episodes`);
    const mean = list.reduce((sum, e) => sum + e.f1, 0) / list.length;
    if (Math.abs(mean - summary.f1.mean) > 1e-9)
      throw new Error(
        `Aggregate and episode scores disagree for ${shot}-shot ${method}`,
      );
  }
}
export const pct = (n: number, d = 2) => `${(n * 100).toFixed(d)}%`;
export const pp = (n: number) =>
  `${n >= 0 ? "+" : ""}${(n * 100).toFixed(2)} pp`;
export const methodName = (m: Method) =>
  m === "point" ? "Point prototype" : "Covariance-aware";
export function result(shot: Shot, method: Method) {
  return saved[`${shot}_shot`][
    method === "point" ? "point_prototype" : "gaussian_prototype"
  ];
}
export function episodeList(shot: Shot, method: Method) {
  return episodes[shot][method];
}
export function macroEpisode(e: Episode) {
  return classes.reduce((sum, c) => sum + e.per_type[c].f1, 0) / classes.length;
}
export function paired(shot: Shot, metric: "micro" | "macro") {
  const a = episodes[shot].point,
    b = episodes[shot].gaussian;
  if (a.length !== b.length) throw new Error("Unpaired episode results");
  let wins = 0,
    losses = 0,
    ties = 0;
  a.forEach((p, i) => {
    const x = metric === "micro" ? p.f1 : macroEpisode(p),
      y = metric === "micro" ? b[i].f1 : macroEpisode(b[i]);
    const d = y - x;
    if (d > 1e-12) wins++;
    else if (d < -1e-12) losses++;
    else ties++;
  });
  return { wins, losses, ties, total: a.length, rate: wins / a.length };
}
export function perEntity(shot: Shot, method: Method) {
  return classes.map((c) => ({
    name: c,
    value:
      episodeList(shot, method).reduce((s, e) => s + e.per_type[c].f1, 0) /
      episodeList(shot, method).length,
  }));
}
export const series = shots.map((shot) => ({
  shot,
  point: result(shot, "point").metrics.f1.mean,
  gaussian: result(shot, "gaussian").metrics.f1.mean,
  pointMacro: result(shot, "point").metrics.macro_f1.mean,
  gaussianMacro: result(shot, "gaussian").metrics.macro_f1.mean,
  pointEce: result(shot, "point").calibration.ece,
  gaussianEce: result(shot, "gaussian").calibration.ece,
}));
const plots = {
  ...import.meta.glob("../../results/experiment/test/plots/confusion_*.png", {
    eager: true,
    query: "?url",
    import: "default",
  }),
  ...import.meta.glob("../../results/experiment/test/plots/calibration_*.png", {
    eager: true,
    query: "?url",
    import: "default",
  }),
} as Record<string, string>;
export const plotUrl = (
  kind: "confusion" | "calibration",
  method: Method,
  shot: Shot,
) =>
  plots[
    `../../results/experiment/test/plots/${kind}_${method === "point" ? "point" : "gauss"}_${shot}shot.png`
  ];
