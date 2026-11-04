import { useState } from "react";
import {
  ArrowRight,
  ArrowDown,
  ArrowUpRight,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Info,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import {
  classes,
  methodName,
  paired,
  pct,
  perEntity,
  plotUrl,
  pp,
  result,
  series,
  shots,
  type Method,
  type Shot,
} from "./results";

type Page = "Overview" | "Method" | "Evaluation" | "Explore" | "Protocol";
type Metric =
  | "Micro-F1"
  | "Macro-F1"
  | "Mean ± SD"
  | "Win rate"
  | "ECE"
  | "Per-entity F1"
  | "Confusion matrix"
  | "Examples";
const pages: Page[] = [
  "Overview",
  "Method",
  "Evaluation",
  "Explore",
  "Protocol",
];
const metrics: Metric[] = [
  "Micro-F1",
  "Macro-F1",
  "Mean ± SD",
  "Win rate",
  "ECE",
  "Per-entity F1",
  "Confusion matrix",
  "Examples",
];
const methodSteps = [
  {
    name: "Start with clinical notes",
    summary: "Use discharge notes and annotations that already mark medical phrases.",
    detail: "The study uses 272 discharge notes from the PhysioNet SNOMED CT challenge. The notes were de-identified upstream by MIMIC-IV",
  },
  {
    name: "Check the labels and phrases",
    summary: "Make sure each marked phrase matches the note and has a usable type.",
    detail: "The pipeline checks character offsets against the note text and removes invalid or duplicate annotations. SNOMED concepts were mapped to Problem, Treatment, and Test in an external preparation step",
  },
  {
    name: "Separate the notes",
    summary: "Keep examples used for learning apart from notes used for testing.",
    detail: "Notes are split by note into 60% support, 20% development, and 20% held-out test. Identical note texts stay in the same split. Development notes are used to select settings before the final test.",
  },
  {
    name: "Make a few-shot task",
    summary: "Pick a few labeled examples of each type and a separate set to classify.",
    detail: "Each episode contains 1, 5, 10, or 20 support phrases per type and 50 query phrases per type. Support examples come from support notes, while final queries come from held-out test notes. Both methods receive exactly the same episode.",
  },
  {
    name: "Turn phrases into numbers",
    summary: "GLiNER reads each known phrase with nearby words and encodes it.",
    detail: "The frozen GLiNER Medium v2.1 encoder uses a mention-centered context window. It averages the mention token vectors and normalizes the resulting embedding. Its weights do not change during this study.",
  },
  {
    name: "Build two references",
    summary: "Each method summarizes the labeled examples of every type.",
    detail: "The point prototype stores one mean vector per class. The covariance-aware prototype stores a mean and a regularized diagonal estimate of how the class examples vary. Both use the same embeddings.",
  },
  {
    name: "Classify the known phrase",
    summary: "Compare the query phrase with the three types and pick one.",
    detail: "The point head uses cosine distance to class means. The covariance-aware head uses squared Mahalanobis distance with its estimated variances. Gold annotation offsets supply the phrase boundaries; neither head is tested on finding spans in free text.",
  },
  {
    name: "Measure the results",
    summary: "Check each chosen type against its label across paired test episodes.",
    detail: "The run repeats 50 paired episodes per shot count. It reports micro- and macro-F1, variation across episodes, per-type scores, calibration error, and confusion plots. The comparison uses the same queries for both heads.",
  },
];
const source = "Archived held-out test run · 50 paired episodes per shot count";
function SectionTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="section-heading">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
      </div>
      {children}
    </div>
  );
}
function Legend() {
  return (
    <div className="legend">
      <span>
        <i className="dot point" />
        Point prototype
      </span>
      <span>
        <i className="dot gaussian" />
        Covariance-aware
      </span>
    </div>
  );
}
function Chart({ kind = "micro" }: { kind?: "micro" | "macro" | "ece" }) {
  const keys =
    kind === "macro"
      ? ["pointMacro", "gaussianMacro"]
      : kind === "ece"
        ? ["pointEce", "gaussianEce"]
        : ["point", "gaussian"];
  return (
    <div
      className="chart"
      role="img"
      aria-label={`${kind} comparison across one, five, ten and twenty shots`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={series}
          margin={{ top: 12, right: 10, bottom: 0, left: -12 }}
        >
          <CartesianGrid vertical={false} stroke="#e9e9e9" />
          <XAxis
            dataKey="shot"
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${v} shot`}
            tick={{ fill: "#777a74", fontSize: 12 }}
            dy={10}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#777a74", fontSize: 12 }}
            domain={kind === "ece" ? [0, 0.42] : [0.4, 0.85]}
            tickFormatter={(v) =>
              kind === "ece"
                ? Number(v).toFixed(1)
                : `${Math.round(Number(v) * 100)}%`
            }
          />
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <div className="tooltip">
                  <strong>
                    {label} shot{label === 1 ? "" : "s"}
                  </strong>
                  <span>
                    Point ·{" "}
                    {kind === "ece"
                      ? Number(payload[0].value).toFixed(3)
                      : pct(Number(payload[0].value))}
                  </span>
                  <span>
                    Covariance ·{" "}
                    {kind === "ece"
                      ? Number(payload[1].value).toFixed(3)
                      : pct(Number(payload[1].value))}
                  </span>
                  <small>
                    {kind === "ece"
                      ? "Difference: " +
                        Math.abs(
                          Number(payload[1].value) - Number(payload[0].value),
                        ).toFixed(3)
                      : "Difference: " +
                        pp(Number(payload[1].value) - Number(payload[0].value))}
                  </small>
                </div>
              ) : null
            }
          />
          <Line
            type="monotone"
            dataKey={keys[0]}
            stroke="#a8aaa5"
            strokeWidth={2.5}
            dot={{ r: 4, fill: "#a8aaa5", strokeWidth: 0 }}
            activeDot={{ r: 6 }}
          />
          <Line
            type="monotone"
            dataKey={keys[1]}
            stroke="#171717"
            strokeWidth={3}
            dot={{ r: 4, fill: "#171717", strokeWidth: 0 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
function SelectShot({
  value,
  onChange,
}: {
  value: Shot;
  onChange: (v: Shot) => void;
}) {
  return (
    <label className="select-label">
      Support examples{" "}
      <NativeSelect
        value={value}
        onChange={(e) => onChange(Number(e.target.value) as Shot)}
      >
        {shots.map((s) => (
          <NativeSelectOption key={s} value={s}>
            {s} shot{s === 1 ? "" : "s"}
          </NativeSelectOption>
        ))}
      </NativeSelect>
    </label>
  );
}
function SelectMethod({
  value,
  onChange,
}: {
  value: Method;
  onChange: (v: Method) => void;
}) {
  return (
    <label className="select-label">
      Method{" "}
      <NativeSelect
        value={value}
        onChange={(e) => onChange(e.target.value as Method)}
      >
        <NativeSelectOption value="point">Point prototype</NativeSelectOption>
        <NativeSelectOption value="gaussian">Covariance-aware</NativeSelectOption>
      </NativeSelect>
    </label>
  );
}
function Note() {
  return (
    <div className="note">
      <div className="note-head">
        <span>Illustrative clinical sentence</span>
      </div>
      <p>
        Patient reports <mark className="problem">chest pain</mark>. Started on{" "}
        <mark className="treatment">aspirin</mark> and an{" "}
        <mark className="test">ECG</mark> was ordered.
      </p>
      <div className="note-key">
        <span>
          <i className="key problem" />
          Problem
        </span>
        <span>
          <i className="key treatment" />
          Treatment
        </span>
        <span>
          <i className="key test" />
          Test
        </span>
      </div>
    </div>
  );
}
function Overview({ go }: { go: (p: Page, m?: Metric) => void }) {
  return (
    <>
      <section className="hero">
        <h2 className="study-title">Variance-Aware Few-Shot Prototypes on PHI-Scrubbed Clinical Text</h2>
        <h1>
          What can a few labeled
          <br />
          mentions teach us?
        </h1>
        <p>
          Comparing point and covariance-aware prototypes for few-shot clinical
          entity classification with frozen GLiNER embeddings.
        </p>
        <Button variant="link" className="text-link" onClick={() => go("Method")}>
          Understand the method <ArrowRight size={16} />
        </Button>
      </section>
      <section className="overview-story" aria-label="Research problem and aim">
        <article>
          <div><span className="eyebrow">01 / The problem</span><h2>Clinical language is varied. Labels are expensive.</h2></div>
          <div className="overview-story-copy">
            <p>Imagine a research team with a million de-identified case files. They want to organize medical phrases across those notes as Problems, Treatments, or Tests. Manually labeling enough examples to retrain a large model every time the categories change would be expensive. A few-shot approach could make that adaptation lighter: label a small set of examples, keep the language model fixed, and update the much smaller classifier.</p>
            <p>Clinical notes describe the same condition, treatment, or test in many ways, so those few examples still have to represent varied language. In our experiment, each phrase is already marked in the note. The system uses the surrounding words to decide whether it is a <strong>Problem</strong>, <strong>Treatment</strong>, or <strong>Test</strong>.</p>
          </div>
        </article>
        <article>
          <div><span className="eyebrow">02 / The research gap</span><h2>Does the spread of examples help?</h2></div>
          <div className="overview-story-copy">
            <p>Think of each labeled phrase as a point on a map. The <strong>point prototype</strong> method keeps one average point for each type. To classify a new phrase, it compares that phrase with the three averages and chooses the closest match. This is simple, but it loses information about how varied the examples within a type are.</p>
            <p>For instance, “Problem” may include phrases as different as “chest pain” and “diabetes.” Their points may be scattered. That scatter is a type’s <strong>spread</strong>. The <strong>covariance-aware</strong> method keeps the average and estimates which parts of the numerical representation vary more or less within that type. A difference along a part that usually varies counts less than a difference along a part that stays consistent.</p>
            <p>We test whether this extra information helps classify phrases in de-identified clinical notes. We also check confidence, since a method can choose the correct type while giving a misleading confidence score. With only one example per type, there is little class-specific spread to estimate.</p>
          </div>
        </article>
        <article>
          <div><span className="eyebrow">03 / What we test</span><h2>Change the classifier, keep the comparison fair.</h2></div>
          <div className="overview-story-copy">
            <p><strong>GLiNER</strong> reads a marked phrase together with nearby words and turns it into a numerical representation, called an embedding. For example, “aspirin” is represented in the context of the sentence around it. We keep GLiNER <strong>frozen</strong>: its learned weights do not change during this experiment.</p>
            <p>Both prototype methods receive those same embeddings and the same test phrases. We compare a point prototype with a covariance-aware prototype using 1, 5, 10, and 20 labeled examples per type across 50 paired test rounds. Keeping the encoder fixed lets us focus the comparison on the prototype heads.</p>
            <p>The covariance-aware method has higher average Micro-F1 from 5 shots onward, with gains of about 2.1–2.6 percentage points; at 1 shot, they tie. This shows an improvement in <em>type classification for supplied spans</em>, while leaving open how well either method would find spans in a new note.</p>
          </div>
        </article>
      </section>
      <Note />
      <section className="main-result">
        <div className="result-header">
          <div>
            <span className="eyebrow">01 / Principal result</span>
            <h2>Performance rises with more examples.</h2>
            <p>
              Mean Micro-F1 across 50 paired held-out episodes for each shot
              count.
            </p>
          </div>
          <Legend />
        </div>
        <Chart />
        <div className="result-footer">
          <span>
            At 20 shots <strong>74.12%</strong> covariance-aware vs{" "}
            <strong>71.49%</strong> point
          </span>
          <span className="positive">+2.63 percentage points</span>
        </div>
        <Button variant="link"
          className="text-link"
          onClick={() => go("Evaluation", "Micro-F1")}
        >
          Explore the full evaluation <ArrowRight size={16} />
        </Button>
      </section>
      <div className="meta-line">
        <span>
          <b>272</b> clinical notes
        </span>
        <span>
          <b>3</b> entity types
        </span>
        <span>
          <b>50</b> paired episodes / setting
        </span>
      </div>
      <p className="footnote">
        Evaluation classifies supplied gold spans. These numbers do not measure
        independent span detection.
      </p>
    </>
  );
}
function Method() {
  const [active, setActive] = useState(0);
  return (
    <>
      <SectionTitle eyebrow="02 / Method" title="How a phrase gets its type">
        <p className="heading-copy">
          Follow one phrase through the experiment. Select a step to see what happens behind the scenes.
        </p>
      </SectionTitle>
      <div className="method-example">
        <span className="eyebrow">Example input</span>
        <p>Patient reports chest pain. Started on <mark>aspirin</mark> and an ECG was ordered.</p>
        <span>The supplied phrase is <strong>aspirin</strong>. Its known type is <strong>Treatment</strong>.</span>
      </div>
      <p className="method-summary">
        We start with clinical notes whose medical phrases are already marked. After checking and separating the data, we give each method a few labeled examples of Problem, Treatment, and Test. A fixed language model turns the phrases into numbers; the two methods use those numbers to classify new marked phrases. Finally, we compare their answers with the known labels across repeated test rounds.
      </p>
      <div className="method-intro"><span className="eyebrow">The full path · preparation to evaluation</span><p>Click any step for more detail.</p></div>
      <div className="pipeline" role="group" aria-label="Method stages">
        {methodSteps.map((s, i) => (
          <button
            key={s.name}
            className={`stage ${active === i ? "active" : ""}`}
            onClick={() => setActive(i)}
            aria-pressed={active === i}
          >
            <span className="stage-number">{String(i + 1).padStart(2, "0")}</span>
            <strong>{s.name}</strong>
            <small>{s.summary}</small>
          </button>
        ))}
      </div>
      <div className="stage-detail" aria-live="polite">
        <span className="eyebrow">
          Step {String(active + 1).padStart(2, "0")} / {String(methodSteps.length).padStart(2, "0")}
        </span>
        <h2>{methodSteps[active].name}</h2>
        <p>{methodSteps[active].detail}</p>
      </div>
      <div className="method-comparison-heading"><span className="eyebrow">At steps 06 and 07</span><h2>Two ways to compare examples</h2></div>
      <div className="branch-grid">
        <article>
          <span className="branch-line point" />
          <h3>Point prototype</h3>
          <p>
            Find the average example for each type. Choose the type whose average is closest to the new phrase.
          </p>
        </article>
        <article>
          <span className="branch-line gaussian" />
          <h3>Covariance-aware prototype</h3>
          <p>
            Find each type’s average and how much its examples vary. Use both to judge which type fits the new phrase.
          </p>
        </article>
      </div>
      <figure className="architecture">
        <figcaption><span className="eyebrow">System architecture</span><h2>How the parts connect</h2></figcaption>
        <div className="architecture-inputs">
          <div>Clinical notes<br/><small>and supplied annotations</small></div>
          <ArrowRight aria-hidden="true" size={17}/>
          <div>Validate &amp; prepare<br/><small>spans and three labels</small></div>
          <ArrowRight aria-hidden="true" size={17}/>
          <div>Split notes &amp; sample<br/><small>paired few-shot episodes</small></div>
        </div>
        <ArrowDown className="architecture-down" aria-hidden="true" size={17}/>
        <div className="architecture-encoder">Frozen GLiNER encoder <small>Known span + nearby context → shared embedding</small></div>
        <ArrowDown className="architecture-down" aria-hidden="true" size={17}/>
        <div className="architecture-heads"><div className="architecture-point">Point prototype <small>Class mean · cosine distance</small></div><div className="architecture-gaussian">Covariance-aware prototype <small>Mean + spread · Mahalanobis distance</small></div></div>
        <ArrowDown className="architecture-down" aria-hidden="true" size={17}/>
        <div className="architecture-output">Predicted type for each supplied span <small>Compare with known type → F1, calibration, and error analysis</small></div>
      </figure>
      <Collapsible className="method-technical">
        <CollapsibleTrigger className="disclosure-trigger">Technical details <ChevronDown size={15}/></CollapsibleTrigger>
        <CollapsibleContent>
          <p>Both heads receive identical normalized mention embeddings and the same support and query episodes. The point head uses the mean support embedding per class and cosine distance. The covariance-aware head estimates diagonal variance with pooled-variance shrinkage (λ = 0.8) and ridge regularization (ε = 0.001), then scores classes by squared Mahalanobis distance.</p>
          <p>The frozen encoder uses a 64-token context window. Notes are split by note into support, development, and test sets. The evaluated boundaries come from gold annotations; there is no non-entity class or end-to-end span detection score.</p>
        </CollapsibleContent>
      </Collapsible>
    </>
  );
}
function Table({
  heads,
  rows,
}: {
  heads: string[];
  rows: (string | number)[][];
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {heads.map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j}>{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function Evaluation({
  initial,
  onExplore,
}: {
  initial: Metric;
  onExplore: () => void;
}) {
  const [metric, setMetric] = useState<Metric>(initial),
    [shot, setShot] = useState<Shot>(20),
    [method, setMethod] = useState<Method>("gaussian"),
    [winMetric, setWinMetric] = useState<"micro" | "macro">("micro");
  const wins = paired(shot, winMetric);
  const entityPoint = perEntity(shot, "point"),
    entityGauss = perEntity(shot, "gaussian");
  return (
    <>
      <SectionTitle eyebrow="03 / Evaluation" title="The evidence, in detail">
        <p className="heading-copy">{source}</p>
      </SectionTitle>
      <div className="eval-layout">
        <nav className="metric-nav" aria-label="Evaluation metrics">
          {metrics.map((m, i) => (
            <button
              key={m}
              className={metric === m ? "selected" : ""}
              onClick={() => setMetric(m)}
              aria-current={metric === m ? "page" : undefined}
            >
              <span>{String(i + 1).padStart(2, "0")}</span>
              {m}
              <ChevronRight size={14} />
            </button>
          ))}
        </nav>
        <div className="eval-content">
          <div className="eval-top">
            <div>
              <span className="eyebrow">
                Metric {String(metrics.indexOf(metric) + 1).padStart(2, "0")} /
                08
              </span>
              <h2>{metric}</h2>
            </div>
            <Legend />
          </div>
          {metric === "Micro-F1" && (
            <>
              <p>
                Share of correctly classified query spans across the three
                classes, averaged over episodes.
              </p>
              <Chart />
              <Table
                heads={["Support", "Point", "Covariance-aware", "Difference"]}
                rows={shots.map((s) => [
                  `${s} shot`,
                  pct(result(s, "point").metrics.f1.mean),
                  pct(result(s, "gaussian").metrics.f1.mean),
                  pp(
                    result(s, "gaussian").metrics.f1.mean -
                      result(s, "point").metrics.f1.mean,
                  ),
                ])}
              />
            </>
          )}
          {metric === "Macro-F1" && (
            <>
              <p>
                Each entity class contributes equally within an episode; episode
                Macro-F1 values are then averaged.
              </p>
              <Chart kind="macro" />
              <Table
                heads={["Support", "Point", "Covariance-aware", "Difference"]}
                rows={shots.map((s) => [
                  `${s} shot`,
                  pct(result(s, "point").metrics.macro_f1.mean),
                  pct(result(s, "gaussian").metrics.macro_f1.mean),
                  pp(
                    result(s, "gaussian").metrics.macro_f1.mean -
                      result(s, "point").metrics.macro_f1.mean,
                  ),
                ])}
              />
            </>
          )}
          {metric === "Mean ± SD" && (
            <>
              <p>
                Standard deviation describes variation across 50 episodes. It is
                neither a confidence interval nor individual prediction
                uncertainty.
              </p>
              <div className="controls">
                <label className="select-label">
                  Score{" "}
                  <NativeSelect
                    value={winMetric}
                    onChange={(e) =>
                      setWinMetric(e.target.value as "micro" | "macro")
                    }
                  >
                    <NativeSelectOption value="micro">Micro-F1</NativeSelectOption>
                    <NativeSelectOption value="macro">Macro-F1</NativeSelectOption>
                  </NativeSelect>
                </label>
              </div>
              <Table
                heads={[
                  "Support",
                  "Point mean ± SD",
                  "Covariance mean ± SD",
                  "Episodes",
                ]}
                rows={shots.map((s) => {
                  const k = winMetric === "micro" ? "f1" : "macro_f1";
                  return [
                    `${s} shot`,
                    `${pct(result(s, "point").metrics[k].mean)} ± ${pct(result(s, "point").metrics[k].std)}`,
                    `${pct(result(s, "gaussian").metrics[k].mean)} ± ${pct(result(s, "gaussian").metrics[k].std)}`,
                    50,
                  ];
                })}
              />
              <p className="data-note">
                Calculated by the experiment pipeline from the saved
                episode-level scores; source records are linked below.
              </p>
            </>
          )}
          {metric === "Win rate" && (
            <>
              <p>
                A win means the covariance-aware head has strictly higher F1 on
                the same paired episode. Ties are counted in the denominator and
                shown separately.
              </p>
              <div className="controls">
                <SelectShot value={shot} onChange={setShot} />
                <label className="select-label">
                  Score{" "}
                  <NativeSelect
                    value={winMetric}
                    onChange={(e) =>
                      setWinMetric(e.target.value as "micro" | "macro")
                    }
                  >
                    <NativeSelectOption value="micro">Micro-F1</NativeSelectOption>
                    <NativeSelectOption value="macro">Macro-F1</NativeSelectOption>
                  </NativeSelect>
                </label>
              </div>
              <div className="big-stat">
                {pct(wins.rate, 0)}
                <span>
                  {wins.wins} wins / {wins.total} paired episodes
                </span>
              </div>
              <div
                className="stackbar"
                aria-label={`${wins.wins} wins, ${wins.losses} losses, ${wins.ties} ties`}
              >
                <i style={{ width: `${(wins.wins / wins.total) * 100}%` }} />
                <i style={{ width: `${(wins.ties / wins.total) * 100}%` }} />
                <i style={{ width: `${(wins.losses / wins.total) * 100}%` }} />
              </div>
              <div className="bar-key">
                <span>{wins.wins} wins</span>
                <span>{wins.ties} ties</span>
                <span>{wins.losses} losses</span>
              </div>
              <p className="data-note">
                Derived from corresponding entries in the two saved episode
                arrays. Episodes may overlap and are not independent trials.
              </p>
            </>
          )}
          {metric === "ECE" && (
            <>
              <p>
                Expected Calibration Error compares confidence with observed
                accuracy in 10 uniform confidence bins. Lower is better.
              </p>
              <Chart kind="ece" />
              <Table
                heads={["Support", "Point", "Covariance-aware", "Lower ECE"]}
                rows={shots.map((s) => {
                  const p = result(s, "point").calibration.ece,
                    g = result(s, "gaussian").calibration.ece;
                  return [
                    `${s} shot`,
                    p.toFixed(3),
                    g.toFixed(3),
                    `${p < g ? "Point" : "Covariance"} by ${Math.abs(p - g).toFixed(3)}`,
                  ];
                })}
              />
              <Collapsible className="disclosure">
                <CollapsibleTrigger className="disclosure-trigger">
                  View archived reliability diagram <ChevronDown size={15} />
                </CollapsibleTrigger>
                <CollapsibleContent>
                <div className="controls">
                  <SelectShot value={shot} onChange={setShot} />
                  <SelectMethod value={method} onChange={setMethod} />
                </div>
                <img
                  className="archive-plot"
                  src={plotUrl("calibration", method, shot)}
                  alt={`Archived ${shot}-shot ${methodName(method)} reliability diagram`}
                />
                <p className="data-note">
                  Archived image from the held-out run. Bin-level values are not
                  stored separately.
                </p>
                </CollapsibleContent>
              </Collapsible>
            </>
          )}
          {metric === "Per-entity F1" && (
            <>
              <p>
                Mean per-class F1 across 50 episodes. Problem has the lowest F1
                at 20 shots and the largest improvement.
              </p>
              <div className="controls">
                <SelectShot value={shot} onChange={setShot} />
              </div>
              <div className="entity-list">
                {classes.map((c, i) => (
                  <div className="entity-row" key={c}>
                    <div className="entity-head">
                      <strong>{c}</strong>
                      <span>
                        {pp(entityGauss[i].value - entityPoint[i].value)}
                      </span>
                    </div>
                    <div className="entity-bars">
                      <span
                        className="bar point"
                        style={{ width: `${entityPoint[i].value * 100}%` }}
                      />
                      <span
                        className="bar gaussian"
                        style={{ width: `${entityGauss[i].value * 100}%` }}
                      />
                    </div>
                    <small>
                      {pct(entityPoint[i].value)} / {pct(entityGauss[i].value)}
                    </small>
                  </div>
                ))}
              </div>
              <Legend />
              <p className="data-note">
                Per-class episode F1 scores are averaged, rather than calculated
                from pooled predictions.
              </p>
            </>
          )}
          {metric === "Confusion matrix" && (
            <>
              <p>
                The archived plots show row-normalized classification outcomes.
                Rows are actual classes; columns are predicted classes.
                Off-diagonal cells are errors.
              </p>
              <div className="controls">
                <SelectShot value={shot} onChange={setShot} />
                <SelectMethod value={method} onChange={setMethod} />
              </div>
              <img
                className="archive-plot"
                src={plotUrl("confusion", method, shot)}
                alt={`Archived row-normalized confusion matrix for ${methodName(method)} at ${shot} shots; actual classes on rows, predicted classes on columns`}
              />
              <p className="data-note">
                Only the normalized image was archived. Raw confusion counts and
                an interactive count view are unavailable without saved query
                labels and predictions.
              </p>
            </>
          )}
          {metric === "Examples" && (
            <>
              <p>
                Example-level test predictions and confidence scores were not
                archived. The available result files contain episode summaries
                and aggregate plots.
              </p>
              <div className="empty">
                <Info size={22} />
                <strong>Recorded examples unavailable</strong>
                <span>
                  To populate this view, save query sentence, span offsets, gold
                  label, predicted label, model confidence, and episode/method
                  identifiers during evaluation.
                </span>
                <Button variant="link" className="text-link" onClick={onExplore}>
                  View the span selection explainer <ArrowRight size={15} />
                </Button>
              </div>
            </>
          )}
          <div className="source-line">
            <BookOpen size={15} /> Source:{" "}
            <code>results/experiment/test/metrics/</code> and archived plots
          </div>
        </div>
      </div>
    </>
  );
}
function Explore() {
  const [mention, setMention] = useState<"chest pain" | "aspirin" | "ECG">(
      "chest pain",
    ),
    [shot, setShot] = useState<Shot>(20),
    [method, setMethod] = useState<Method>("gaussian");
  const expectedType = {"chest pain": "Problem", aspirin: "Treatment", ECG: "Test"}[mention];
  const pointGuess = {"chest pain": "Treatment", aspirin: "Problem", ECG: "Problem"}[mention];
  const illustrativeGuess = method === "point" ? pointGuess : expectedType;
  const recordedF1 = result(shot, method).metrics.f1.mean;
  return (
    <>
      <SectionTitle eyebrow="04 / Explore" title="A supplied span, in context">
        <p className="heading-copy">
          Select a mention and compare it with the archived results for each evaluation setting.
        </p>
      </SectionTitle>
      <div className="explore-panel">
        <div className="controls">
          <label className="select-label">
            Selected mention{" "}
            <NativeSelect
              value={mention}
              onChange={(e) => setMention(e.target.value as typeof mention)}
            >
              <NativeSelectOption value="chest pain">chest pain</NativeSelectOption>
              <NativeSelectOption value="aspirin">aspirin</NativeSelectOption>
              <NativeSelectOption value="ECG">ECG</NativeSelectOption>
            </NativeSelect>
          </label>
          <SelectShot value={shot} onChange={setShot} />
          <SelectMethod value={method} onChange={setMethod} />
        </div>
        <div className="sample-text">
          Patient reports{" "}
          {mention === "chest pain" ? <mark>chest pain</mark> : "chest pain"}.
          Started on {mention === "aspirin" ? <mark>aspirin</mark> : "aspirin"}{" "}
          and an {mention === "ECG" ? <mark>ECG</mark> : "ECG"} was ordered.
        </div>
        <div className="explore-grid">
          <div>
            <span className="eyebrow">Input</span>
            <strong>“{mention}”</strong>
            <small>Explicitly selected span · synthetic sentence</small>
          </div>
          <div>
            <span className="eyebrow">Illustrative {methodName(method)} guess</span>
            <strong>{illustrativeGuess} <span className={method === "point" ? "guess-wrong" : "guess-right"}>{method === "point" ? "Incorrect" : "Correct"}</span></strong>
            <small>
              Point: {pointGuess} (incorrect) · Covariance-aware: {expectedType} (correct)
            </small>
          </div>
        </div>
        <div className="explore-summary">
          <span>Recorded mean Micro-F1 · {methodName(method)} · {shot} shots</span>
          <strong>{pct(recordedF1)}</strong>
        </div>
        <p className="footnote">
          Demo mode: classify a supplied entity span. The guesses above are illustrative, not recorded model predictions. Micro-F1 is an aggregate across archived test episodes.
        </p>
      </div>
    </>
  );
}
function Protocol() {
  return (
    <>
      <SectionTitle eyebrow="05 / Protocol" title="How the study was run">
        <p className="heading-copy">
          A compact account of data, evaluation, and limits.
        </p>
      </SectionTitle>
      <div className="protocol-grid">
        <div>
          <span className="eyebrow">Dataset</span>
          <h3>272 discharge notes</h3>
          <p>
            75,490 input annotations; 75,442 retained after validation. The
            project uses Problem, Treatment, and Test labels.
          </p>
        </div>
        <div>
          <span className="eyebrow">Split</span>
          <h3>60 / 20 / 20</h3>
          <p>
            Support, development, and held-out test split at note level;
            identical note texts remain together.
          </p>
        </div>
        <div>
          <span className="eyebrow">Evaluation</span>
          <h3>50 paired episodes</h3>
          <p>
            At each of 1, 5, 10, and 20 shots, both heads evaluate the same
            episodes with 50 queries per type.
          </p>
        </div>
      </div>
      <Separator className="mt-8 bg-border" />
      <div className="protocol-bottom">
        <section>
          <span className="eyebrow">Interpretation</span>
          <h2>What the results establish</h2>
          <p>
            Covariance-aware prototypes have higher average Micro-F1 at 5, 10,
            and 20 shots. The results describe gold-span type classification.
            Aggregate improvements alone do not establish statistical
            significance.
          </p>
          <p>
            ECE uses 10 uniform bins and depends on this evaluation convention.
            Archived confusion plots are normalized images; example-level test
            predictions were not saved.
          </p>
        </section>
        <section>
          <span className="eyebrow">Reproducibility</span>
          <h2>Known limits</h2>
          <p>
            Clinical data requires authorized access. The exact SNOMED
            concept-to-project-label mapping used to prepare this run is absent
            from the bundle. This is a research prototype, not a clinical
            decision system.
          </p>
          <p>
            Encoder: frozen GLiNER Medium v2.1. Prototype settings: diagonal
            covariance, shrinkage 0.8, regularization 0.001.
          </p>
        </section>
      </div>
    </>
  );
}
export default function App() {
  const [page, setPage] = useState<Page>("Overview"),
    [initialMetric, setInitialMetric] = useState<Metric>("Micro-F1");
  function go(p: Page, m?: Metric) {
    if (m) setInitialMetric(m);
    setPage(p);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  return (
    <div className="app min-h-screen bg-background text-foreground">
      <header className="site-header">
        <nav aria-label="Main navigation">
          {pages.map((p) => (
            <button
              key={p}
              className={page === p ? "current" : ""}
              onClick={() => go(p)}
              aria-current={page === p ? "page" : undefined}
            >
              {p}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {page === "Overview" ? (
          <Overview go={go} />
        ) : page === "Method" ? (
          <Method />
        ) : page === "Evaluation" ? (
          <Evaluation
            key={initialMetric}
            initial={initialMetric}
            onExplore={() => go("Explore")}
          />
        ) : page === "Explore" ? (
          <Explore />
        ) : (
          <Protocol />
        )}
      </main>
      <footer>
        <span>Clinical entity classification / Research demo</span>
        <span>
          Gold spans · Frozen encoder · Archived experiment{" "}
          <ArrowUpRight size={13} />
        </span>
      </footer>
    </div>
  );
}
