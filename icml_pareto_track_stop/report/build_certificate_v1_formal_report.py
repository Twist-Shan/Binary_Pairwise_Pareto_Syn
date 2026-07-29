"""Build the certificate-v1 formal two-algorithm TeX report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vb_ege.compat import import_pandas_quietly


pd = import_pandas_quietly()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARY_DIR = ROOT / "results" / "summary_certificate"
FIGURE_DIR = ROOT / "results" / "figures_certificate"
SUMMARY = SUMMARY_DIR / "formal_two_way_summary.csv"
PAIRED = SUMMARY_DIR / "formal_two_way_paired.csv"
DIAGNOSTICS = SUMMARY_DIR / "formal_two_way_diagnostics.csv"
OUTPUT = HERE / "formal_certificate_v1_report.tex"

VB = "VB-EGE-practical"
TRACK = "Pareto BT-GLR Track-and-Stop"

BENCHMARK_LABELS = {
    "fc_convex2d": "Convex-2D",
    "fc_convex3d": "Convex-3D",
    "fc_arena4_small": "Arena-4 small",
    "fc_arena4_medium": "Arena-4 medium",
    "fc_arena10_medium": "Arena-10 medium",
    "fc_witness4": "Witness-4",
    "fc_witness10": "Witness-10",
    "fc_twogroup10_medium": "Two-group-10",
}


def sci_tex(value: float, digits: int = 2) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{digits}f}\!\times\!10^{{{exponent}}}"


def mean_se_tex(mean: float, se: float) -> str:
    return rf"${sci_tex(mean)}\pm{sci_tex(se)}$"


def _cell_pairs(summary, section: str):
    frame = summary[summary["section"] == section]
    keys = ["experiment_id", "params_key", "K", "d", "delta"]
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        algorithms = group.set_index("algorithm")
        yield key, algorithms.loc[VB], algorithms.loc[TRACK]


def _cell_label(
    section: str,
    experiment: str,
    params: dict,
    delta: float,
) -> str:
    if section == "fixed_confidence_scaling":
        if experiment == "fc_K_scaling_d4":
            return rf"$K={int(params['K'])}$"
        if experiment == "fc_d_scaling_K64":
            return rf"$d={int(params['d'])}$"
        return rf"$\Delta={float(params['Delta']):g}$"
    if section == "confidence_scaling_quantiles":
        prefix = "Sym." if "symmetric" in experiment else "Arena-10"
        return rf"{prefix}, $\delta={delta:g}$"
    if section == "fixed_confidence_benchmarks":
        return BENCHMARK_LABELS[experiment]
    if section == "pareto_size_ablation":
        return rf"$d={int(params['d'])},\ |P|={int(params['s'])}$"
    return experiment.replace("_", r"\_")


def _ordered_rows(summary, section: str):
    rows = []
    for (experiment, params_key, _K, _d, delta), vb, track in _cell_pairs(
        summary,
        section,
    ):
        params = json.loads(params_key)
        rows.append(
            {
                "label": _cell_label(section, experiment, params, delta),
                "experiment": experiment,
                "params": params,
                "delta": float(delta),
                "n": int(vb.n),
                "vb": vb,
                "track": track,
                "ratio": float(track.mean_tau / vb.mean_tau),
            }
        )
    if section == "fixed_confidence_scaling":
        order = {
            "fc_K_scaling_d4": 0,
            "fc_d_scaling_K64": 1,
            "fc_gap_scaling_K64_d10": 2,
        }
        rows.sort(
            key=lambda row: (
                order[row["experiment"]],
                row["params"].get(
                    "K",
                    row["params"].get("d", row["params"].get("Delta", 0)),
                ),
            )
        )
    elif section == "confidence_scaling_quantiles":
        experiment_order = {
            "conf_quantile_symmetric_K64_d10": 0,
            "conf_quantile_arena10_medium": 1,
        }
        rows.sort(
            key=lambda row: (
                experiment_order[row["experiment"]],
                -row["delta"],
            )
        )
    elif section == "fixed_confidence_benchmarks":
        order = {name: index for index, name in enumerate(BENCHMARK_LABELS)}
        rows.sort(key=lambda row: order[row["experiment"]])
    elif section == "pareto_size_ablation":
        rows.sort(
            key=lambda row: (
                int(row["params"]["d"]),
                int(row["params"]["s"]),
            )
        )
    return rows


def result_table(summary, section: str, caption: str, label: str) -> str:
    rows = _ordered_rows(summary, section)
    body = []
    for row in rows:
        vb = row["vb"]
        track = row["track"]
        errors = (
            f"{int(vb.errors_stopped)}/{int(vb.n_stopped)};"
            f"{int(track.errors_stopped)}/{int(track.n_stopped)}"
        )
        body.append(
            " & ".join(
                [
                    row["label"],
                    str(row["n"]),
                    mean_se_tex(vb.mean_tau, vb.se_tau),
                    mean_se_tex(track.mean_tau, track.se_tau),
                    f"{row['ratio']:.3g}",
                    errors,
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[H]
\centering
\caption{{{caption}}}
\label{{{label}}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Cell & $n$ & VB-EGE mean $\pm$ SE & Pareto mean $\pm$ SE &
$\tau_P/\tau_V$ & Errors V;P \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def section_interpretation(summary, section: str) -> str:
    rows = _ordered_rows(summary, section)
    ratios = np.array([row["ratio"] for row in rows], dtype=float)
    best = min(rows, key=lambda row: row["ratio"])
    worst = max(rows, key=lambda row: row["ratio"])
    wins = int(np.sum(ratios < 1.0))
    geometric = float(np.exp(np.mean(np.log(ratios))))
    return (
        rf"\paragraph{{Result.}} The Pareto BT-GLR heuristic has the smaller "
        rf"mean stopping time in {wins} of {len(rows)} cells. Its "
        rf"Pareto/VB-EGE mean ratio ranges from {best['ratio']:.3g} "
        rf"({best['label']}) to {worst['ratio']:.3g} ({worst['label']}), "
        rf"with geometric mean {geometric:.3g} across the section. "
        rf"This is a finite-protocol comparison; it is not an optimality claim."
    )


def confidence_fit_table(summary) -> str:
    frame = summary[
        summary["section"] == "confidence_scaling_quantiles"
    ].copy()
    rows = []
    for (experiment, algorithm), group in frame.groupby(
        ["experiment_id", "algorithm"],
        sort=False,
    ):
        x = np.log(1.0 / group["delta"].to_numpy(dtype=float))
        y = group["mean_tau"].to_numpy(dtype=float)
        if np.ptp(y) <= 1e-10 * max(float(np.max(np.abs(y))), 1.0):
            slope = 0.0
            r2_text = "--"
        else:
            design = np.column_stack([np.ones_like(x), x])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ coefficients
            residual = float(np.sum((y - fitted) ** 2))
            total = float(np.sum((y - y.mean()) ** 2))
            slope = float(coefficients[1])
            r2_text = f"{1.0 - residual / total:.4f}"
        setting = "Symmetric" if "symmetric" in experiment else "Arena-10"
        method = "VB-EGE" if algorithm == VB else r"Pareto BT-GLR T\&S"
        rows.append(
            f"{setting} & {method} & ${sci_tex(slope)}$ & "
            f"{r2_text} \\\\"
        )
    return r"""
\begin{table}[H]
\centering
\caption{Descriptive fits $\bar\tau=a+b\log(1/\delta)$. The slope is not a
claimed asymptotic constant for the Pareto heuristic. $R^2$ is undefined,
shown as --, when the response is numerically constant.}
\label{tab:confidence-fits}
\small
\begin{tabular}{llrr}
\toprule
Setting & Method & Slope $b$ & $R^2$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def reliability_table(diagnostics) -> str:
    section_labels = {
        "fixed_confidence_scaling": "Scaling",
        "confidence_scaling_quantiles": "Confidence",
        "fixed_confidence_benchmarks": "Benchmarks",
        "pareto_size_ablation": "Pareto size",
    }
    body = []
    for row in diagnostics.itertuples(index=False):
        method = "VB-EGE" if row.algorithm == VB else "Pareto BT-GLR"
        body.append(
            f"{section_labels[row.section]} & {method} & {int(row.n)} & "
            f"{row.stop_rate:.4f} & "
            f"{int(row.errors_stopped)}/{int(row.n_stopped)} & "
            f"{row.wilson95_error_upper:.4f} & "
            f"{row.mle_convergence_rate:.4f} \\\\"
        )
    return r"""
\begin{table}[H]
\centering
\caption{Stopping and empirical-error audit. Wilson upper is the two-sided
95\% Wilson upper endpoint for the conditional error rate among stopped runs.}
\label{tab:reliability}
\small
\begin{tabular}{llrrrrr}
\toprule
Section & Method & $n$ & Stop rate & Errors & Wilson upper & MLE conv. \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def build() -> None:
    summary = pd.read_csv(SUMMARY)
    paired = pd.read_csv(PAIRED)
    diagnostics = pd.read_csv(DIAGNOSTICS)

    total_per_algorithm = int(
        summary[summary["algorithm"] == VB]["n"].sum()
    )
    total_runs = 2 * total_per_algorithm
    vb = summary[summary["algorithm"] == VB]
    track = summary[summary["algorithm"] == TRACK]
    vb_errors = int(vb["errors_stopped"].sum())
    track_errors = int(track["errors_stopped"].sum())
    vb_capped = int((vb["n"] - vb["n_stopped"]).sum())
    track_capped = int((track["n"] - track["n_stopped"]).sum())
    vb_min_stop = float(vb["stop_rate"].min())
    track_min_stop = float(track["stop_rate"].min())
    track_min_convergence = float(track["mle_convergence_rate"].min())

    all_rows = []
    for section in [
        "fixed_confidence_scaling",
        "confidence_scaling_quantiles",
        "fixed_confidence_benchmarks",
        "pareto_size_ablation",
    ]:
        all_rows.extend(_ordered_rows(summary, section))
    ratios = np.array([row["ratio"] for row in all_rows], dtype=float)
    track_wins = int(np.sum(ratios < 1.0))
    global_geometric_ratio = float(np.exp(np.mean(np.log(ratios))))

    track_raw_diagnostics = diagnostics[diagnostics["algorithm"] == TRACK]
    certified_rate = float(
        track_raw_diagnostics["certified_statistic_rate"].fillna(0.0).mean()
    )
    profile_modes = sorted(
        {
            item.split(":", 1)[0]
            for value in track_raw_diagnostics["profile_modes"].dropna()
            for item in str(value).split("; ")
            if item
        }
    )
    profile_mode_text = "; ".join(profile_modes).replace("_", r"\_")

    scaling_table = result_table(
        summary,
        "fixed_confidence_scaling",
        "Fixed-confidence scaling at $\\delta=0.05$. Means and standard "
        "errors are computed from independent latent-instance units.",
        "tab:scaling",
    )
    confidence_table = result_table(
        summary,
        "confidence_scaling_quantiles",
        "Confidence-scaling quantiles. The same latent replication is reused "
        "across $\\delta$ values; each displayed SE is computed within its "
        "fixed-$\\delta$ cell.",
        "tab:confidence",
    )
    benchmark_table = result_table(
        summary,
        "fixed_confidence_benchmarks",
        "Eight fixed-confidence benchmark settings. For instance banks with "
        "multiple observation replicates, SE is clustered by latent instance.",
        "tab:benchmarks",
    )
    pareto_table = result_table(
        summary,
        "pareto_size_ablation",
        "Pareto-front-size ablation at $K=64$. Means and one standard error "
        "use latent instances as independent units.",
        "tab:pareto-size",
    )

    document = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.82in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,array,graphicx,float,xcolor,hyperref}}
\usepackage[T1]{{fontenc}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}
\title{{Formal Fixed-Confidence Synthetic Comparison:\\
VB-EGE versus Pareto-Specific BT-GLR Track-and-Stop}}
\author{{Synthetic experiment report}}
\date{{July 28, 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We compare VB-EGE with a Pareto-specific heuristic extension of the
Bradley--Terry generalized-likelihood-ratio Track-and-Stop method of
Goldberger and Rudi. The experiment contains {total_runs:,} algorithm runs,
{total_per_algorithm:,} per method, and exactly reuses the formal synthetic
settings and repetition counts used for VB-EGE: fixed-confidence scaling,
confidence-scaling quantiles, eight benchmark families, and Pareto-front-size
ablation. Both methods identify the all-coordinate-strict Pareto set and are
evaluated by mean stopping time, one standard error, stopping rate, and set
error. The Pareto heuristic has smaller mean stopping time in {track_wins} of
{len(all_rows)} parameter cells; its geometric-mean stopping-time ratio to
VB-EGE is {global_geometric_ratio:.3g}. Observed set errors are
{vb_errors} for VB-EGE and {track_errors} for the Pareto heuristic. The latter
has {track_capped} runs that reach the predeclared $10^{{18}}$ cap without
stopping, versus {vb_capped} for VB-EGE. It is an explicitly documented
high-dimensional extension, not a transfer of the scalar top-$k$ correctness
or optimality theorem.
\end{{abstract}}

\section{{Protocol, target, and reported quantities}}

\subsection{{Observation model and Pareto convention}}

There are $K$ arms and $d$ objectives. Arm $i$ has utility vector
$\theta_i\in\mathbb R^d$. A query of unordered pair $\{{i,j\}}$ on objective
$r$ returns
\[
Y\sim\operatorname{{Bernoulli}}\!\left(p_{{ij,r}}\right),\qquad
p_{{ij,r}}=\sigma(\theta_{{i,r}}-\theta_{{j,r}}),\qquad
\sigma(x)=\frac{{1}}{{1+e^{{-x}}}}.
\]
We use the main algorithm's all-coordinate-strict convention:
\[
j\succ i
\quad\Longleftrightarrow\quad
\theta_{{j,r}}>\theta_{{i,r}}\quad\text{{for every }}r\in[d].
\]
The target is
$\mathcal P(\theta)=\{{i:\nexists j\ne i\text{{ with }}j\succ i\}}$.
This convention is used in instance generation, both stopping rules, and
error evaluation. Equality is therefore not silently replaced by ordinary
weak Pareto dominance.

\subsection{{Fixed-confidence endpoint and metrics}}

For target failure probability $\delta$, a run stops at query count $\tau$ and
returns $\widehat{{\mathcal P}}$. Its set error is
\[
E=\mathbf 1\{{\widehat{{\mathcal P}}\ne\mathcal P(\theta)\}}.
\]
The primary ordinate in every result figure is the arithmetic mean
$\bar\tau=n^{{-1}}\sum_u\tau_u$. The shaded ribbon is
$\bar\tau\pm\operatorname{{SE}}(\bar\tau)$, where
\[
\operatorname{{SE}}(\bar\tau)
=\frac{{s_\tau}}{{\sqrt{{n_{{\rm ind}}}}}}.
\]
Here $n_{{\rm ind}}$ counts independent latent instances. When a benchmark has
several observation replicates for one latent instance, we first average
within instance and compute SE across instance means. Thus the ribbon does not
mistreat repeated observations of one $\theta$ as independent instances.
The benchmark bars use the same mean and SE definition; their darker overlays
mark the interval $\bar\tau\pm\operatorname{{SE}}$.
If every independent unit in a cell stops at the same geometric phase, the
empirical SE is exactly zero and the ribbon correctly collapses onto the mean
line; no artificial visual width is added.

A run that does not stop before the predeclared maximum is recorded at
$\tau_{{\max}}=10^{{18}}$. Consequently, a cell containing capped runs reports the
cap-coded mean
\[
\bar\tau_{{\rm cap}}=\frac1n\sum_u\min\{{\tau_u,\tau_{{\max}}\}}.
\]
This is a lower bound on the uncapped mean stopping time, not an estimate that
pretends the run stopped at the cap. Its SE only describes dispersion of these
cap-coded observations. The stopping-rate table and annotations on the
benchmark figure identify every affected cell.

\subsection{{Formal experiment bank}}

\begin{{table}}[H]
\centering
\caption{{Formal protocol counts. Counts are per algorithm.}}
\label{{tab:protocol}}
\small
\begin{{tabular}}{{lrrr}}
\toprule
Section & Cells & Repetitions per cell & Runs \\
\midrule
Fixed-confidence scaling & 12 & 500 & 6,000 \\
Confidence-scaling quantiles & 14 & 500 Sym.; 300 Arena & 5,800 \\
Fixed-confidence benchmark suite & 8 & 500 & 4,000 \\
Pareto-size ablation & 8 & 300 & 2,400 \\
\midrule
Total & 42 & -- & 18,200 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Instance families}}

The Symmetric generator places one arm at the zero vector and every other arm
at $-\Delta\mathbf 1$, then randomly permutes arm identities; hence the true
Pareto size is one. Arena instances draw $s$ mutually non-dominated anchors
from a rescaled Dirichlet trade-off surface and generate every remaining arm
by subtracting an independent coordinate-wise margin from one anchor.
Convex-2D uses 15 evenly spaced points on a decreasing line segment as its
frontier; Convex-3D uses 15 Dirichlet trade-off anchors. Witness instances give
each frontier arm four generated arms for which it is the unique frontier
dominator. Two-group-10 draws 48 ``low'' arms independently from
$[0.20,0.45]^{{10}}$ and 16 ``high'' arms from $[0.55,0.75]^{{10}}$; its true
frontier size is computed from the draw rather than imposed.

\begin{{table}}[H]
\centering
\caption{{Exact benchmark-suite generator settings. All use $\delta=0.05$ and
500 algorithm runs.}}
\label{{tab:benchmark-settings}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llll}}
\toprule
Setting & $(K,d,s)$ or group sizes & Generator & Additional parameters \\
\midrule
Convex-2D & $(60,2,15)$ & convex frontier & margins $[0.03,0.18]$ \\
Convex-3D & $(60,3,15)$ & Dirichlet frontier & $\alpha=1$, margins $[0.03,0.18]$ \\
Arena-4 small & $(32,4,8)$ & trade-off arena & $\alpha=0.7$, margins $[0.08,0.25]$ \\
Arena-4 medium & $(64,4,12)$ & trade-off arena & $\alpha=0.7$, margins $[0.08,0.25]$ \\
Arena-10 medium & $(64,10,16)$ & trade-off arena & $\alpha=0.7$, margins $[0.06,0.20]$ \\
Witness-4 & $(40,4,8)$ & unique witness & $q/p=4$, margins $[0.04,0.16]$ \\
Witness-10 & $(80,10,16)$ & unique witness & $q/p=4$, margins $[0.035,0.14]$ \\
Two-group-10 & $(48+16,10,\text{{drawn}})$ & high/low groups & intervals stated above \\
\bottomrule
\end{{tabular}}
\end{{table}}

Convex-3D and Arena-10 medium use banks of 50 independently generated latent
instances with 10 observation replicates per instance. Every other benchmark
uses 500 independently generated latent instances. This distinction is why
their reported SEs are clustered by latent instance.

All paired rows share the same latent instance $\theta$ but use independent,
deterministic observation seeds. There is no result-dependent filtering and no
per-setting retuning. Fixed-confidence scaling and all benchmark/ablation
cells use $\delta=0.05$. Confidence scaling varies $\delta$ while reusing the
same latent instance within a replication.

\section{{Algorithms}}

\subsection{{VB-EGE}}

VB-EGE samples a focal arm $i$ and objective $r$ against a uniformly random
opponent. The resulting coordinate-wise Vector-Borda target is
\[
b_{{i,r}}=\frac{{1}}{{K-1}}\sum_{{j\ne i}}
\sigma(\theta_{{i,r}}-\theta_{{j,r}}).
\]
At phase $m$, it uses $\varepsilon_m=2^{{-m}}$,
\[
L_m=\log\!\left(\frac{{c_{{\log}}Kdm^2}}{{\delta}}\right),\qquad
n_m=\left\lceil\frac{{c_sL_m}}{{\varepsilon_m^2}}\right\rceil,
\]
and compares empirical accept/reject gaps with
$c_\theta\sqrt{{L_m/(2n_m)}}$. The practical constants
$(c_s,c_\theta,c_{{\log}})=(2,4,4)$ are fixed in every cell. These practical
constants differ from conservative proof constants but were selected before
the experiment, checked by the separate constant-sensitivity study, and then
held fixed throughout all sections.

\subsection{{Scalar BT-GLR Track-and-Stop source}}

Goldberger and Rudi \cite{{goldberger2026}} study scalar top-$k$
identification. Their algorithm fits a constrained Bradley--Terry MLE,
profiles the likelihood over alternatives that reverse a top-$k$ boundary,
tracks a max--min information allocation, forces exploration, and stops when a
GLR statistic exceeds a time-uniform mixture threshold. The theorem is for the
scalar top-$k$ alternative geometry. A Pareto frontier is not characterized by
one inside/outside boundary pair, so that theorem does not directly specify a
$d$-objective algorithm.

\subsection{{Pareto-specific extension used here}}

\paragraph{{Coordinate-wise BT MLE.}}
For each objective $r$, all sampled opponent identities are retained and an
independent centered, box-constrained BT MLE is fitted:
\[
\widehat\theta_r\in\arg\max_\vartheta\ell_{{r,t}}(\vartheta)
\quad\text{{s.t.}}\quad
\mathbf 1^\top\vartheta=0,\qquad\|\vartheta\|_\infty\le B,\qquad B=2.
\]
This is the same pair-objective transcript used by the Track-and-Stop sampler;
it is not a Borda reduction. The box contains every centered formal instance
and prevents complete-separation non-existence.

\paragraph{{Frontier-changing alternatives.}}
Let $\widehat P=\mathcal P(\widehat\theta)$. In the closure of the strict
alternative set, removing $p\in\widehat P$ can be witnessed by another current
frontier arm $p'$ satisfying
\[
\lambda_{{p',r}}\ge\lambda_{{p,r}}\quad\forall r.
\]
Adding $q\notin\widehat P$ requires breaking every current frontier arm as a
possible dominator:
\[
\forall p\in\widehat P,\quad
\exists r\in[d]\ \text{{such that}}\
\lambda_{{q,r}}\ge\lambda_{{p,r}}.
\]
Equivalently, an add branch is a union over $d^{{|\widehat P|}}$ witness
assignments $a:\widehat P\to[d]$. This remove/add decomposition follows the
Pareto-front identification geometry used by
Cr\'epon, Garivier, and Koolen \cite{{crepon2024}}.

\paragraph{{Exact small path and formal large-front path.}}
The implementation contains an exact certificate-enumeration path for
$K\le8$ when all add assignments fit under the enumeration cap. It profiles
the original BT likelihood under each grouped set of order constraints and
uses numerical primal--dual bounds. That path is covered by unit and smoke
tests, but none of the present formal cells uses it: all have $K\ge16$.

For formal-size fronts, we first compute local-Fisher order costs
\[
q_{{ab,r}}=
\frac{{[\widehat\theta_{{b,r}}-\widehat\theta_{{a,r}}]_+^2}}
{{2(e_a-e_b)^\top H_r^+(e_a-e_b)}},
\]
where $H_r$ is the observed BT Fisher Laplacian. Every remove branch receives
screening score $\sum_r q_{{p'p,r}}$. Each add arm receives
\[
s_q=\max_{{p\in\widehat P}}\min_{{r\in[d]}}q_{{qp,r}}.
\]
The globally smallest screening branch is selected. For an add arm, a feasible
joint witness assignment is built from the coordinate-wise cheapest
assignments, all same-coordinate assignments, and retained assignments from
earlier phases. The original constrained BT likelihood is then optimized for
that one joint certificate. Denote its numerical profile lower value by
$G_t^{{\rm cert}}$ and the global screening value by $S_t$. The implemented
stopping statistic is
\[
Z_t=\min\{{S_t,G_t^{{\rm cert}}\}}.
\]
This is substantially closer to a Pareto profile likelihood than the earlier
single-pair local-quadratic prototype: the active branch is a jointly feasible
frontier-changing certificate and is evaluated with the original BT
likelihood. However, $S_t$ is a scalable screening heuristic rather than a
proved lower bound on the global Pareto profile. Therefore the formal
large-front statistic is explicitly marked \emph{{not certified}}.

\paragraph{{Allocation and threshold.}}
The fitted MLE and active feasible alternative define one-cell information
\[
I_{{ij,r}}=
\operatorname{{kl}}\!\left(
\sigma(\widehat\theta_{{i,r}}-\widehat\theta_{{j,r}}),
\sigma(\widetilde\theta_{{i,r}}-\widetilde\theta_{{j,r}})
\right).
\]
Exponentiated mirror updates track high-information cells; C-tracking converts
the target proportions into integer queries. Uniform mass
$\rho_t=t^{{-1/3}}$ is mixed over all $dK(K-1)/2$ pair-objective cells, and
geometric checks use target size $\lceil1.8t\rceil$. The stopping threshold is
\[
\beta_t(\delta)=
\log\frac1\delta+
\frac\lambda2\sum_r\|\widehat\theta_r\|_2^2+
\frac12\sum_r\log\det\!\left(
I+\frac{{\sigma^2}}\lambda L_{{r,t}}
\right),
\]
with $(\lambda,\sigma^2)=(0.1,0.25)$ and count Laplacian $L_{{r,t}}$.
The run stops when all MLE/profile fits converge and $Z_t\ge\beta_t(\delta)$.

\paragraph{{Status of the method.}}
This report tests a fixed-confidence \emph{{heuristic extension}}: it uses
$\delta$ in a time-uniform-threshold-shaped stopping rule and empirically
audits errors, but it does not claim that the scalar ICML theorem proves
$\delta$-correctness or asymptotic optimality after the Pareto extension.
That limitation is methodological, not hidden in a footnote.

\begin{{table}}[H]
\centering
\caption{{Pareto BT-GLR implementation constants, fixed before the formal run.}}
\label{{tab:track-constants}}
\small
\begin{{tabular}}{{lr@{{\qquad}}lr}}
\toprule
Parameter & Value & Parameter & Value \\
\midrule
Box bound $B$ & 2 & Burn-in per pair-objective cell & 2 \\
Mixture $\lambda$ & 0.1 & Sub-Gaussian proxy $\sigma^2$ & 0.25 \\
Batch growth & 1.8 & Mirror step & 0.8 \\
Forced-exploration exponent & $1/3$ & Maximum queries & $10^{{18}}$ \\
Exact-profile arm cutoff & 8 & Screened certificate pool & 8 \\
MLE tolerance & $10^{{-7}}$ & Maximum MLE iterations & 1,000 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Fixed-confidence scaling}}

This section fixes $\delta=0.05$ and varies one structural parameter at a time:
$K\in\{{16,32,64,128\}}$ at $(d,\Delta)=(4,1)$;
$d\in\{{2,4,10\}}$ at $(K,\Delta)=(64,1)$; and
$\Delta\in\{{0.5,0.75,1,1.25,1.5\}}$ at $(K,d)=(64,10)$.
The horizontal axes are respectively arm count, objective count, and latent
separation; the vertical axis is mean stopping time on a logarithmic scale.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{../results/figures_certificate/formal_scaling.pdf}}
\caption{{Fixed-confidence scaling. Lines show mean stopping time and shaded
ribbons show mean $\pm$ one standard error. Lower is better.}}
\label{{fig:scaling}}
\end{{figure}}

{scaling_table}
{section_interpretation(summary, "fixed_confidence_scaling")}

\section{{Confidence-scaling quantiles}}

The horizontal coordinate is $\log(1/\delta)$ and the ordinate is mean stopping
time. Symmetric $K=64,d=10$ uses
$\delta\in\{{0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001\}}$ with 500 paired
replications. Arena-10 medium uses the first six values through $0.005$ with
300 paired replications. Reusing each latent $\theta$ across confidence levels
isolates the effect of the requested confidence from instance variation.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{../results/figures_certificate/formal_confidence.pdf}}
\caption{{Confidence scaling. Lines are mean stopping time and ribbons are one
standard error. The abscissa is $\log(1/\delta)$, not $\delta$.}}
\label{{fig:confidence}}
\end{{figure}}

{confidence_table}
{confidence_fit_table(summary)}
{section_interpretation(summary, "confidence_scaling_quantiles")}

\section{{Fixed-confidence benchmark suite}}

The eight settings probe distinct frontier geometries rather than one smooth
scaling path: Convex-2D, Convex-3D, Arena-4 small, Arena-4 medium,
Arena-10 medium, Witness-4, Witness-10, and Two-group-10. The vertical axis is
mean stopping time at $\delta=0.05$ on a logarithmic scale. The darker interval
inside each pastel bar is mean $\pm$ one SE.

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{../results/figures_certificate/formal_benchmarks.pdf}}
\caption{{All eight fixed-confidence benchmarks in one panel. Lower bars are
better; darker overlays show mean $\pm$ one standard error. Text above a bar
reports runs that reached the $10^{{18}}$ cap without stopping. Means in those
cells are cap-coded lower bounds on uncapped mean stopping time.}}
\label{{fig:benchmarks}}
\end{{figure}}

{benchmark_table}
{section_interpretation(summary, "fixed_confidence_benchmarks")}

Pareto BT-GLR reaches the cap in 10 of 500 Convex-3D runs and 9 of 500
Two-group-10 runs. All other benchmark runs stop. The corresponding displayed
Pareto/VB-EGE ratios are therefore conservative lower bounds: removing the cap
cannot turn either cell into a Pareto BT-GLR win.

\section{{Pareto-size ablation}}

This section fixes $K=64$ and varies the true frontier size
$|P|\in\{{4,8,16,32\}}$ for $d=4$ and $d=10$. The abscissa is the generator's
verified true Pareto-set size, and the ordinate is mean stopping time at
$\delta=0.05$. This experiment directly tests the combinatorial pressure in
the add-frontier alternative.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\textwidth]{{../results/figures_certificate/formal_pareto_size.pdf}}
\caption{{Pareto-size ablation. Lines show mean stopping time and ribbons show
mean $\pm$ one standard error.}}
\label{{fig:pareto-size}}
\end{{figure}}

{pareto_table}
{section_interpretation(summary, "pareto_size_ablation")}

\section{{Reliability and implementation diagnostics}}

{reliability_table(diagnostics)}

Across cells, the minimum stopping rates are {vb_min_stop:.4f} for VB-EGE and
{track_min_stop:.4f} for Pareto BT-GLR. The minimum Pareto BT-GLR MLE
convergence rate is {track_min_convergence:.4f}. A zero observed error count is
computed only among stopped runs and is not a proof of zero risk;
Table~\ref{{tab:reliability}} reports Wilson upper endpoints to make the
finite-repetition resolution explicit. In particular, the 19 capped Pareto
BT-GLR benchmark runs have no returned set and are excluded from the
conditional set-error denominator, while remaining visible through the stop
rate.
At the individual-cell level, zero failures in 500 repetitions has a two-sided
95\% Wilson upper endpoint of approximately $0.0076$, and zero in 300 has
endpoint approximately $0.0126$. Consequently these repetitions can reveal
gross miscalibration but cannot empirically certify a target such as
$\delta=0.001$.

All formal Pareto BT-GLR rows use the large-front profile mode recorded as
\texttt{{{profile_mode_text}}}. The fraction marked as theorem-level certified
statistics is {certified_rate:.4f}, as expected from the declared heuristic
status. The implementation nevertheless checks numerical convergence, uses a
joint feasible certificate for the active branch, records the original-BT
profile estimate and bounds, and records the threshold at stopping.

\section{{Conclusions}}

The experiment supports a comparison between two fully specified
fixed-confidence stopping procedures under the same formal synthetic bank.
It does not support relabeling the Pareto extension as the original scalar
ICML algorithm. VB-EGE operates in Vector-Borda certificate space; Pareto
BT-GLR fits the full coordinate-wise BT model and adaptively targets a
frontier-changing certificate. Their stopping-time difference therefore
combines estimator dimension, alternative geometry, sampling allocation, and
stopping calibration.

Numerically, Pareto BT-GLR wins {track_wins} of {len(all_rows)} aggregate-mean
cells and has a cross-cell geometric mean ratio of
{global_geometric_ratio:.3g} relative to VB-EGE. The section-level figures and
tables identify where that advantage or disadvantage occurs; the empirical
error audit states the resolution supported by the actual repetition counts.

\section{{Reproducibility}}

\begin{{verbatim}}
python -u -m icml_pareto_track_stop.run_formal \
  --jobs 16 --prepare-jobs 1 --checkpoint-every 25 --resume
python -m icml_pareto_track_stop.summarize_formal
python -m icml_pareto_track_stop.report.build_certificate_v1_formal_report
\end{{verbatim}}

The run-level CSV, resumable JSONL checkpoint, aggregate summaries, paired
summaries, diagnostics, figures, TeX source, and PDF all reside under
\path{{icml_pareto_track_stop/}}. Generated data and build artifacts are kept
out of version control; only the canonical report PDF is published.

\begin{{thebibliography}}{{9}}
\bibitem{{goldberger2026}}
M. Goldberger and N. Rudi.
\newblock Optimal Top-$k$ Identification from Pairwise Comparisons.
\newblock \emph{{Proceedings of the 43rd International Conference on Machine
Learning}}, PMLR 306, 2026.
\newblock \url{{https://arxiv.org/abs/2607.08979}}.

\bibitem{{crepon2024}}
B. Cr\'epon, A. Garivier, and W. M. Koolen.
\newblock Sequential Learning of the Pareto Front for Multi-objective Bandits.
\newblock \emph{{Proceedings of AISTATS}}, PMLR 238, 2024.
\newblock \url{{https://proceedings.mlr.press/v238/crepon24a.html}}.
\end{{thebibliography}}

\end{{document}}
"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(
        {
            "total_runs": total_runs,
            "track_mean_wins": track_wins,
            "geometric_mean_ratio": global_geometric_ratio,
            "vb_errors": vb_errors,
            "track_errors": track_errors,
            "paired_rows": len(paired),
        }
    )


if __name__ == "__main__":
    build()
