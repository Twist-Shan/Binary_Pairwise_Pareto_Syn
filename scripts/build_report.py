from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIG_DIR = REPORT / "figures"
SUMMARY_DIR = ROOT / "results" / "summary"
SOURCE_FIG_DIR = ROOT / "results" / "figures"
TODAY = date.today()
REPORT_DATE = f"{TODAY:%B} {TODAY.day}, {TODAY.year}"

ALGORITHMS = [
    "VB-EGE-practical",
    "UniformFocalBorda-FC",
    "UniformPairwiseBT-MLE-Cert",
    "UniformPairwiseBT-BordaPlugIn-FC",
]

BENCHMARK_SETTINGS = [
    {
        "id": "fc_convex2d",
        "label": "Convex-2D",
        "generator": "convex_frontier_2d",
        "K": "60",
        "d": "2",
        "pareto": "15",
        "params": "s=15",
        "reps": "500",
        "description": "Two-dimensional convex trade-off frontier with dominated arms below randomly selected frontier witnesses.",
    },
    {
        "id": "fc_convex3d",
        "label": "Convex-3D",
        "generator": "convex_frontier_3d",
        "K": "60",
        "d": "3",
        "pareto": "15",
        "params": "s=15, simplex alpha=1, margins [0.03,0.18]",
        "reps": "500",
        "description": "Three-dimensional simplex frontier with dominated arms below randomly selected frontier witnesses.",
    },
    {
        "id": "fc_arena4_small",
        "label": "Arena-4 small",
        "generator": "arena_tradeoff_frontier",
        "K": "32",
        "d": "4",
        "pareto": "8",
        "params": "s=8, margins [0.08,0.25], alpha=0.7",
        "reps": "500",
        "description": "Four-dimensional mutually non-dominated anchors, with dominated arms offset from one anchor.",
    },
    {
        "id": "fc_arena4_medium",
        "label": "Arena-4 medium",
        "generator": "arena_tradeoff_frontier",
        "K": "64",
        "d": "4",
        "pareto": "12",
        "params": "s=12, margins [0.08,0.25], alpha=0.7",
        "reps": "500",
        "description": "Larger four-dimensional arena instance with more dominated alternatives.",
    },
    {
        "id": "fc_arena10_medium",
        "label": "Arena-10 medium",
        "generator": "arena_tradeoff_frontier",
        "K": "64",
        "d": "10",
        "pareto": "16",
        "params": "s=16, margins [0.06,0.20], alpha=0.7",
        "reps": "500",
        "description": "High-dimensional arena instance with a larger Pareto frontier and smaller margins.",
    },
    {
        "id": "fc_witness4",
        "label": "Witness-4",
        "generator": "unique_witness_d",
        "K": "40",
        "d": "4",
        "pareto": "8",
        "params": "s=8, q=4, margins [0.04,0.16]",
        "reps": "500",
        "description": "Each Pareto arm has four dominated arms for which it is the unique dominating witness.",
    },
    {
        "id": "fc_witness10",
        "label": "Witness-10",
        "generator": "unique_witness_d",
        "K": "80",
        "d": "10",
        "pareto": "16",
        "params": "s=16, q=4, margins [0.035,0.14]",
        "reps": "500",
        "description": "Ten-dimensional unique-witness instance with small coordinate margins.",
    },
    {
        "id": "fc_twogroup10_medium",
        "label": "Two-group-10",
        "generator": "highdim_two_group",
        "K": "64",
        "d": "10",
        "pareto": "data",
        "params": "K_low=48, K_high=16",
        "reps": "500",
        "description": "Low-quality and high-quality groups in ten dimensions; the true Pareto size is instance-dependent.",
    },
]

SCALING_SETTINGS = [
    {
        "id": "fc_K_scaling_d4",
        "label": "K scaling",
        "generator": "symmetric_hard",
        "grid": "K in {16,32,64,128}; d=4; Delta=1; delta=0.05",
        "reps": "500",
        "var": "K",
        "description": "One Pareto arm at the origin and K-1 dominated arms at -Delta in every coordinate.",
    },
    {
        "id": "fc_d_scaling_K64",
        "label": "d scaling",
        "generator": "symmetric_hard",
        "grid": "K=64; d in {2,4,10}; Delta=1; delta=0.05",
        "reps": "500",
        "var": "d",
        "description": "Symmetric hard instance with the number of dimensions varied.",
    },
    {
        "id": "fc_gap_scaling_K64_d10",
        "label": "Gap scaling",
        "generator": "symmetric_hard",
        "grid": "K=64; d=10; Delta in {0.5,0.75,1,1.25,1.5}; delta=0.05",
        "reps": "500",
        "var": "Delta",
        "description": "Symmetric hard instance with the latent separation Delta varied.",
    },
]

CONFIDENCE_QUANTILE_SETTINGS = [
    {
        "id": "conf_quantile_symmetric_K64_d10",
        "label": "Symmetric K64 d10",
        "generator": "symmetric_hard",
        "grid": "K=64; d=10; Delta=1; delta in {0.20,0.10,0.05,0.02,0.01,0.005,0.002,0.001}",
        "reps": "500",
        "purpose": "Stress the tail of fixed-confidence stopping times as log(1/delta) grows.",
    },
    {
        "id": "conf_quantile_arena10_medium",
        "label": "Arena-10 medium",
        "generator": "arena_tradeoff_frontier",
        "grid": "K=64; d=10; s=16; margins [0.06,0.20]; delta in {0.20,0.10,0.05,0.02,0.01,0.005}",
        "reps": "300",
        "purpose": "Check confidence scaling on a high-dimensional multi-Pareto arena.",
    },
]

CALIBRATION_SETTINGS = [
    {
        "id": "calib_symmetric_K64_d10",
        "label": "Symmetric K64 d10",
        "generator": "symmetric_hard",
        "params": "K=64; d=10; Delta=1",
        "reps": "500",
    },
    {
        "id": "calib_arena10_medium",
        "label": "Arena-10 medium",
        "generator": "arena_tradeoff_frontier",
        "params": "K=64; d=10; s=16; margins [0.06,0.20]",
        "reps": "300",
    },
    {
        "id": "calib_witness10",
        "label": "Witness-10",
        "generator": "unique_witness_d",
        "params": "K=80; d=10; s=16; q=4; margins [0.035,0.14]",
        "reps": "300",
    },
]

PARETO_SIZE_SETTINGS = [
    {
        "id": "pareto_size_arena4_K64",
        "label": "Arena-4",
        "generator": "arena_tradeoff_frontier",
        "grid": "K=64; d=4; s in {4,8,16,32}; margins [0.08,0.25]",
        "reps": "300",
    },
    {
        "id": "pareto_size_arena10_K64",
        "label": "Arena-10",
        "generator": "arena_tradeoff_frontier",
        "grid": "K=64; d=10; s in {4,8,16,32}; margins [0.06,0.20]",
        "reps": "300",
    },
]

FIGURES_TO_COPY = {
    "scale_tau_K.pdf": SOURCE_FIG_DIR / "fixed_confidence_scaling" / "tau_vs_K_fc_K_scaling_d4.pdf",
    "scale_tau_d.pdf": SOURCE_FIG_DIR / "fixed_confidence_scaling" / "tau_vs_d_fc_d_scaling_K64.pdf",
    "scale_tau_gap.pdf": SOURCE_FIG_DIR / "fixed_confidence_scaling" / "tau_vs_param_Delta_fc_gap_scaling_K64_d10.pdf",
    "scale_bar_K.pdf": SOURCE_FIG_DIR / "fixed_confidence_scaling" / "tau_bar_vs_K_fc_K_scaling_d4.pdf",
    "scale_bar_d.pdf": SOURCE_FIG_DIR / "fixed_confidence_scaling" / "tau_bar_vs_d_fc_d_scaling_K64.pdf",
    "scale_bar_gap.pdf": SOURCE_FIG_DIR / "fixed_confidence_scaling" / "tau_bar_vs_param_Delta_fc_gap_scaling_K64_d10.pdf",
    "scale_pair_coverage.pdf": SOURCE_FIG_DIR / "fixed_confidence_scaling" / "pair_cell_coverage_effect.pdf",
    "bench_pair_coverage.pdf": SOURCE_FIG_DIR / "fixed_confidence_benchmarks" / "pair_cell_coverage_effect.pdf",
    "conf_mean_ci_symmetric.pdf": SOURCE_FIG_DIR / "confidence_scaling_quantile" / "tau_mean_ci_vs_log_inv_delta_symmetric.pdf",
    "conf_mean_ci_arena10.pdf": SOURCE_FIG_DIR / "confidence_scaling_quantile" / "tau_mean_ci_vs_log_inv_delta_arena10.pdf",
    "conf_bar_ci_symmetric.pdf": SOURCE_FIG_DIR / "confidence_scaling_quantile" / "tau_mean_ci_bar_vs_log_inv_delta_symmetric.pdf",
    "conf_bar_ci_arena10.pdf": SOURCE_FIG_DIR / "confidence_scaling_quantile" / "tau_mean_ci_bar_vs_log_inv_delta_arena10.pdf",
    "calib_error_frontier.pdf": SOURCE_FIG_DIR / "constants_calibration" / "error_tau_frontier_all.pdf",
    "calib_wilson_frontier.pdf": SOURCE_FIG_DIR / "constants_calibration" / "wilson_tau_frontier_all.pdf",
    "calib_heatmap_tau_symmetric.pdf": SOURCE_FIG_DIR / "constants_calibration" / "heatmap_mean_tau_by_constants_symmetric.pdf",
    "calib_heatmap_tau_arena10.pdf": SOURCE_FIG_DIR / "constants_calibration" / "heatmap_mean_tau_by_constants_arena10.pdf",
    "calib_heatmap_tau_witness10.pdf": SOURCE_FIG_DIR / "constants_calibration" / "heatmap_mean_tau_by_constants_witness10.pdf",
    "pareto_tau_arena4.pdf": SOURCE_FIG_DIR / "pareto_size_ablation" / "tau_vs_pareto_size_arena4.pdf",
    "pareto_tau_arena10.pdf": SOURCE_FIG_DIR / "pareto_size_ablation" / "tau_vs_pareto_size_arena10.pdf",
    "bench_group_all.pdf": SOURCE_FIG_DIR / "fixed_confidence_benchmarks" / "benchmark_group_all.pdf",
    "theory_constants_symmetric.pdf": SOURCE_FIG_DIR / "theory_constants_sanity" / "tau_by_algorithm_theory_constants_symmetric_K16_d4.pdf",
    "theory_constants_arena4.pdf": SOURCE_FIG_DIR / "theory_constants_sanity" / "tau_by_algorithm_theory_constants_arena4_K16.pdf",
    "allocation_vs_gap.pdf": SOURCE_FIG_DIR / "mechanism" / "allocation_vs_gap.pdf",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    return float(value)


def tex_escape(text: object) -> str:
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def sci_tex(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    if abs(value) < 1e4:
        return f"{value:.2f}"
    return rf"${mantissa:.2f}\times 10^{{{exponent}}}$"


def plain_num(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "--"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def ratio_tex(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value < 10:
        return f"{value:.2f}x"
    if value < 100:
        return f"{value:.1f}x"
    return f"{value:.0f}x"


def mean_se_tex(row: dict[str, str]) -> str:
    mean = sci_tex(as_float(row.get("mean_tau")))
    se = sci_tex(as_float(row.get("se_tau")))
    return f"{mean} ({se})"


def ratio_mean_se_tex(row: dict[str, str] | None) -> str:
    if row is None:
        return "--"
    mean = as_float(row.get("mean_ratio"))
    se = as_float(row.get("se_ratio"))
    if not all(math.isfinite(value) and value >= 0 for value in (mean, se)):
        return "--"
    return f"{ratio_tex(mean)} ({ratio_tex(se)})"


def group_by_experiment(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["experiment_id"]][row["algorithm"]] = row
    return grouped


def rows_for_experiment(rows: list[dict[str, str]], exp_id: str) -> list[dict[str, str]]:
    return [row for row in rows if row["experiment_id"] == exp_id]


def bench_settings_table(bench_rows_by_exp: dict[str, dict[str, dict[str, str]]]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Fixed-confidence benchmark settings. All algorithms use target failure probability $\delta=0.05$; $|P|$ denotes the true Pareto set size.}",
        r"\label{tab:benchmark-settings}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{@{}lYrrrrY@{}}",
        r"\toprule",
        r"Setting & Generator & $K$ & $d$ & $|P|$ & Reps & Parameters \\",
        r"\midrule",
    ]
    for setting in BENCHMARK_SETTINGS:
        pareto = setting["pareto"]
        if pareto == "data":
            vb_row = bench_rows_by_exp[setting["id"]]["VB-EGE-practical"]
            pareto = plain_num(as_float(vb_row["mean_pareto_size_hat"]))
        lines.append(
            " & ".join(
                [
                    tex_escape(setting["label"]),
                    tex_escape(setting["generator"]),
                    setting["K"],
                    setting["d"],
                    str(pareto),
                    setting["reps"],
                    tex_escape(setting["params"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def scaling_settings_table() -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Fixed-confidence scaling settings. All tasks use the symmetric hard generator and four algorithms; each cell is averaged over 500 random permutations/seeds.}",
        r"\label{tab:scaling-settings}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{@{}llYY@{}}",
        r"\toprule",
        r"Sweep & Generator & Grid & Purpose \\",
        r"\midrule",
    ]
    for setting in SCALING_SETTINGS:
        lines.append(
            " & ".join(
                [
                    tex_escape(setting["label"]),
                    tex_escape(setting["generator"]),
                    tex_escape(setting["grid"]),
                    tex_escape(setting["description"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def benchmark_result_table(
    bench_rows_by_exp: dict[str, dict[str, dict[str, str]]],
    paired_rows_by_exp: dict[str, dict[str, dict[str, str]]],
) -> str:
    groups = [
        ("convex-witness", "Convex and witness settings", BENCHMARK_SETTINGS[:2] + BENCHMARK_SETTINGS[5:7]),
        ("arena", "Arena and two-group settings", BENCHMARK_SETTINGS[2:5] + BENCHMARK_SETTINGS[7:]),
    ]
    tables: list[str] = []
    for suffix, title, settings in groups:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{Benchmark mean stopping times and paired mean slowdown ratios: {title}. Parentheses contain one standard error, clustered by latent instance when complete instance IDs are available. All empirical error rates are zero.}}",
            rf"\label{{tab:benchmark-results-{suffix}}}",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Setting & VB mean $\tau$ (SE) & Focal/VB (SE) & MLE/VB (SE) & Plug-in/VB (SE) \\",
            r"\midrule",
        ]
        for setting in settings:
            rows = bench_rows_by_exp[setting["id"]]
            paired = paired_rows_by_exp.get(setting["id"], {})
            lines.append(
                " & ".join(
                    [
                        tex_escape(setting["label"]),
                        mean_se_tex(rows["VB-EGE-practical"]),
                        ratio_mean_se_tex(paired.get("UniformFocalBorda-FC")),
                        ratio_mean_se_tex(paired.get("UniformPairwiseBT-MLE-Cert")),
                        ratio_mean_se_tex(paired.get("UniformPairwiseBT-BordaPlugIn-FC")),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        tables.append("\n".join(lines))
    return "\n".join(tables)


def scaling_result_table(scaling_rows: list[dict[str, str]]) -> str:
    groups = [("size", SCALING_SETTINGS[:2]), ("gap", SCALING_SETTINGS[2:])]
    tables: list[str] = []
    for suffix, settings in groups:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Scaling means and baseline ratios. Parentheses contain one standard error for VB-EGE; all empirical error rates are zero.}",
            rf"\label{{tab:scaling-results-{suffix}}}",
            r"\small",
            r"\setlength{\tabcolsep}{7pt}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Value & VB mean $\tau$ (SE) & Focal/VB & MLE/VB & Plug-in/VB \\",
            r"\midrule",
        ]
        for setting in settings:
            var = setting["var"]
            exp_rows = rows_for_experiment(scaling_rows, setting["id"])
            keyed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
            for row in exp_rows:
                key = row[var] if var in row else row.get(f"param_{var}", "")
                if var == "Delta":
                    key = row["param_Delta"]
                keyed[key][row["algorithm"]] = row
            lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{tex_escape(setting['label'])}}}}} \\")
            for key in sorted(keyed, key=as_float):
                rows = keyed[key]
                vb = rows["VB-EGE-practical"]
                vb_tau = as_float(vb["mean_tau"])
                lines.append(
                    " & ".join(
                        [
                            plain_num(as_float(key)),
                            mean_se_tex(vb),
                            ratio_for(rows, "UniformFocalBorda-FC", vb_tau),
                            ratio_for(rows, "UniformPairwiseBT-MLE-Cert", vb_tau),
                            ratio_for(rows, "UniformPairwiseBT-BordaPlugIn-FC", vb_tau),
                        ]
                    )
                    + r" \\"
                )
            if setting is not settings[-1]:
                lines.append(r"\addlinespace")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        tables.append("\n".join(lines))
    return "\n".join(tables)


def rows_by_setting_and_value(
    rows: list[dict[str, str]],
    value_col: str,
) -> dict[str, dict[float, dict[str, dict[str, str]]]]:
    grouped: dict[str, dict[float, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        value = as_float(row.get(value_col))
        if math.isfinite(value):
            grouped[row["experiment_id"]][value][row["algorithm"]] = row
    return grouped


def ratio_for(rows: dict[str, dict[str, str]], algorithm: str, vb_tau: float) -> str:
    row = rows.get(algorithm)
    if row is None or not math.isfinite(vb_tau) or vb_tau <= 0:
        return "--"
    return ratio_tex(as_float(row.get("mean_tau")) / vb_tau)


def setting_label(settings: list[dict[str, str]], exp_id: str) -> str:
    for setting in settings:
        if setting["id"] == exp_id:
            return setting["label"]
    return exp_id


def confidence_quantile_settings_table() -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Confidence-scaling quantile settings. These runs use the same fixed-confidence algorithms as the main benchmark suite and vary only the requested failure probability.}",
        r"\label{tab:confidence-quantile-settings}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{@{}llYYr@{}}",
        r"\toprule",
        r"Setting & Generator & Grid & Purpose & Reps \\",
        r"\midrule",
    ]
    for setting in CONFIDENCE_QUANTILE_SETTINGS:
        breakable_grid = tex_escape(setting["grid"]).replace(",", r",\allowbreak{}")
        lines.append(
            " & ".join(
                [
                    tex_escape(setting["label"]),
                    tex_escape(setting["generator"]),
                    breakable_grid,
                    tex_escape(setting["purpose"]),
                    setting["reps"],
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def confidence_quantile_result_table(rows: list[dict[str, str]]) -> str:
    grouped = rows_by_setting_and_value(rows, "delta")
    tables: list[str] = []
    for setting in CONFIDENCE_QUANTILE_SETTINGS:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{Confidence scaling on {tex_escape(setting['label'])}. VB-EGE is reported by mean stopping time with one standard error in parentheses; q95 is retained as a tail diagnostic.}}",
            rf"\label{{tab:confidence-quantile-results-{setting['id'].replace('_', '-')}}}",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"$\delta$ & VB mean $\tau$ (SE) & VB q95 $\tau$ & Focal/VB & MLE/VB & Plug-in/VB \\",
            r"\midrule",
        ]
        exp_group = grouped.get(setting["id"], {})
        for delta in sorted(exp_group, reverse=True):
            alg_rows = exp_group[delta]
            vb = alg_rows.get("VB-EGE-practical")
            if vb is None:
                continue
            vb_tau = as_float(vb.get("mean_tau"))
            lines.append(
                " & ".join(
                    [
                        plain_num(delta, 3),
                        mean_se_tex(vb),
                        sci_tex(as_float(vb.get("q95_tau"))),
                        ratio_for(alg_rows, "UniformFocalBorda-FC", vb_tau),
                        ratio_for(alg_rows, "UniformPairwiseBT-MLE-Cert", vb_tau),
                        ratio_for(alg_rows, "UniformPairwiseBT-BordaPlugIn-FC", vb_tau),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        tables.append("\n".join(lines))
    return "\n".join(tables)


def calibration_settings_table() -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Constant-sensitivity settings. VB-EGE is swept over sample constants $\{0.5,1,2,4\}$ and threshold constants $\{2,3,4,6\}$ at $\delta=0.05$; the focal and BT-MLE-Cert references use the default constants $(2,4)$.}",
        r"\label{tab:calibration-settings}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{@{}llYYr@{}}",
        r"\toprule",
        r"Setting & Generator & Parameters & Grid & Reps \\",
        r"\midrule",
    ]
    for setting in CALIBRATION_SETTINGS:
        lines.append(
            " & ".join(
                [
                    tex_escape(setting["label"]),
                    tex_escape(setting["generator"]),
                    tex_escape(setting["params"]),
                    r"$c_s\in\{0.5,1,2,4\}$; $c_\theta\in\{2,3,4,6\}$",
                    setting["reps"],
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def _constant_pair(row: dict[str, str]) -> str:
    return f"({plain_num(as_float(row.get('sample_const')))}, {plain_num(as_float(row.get('threshold_const')))})"


def _default_vb_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if row["algorithm"] != "VB-EGE-practical":
            continue
        if abs(as_float(row.get("sample_const")) - 2.0) < 1e-9 and abs(as_float(row.get("threshold_const")) - 4.0) < 1e-9:
            return row
    return None


def _best_vb_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [row for row in rows if row["algorithm"] == "VB-EGE-practical"]
    valid = [
        row
        for row in candidates
        if as_float(row.get("error_rate"), 1.0) <= 0.05 and as_float(row.get("mean_stopped"), 1.0) >= 0.999
    ]
    pool = valid or candidates
    if not pool:
        return None
    return min(pool, key=lambda row: as_float(row.get("mean_tau"), float("inf")))


def calibration_result_table(rows: list[dict[str, str]]) -> str:
    by_exp: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_exp[row["experiment_id"]].append(row)
    summary_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Constant-sensitivity stopping times. The default row uses $(c_s,c_\theta)=(2,4)$. The fastest row minimizes mean stopping time among grid points with empirical error at most 0.05 and complete stopping. Parentheses contain one standard error.}",
        r"\label{tab:calibration-results-times}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Setting & Default VB mean $\tau$ (SE) & Fastest constants & Fastest mean $\tau$ (SE) \\",
        r"\midrule",
    ]
    ratio_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Constant-sensitivity baseline ratios at the paper constants. Ratios compare mean stopping times; values above one favor VB-EGE. Every tested constant pair has zero observed error, so the sweep is a sensitivity analysis rather than an identified efficiency--reliability frontier.}",
        r"\label{tab:calibration-results-ratios}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Setting & Focal/default VB & MLE/default VB \\",
        r"\midrule",
    ]
    for setting in CALIBRATION_SETTINGS:
        exp_rows = by_exp.get(setting["id"], [])
        default = _default_vb_row(exp_rows)
        best = _best_vb_row(exp_rows)
        focal = next((row for row in exp_rows if row["algorithm"] == "UniformFocalBorda-FC"), None)
        mle = next((row for row in exp_rows if row["algorithm"] == "UniformPairwiseBT-MLE-Cert"), None)
        default_tau = as_float(default.get("mean_tau")) if default else float("nan")
        summary_lines.append(
            " & ".join(
                [
                    tex_escape(setting["label"]),
                    mean_se_tex(default) if default else "--",
                    _constant_pair(best) if best else "--",
                    mean_se_tex(best) if best else "--",
                ]
            )
            + r" \\"
        )
        ratio_lines.append(
            " & ".join(
                [
                    tex_escape(setting["label"]),
                    ratio_tex(as_float(focal.get("mean_tau")) / default_tau)
                    if focal and math.isfinite(default_tau)
                    else "--",
                    ratio_tex(as_float(mle.get("mean_tau")) / default_tau)
                    if mle and math.isfinite(default_tau)
                    else "--",
                ]
            )
            + r" \\"
        )
    summary_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    ratio_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(summary_lines + ratio_lines)


def theory_constants_result_table(rows: list[dict[str, str]]) -> str:
    grouped = group_by_experiment(rows)
    settings = [
        ("theory_constants_symmetric_K16_d4", "Symmetric K16 d4"),
        ("theory_constants_arena4_K16", "Arena-4 K16"),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Paper/conservative constants sensitivity check. Both variants use the same instances and observation replications. The paper constants are $(c_s,c_\theta,c_{\log})=(2,4,4)$ and the more conservative profile is $(8,16,4)$. Historical raw algorithm labels are retained for artifact compatibility.}",
        r"\label{tab:theory-constants}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Setting & Paper mean $\tau$ (SE) & Conservative mean $\tau$ (SE) & Conservative/paper \\",
        r"\midrule",
    ]
    for exp_id, label in settings:
        exp_rows = grouped[exp_id]
        practical = exp_rows["VB-EGE-practical"]
        theory = exp_rows["VB-EGE-theory"]
        practical_tau = as_float(practical.get("mean_tau"))
        theory_tau = as_float(theory.get("mean_tau"))
        lines.append(
            " & ".join(
                [
                    label,
                    mean_se_tex(practical),
                    mean_se_tex(theory),
                    ratio_tex(theory_tau / practical_tau),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def arena10_consistency_table(
    bench_rows: list[dict[str, str]],
    confidence_rows: list[dict[str, str]],
    calibration_rows: list[dict[str, str]],
) -> str:
    candidates = [
        (
            "Main benchmark",
            next(
                row
                for row in bench_rows
                if row["experiment_id"] == "fc_arena10_medium"
                and row["algorithm"] == "VB-EGE-practical"
            ),
        ),
        (
            r"Confidence sweep, $\delta=0.05$",
            next(
                row
                for row in confidence_rows
                if row["experiment_id"] == "conf_quantile_arena10_medium"
                and row["algorithm"] == "VB-EGE-practical"
                and abs(as_float(row.get("delta")) - 0.05) < 1e-12
            ),
        ),
        (
            r"Constant sensitivity, $(2,4)$",
            next(
                row
                for row in calibration_rows
                if row["experiment_id"] == "calib_arena10_medium"
                and row["algorithm"] == "VB-EGE-practical"
                and abs(as_float(row.get("sample_const")) - 2.0) < 1e-12
                and abs(as_float(row.get("threshold_const")) - 4.0) < 1e-12
            ),
        ),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Arena-10 cross-section audit. The benchmark uses 50 instances from bank \texttt{arena10\_medium\_v2}; constant sensitivity uses its first 30 instances, with 10 observation replications and common random numbers. The confidence sweep uses an independent paired-$\delta$ bank.}",
        r"\label{tab:arena-consistency}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Section & Reps & Mean $\tau$ (SE) \\",
        r"\midrule",
    ]
    for label, row in candidates:
        lines.append(
            " & ".join(
                [
                    label,
                    str(int(as_float(row.get("n_reps")))),
                    mean_se_tex(row),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def pareto_size_settings_table() -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Pareto-set-size sensitivity settings. The arena generator is held fixed while the target frontier size $s=|P|$ changes.}",
        r"\label{tab:pareto-size-settings}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{@{}llYYr@{}}",
        r"\toprule",
        r"Setting & Generator & Grid & Purpose & Reps \\",
        r"\midrule",
    ]
    for setting in PARETO_SIZE_SETTINGS:
        lines.append(
            " & ".join(
                [
                    tex_escape(setting["label"]),
                    tex_escape(setting["generator"]),
                    tex_escape(setting["grid"]),
                    "Measure how certificate cost changes as the frontier grows.",
                    setting["reps"],
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def pareto_size_result_table(rows: list[dict[str, str]]) -> str:
    grouped = rows_by_setting_and_value(rows, "param_s")
    tables: list[str] = []
    for setting in PARETO_SIZE_SETTINGS:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{Stopping times by Pareto-set size on {tex_escape(setting['label'])}. Parentheses contain one standard error; ratios compare mean stopping times at the same target $|P|$.}}",
            rf"\label{{tab:pareto-size-results-{setting['id'].replace('_', '-')}}}",
            r"\small",
            r"\setlength{\tabcolsep}{7pt}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"$|P|$ & VB mean $\tau$ (SE) & Focal/VB & MLE/VB & Plug-in/VB \\",
            r"\midrule",
        ]
        exp_group = grouped.get(setting["id"], {})
        for size in sorted(exp_group):
            alg_rows = exp_group[size]
            vb = alg_rows.get("VB-EGE-practical")
            if vb is None:
                continue
            vb_tau = as_float(vb.get("mean_tau"))
            lines.append(
                " & ".join(
                    [
                        plain_num(size),
                        mean_se_tex(vb),
                        ratio_for(alg_rows, "UniformFocalBorda-FC", vb_tau),
                        ratio_for(alg_rows, "UniformPairwiseBT-MLE-Cert", vb_tau),
                        ratio_for(alg_rows, "UniformPairwiseBT-BordaPlugIn-FC", vb_tau),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        tables.append("\n".join(lines))
    return "\n".join(tables)


def algorithm_table() -> str:
    return r"""
\begin{table}[H]
\centering
\caption{Algorithms compared with fixed-confidence stopping interfaces. The MLE row is an empirical certificate heuristic, as detailed below.}
\label{tab:algorithms}
\footnotesize
\begin{tabularx}{\textwidth}{@{}lYY@{}}
\toprule
Name & Sampling model & Fixed-confidence rule \\
\midrule
VB-EGE-practical & Coordinate-wise Vector-Borda estimates & Adaptive accept/reject using empirical gap certificates \\
UniformFocalBorda-FC & Uniform coordinate-wise Borda sampling & Same confidence threshold family without adaptive elimination \\
UniformPairwiseBT-MLE-Cert & Uniform pair-coordinate Bradley--Terry sampling with MLE & Empirical plug-in certificate; no formal MLE confidence region \\
UniformPairwiseBT-BordaPlugIn-FC & Uniform pair-coordinate sampling with Borda plug-in & Stop when Borda plug-in certificate is satisfied \\
\bottomrule
\end{tabularx}
\end{table}
""".strip()


def protocol_definitions() -> str:
    return r"""
\paragraph{Definitions used by the report.}
There are $K$ arms and $d$ objectives. A coordinate-specific comparison between
arms $i$ and $j$ on objective $r$ follows a Bradley--Terry model,
\[
Y_{ijr}\sim\operatorname{Bernoulli}(p_{ijr}),\qquad
p_{ijr}=\sigma(\theta_{ir}-\theta_{jr}).
\]
The coordinate-wise Borda embedding is
\[
B_{ir}=\frac{1}{K-1}\sum_{j\ne i}p_{ijr}.
\]
Most certificates in this report are computed in this Vector-Borda space. The
synthetic generators used for the main experiments pass a Borda/latent Pareto
consistency check, so the reported true Pareto set agrees under $\theta$ and
under $B$.

For any score matrix $X\in\mathbb{R}^{K\times d}$, define
\[
m_{ij}(X)=\min_r(X_{jr}-X_{ir}),\qquad
M_{ij}(X)=\max_r(X_{ir}-X_{jr}),\qquad
\Delta_i^\star(X)=\max_{j\ne i}m_{ij}(X).
\]
If arm $i$ is non-Pareto, its identification gap is
$\Delta_i(X)=\Delta_i^\star(X)$. If arm $i$ is Pareto, the reported gap is
\[
\Delta_i(X)=\min_{j\ne i}
\min\{M_{ij}(X),\max(M_{ji}(X),0)+\max(\Delta_j^\star(X),0)\}.
\]
The minimum Borda gap is $\Delta_{\min,B}=\min_i\Delta_i(B)$. The Borda
hardness is
\[
H_B=\sum_i\Delta_i(B)^{-2},
\]
with $H_B=\infty$ when a gap is non-positive or non-finite. Large $H_B$ means
  that one or more arms lie close to the Pareto decision boundary.

All fixed-confidence methods use the paper's common constants:
$c_{\mathrm{samp}}=2$, $c_{\mathrm{thr}}=4$, and $c_{\log}=4$. In phase $m$,
with cell count $C$, the schedule is
\[
\varepsilon_m=2^{-m},\qquad
L_m=\log(c_{\log} C m^2/\delta),\qquad
n_m=\left\lceil c_{\mathrm{samp}}L_m/\varepsilon_m^2\right\rceil,\qquad
r_m=\sqrt{L_m/(2n_m)}.
\]
The sample constant $c_{\mathrm{samp}}$ multiplies the per-cell sample count,
the log constant $c_{\log}$ is the union-bound safety factor inside $L_m$, and
the threshold constant $c_{\mathrm{thr}}$ is the certificate margin. A
fixed-confidence baseline stops when its estimated minimum gap satisfies
$\widehat{\Delta}_{\min}>c_{\mathrm{thr}}r_m$.

\paragraph{Paper versus conservative constants.}
Algorithm~1 and Theorem~4.1 use
$(c_{\mathrm{samp}},c_{\mathrm{thr}},c_{\log})=(2,4,4)$. The historical
``theory constants'' artifact additionally evaluates $(8,16,4)$ as a more
conservative sensitivity profile; it is not the theorem configuration in the
current paper. The main choice is not retuned by benchmark or algorithm.
Except for sections explicitly varying constants, every scaling, confidence,
benchmark, and Pareto-set-size experiment uses $(2,4,4)$.

\paragraph{Benchmark implementations.}
The three benchmarks other than VB-EGE are implemented as follows.
\begin{itemize}
\item \textbf{UniformFocalBorda-FC.} In each phase it samples every arm-objective
cell $(i,r)$ up to $n_m$ Borda observations. Equivalently, the simulation draws
$S_{ir}\sim\operatorname{Binomial}(N_{ir},B_{ir})$ and estimates
$\widehat B_{ir}=S_{ir}/N_{ir}$. It computes the plug-in strict Pareto set from
$\widehat B$, computes $\widehat{\Delta}_{\min}$ using the gap formula above,
and stops when $\widehat{\Delta}_{\min}>c_{\mathrm{thr}}r_m$. It does not
eliminate arms adaptively; it keeps all $Kd$ cells balanced.
\item \textbf{UniformPairwiseBT-MLE-Cert.} In each phase it samples every
pair-coordinate cell $(i,j,r)$, $i<j$, up to $n_m$ comparisons, so
$C=dK(K-1)/2$. For each objective $r$, it fits a separate ridge-stabilized BT
MLE from the win counts:
\[
\min_{\vartheta_{\cdot r}}
\sum_{i<j}\{W_{ijr}\log(1+\exp[-(\vartheta_{ir}-\vartheta_{jr})])
+W_{jir}\log(1+\exp[\vartheta_{ir}-\vartheta_{jr}])\}
+\frac{\lambda}{2}\|\vartheta_{\cdot r}\|_2^2.
\]
The fitted coordinate scores are centered for identifiability. The algorithm
then computes the plug-in Pareto set and $\widehat{\Delta}_{\min}$ from
$\widehat\theta$ and stops when the same gap certificate is satisfied. If the
graph is disconnected, the optimizer fails, or the estimate is too large, the
implementation reruns the fit with the fallback ridge value.
This method is labeled ``Cert'' rather than ``FC'' because the Bernoulli cell
radius $r_m$ is not, by itself, a valid uniform confidence radius for
$\|\widehat\theta-\theta\|_\infty$. A formal BT-MLE guarantee would also need
likelihood curvature, the comparison-graph Laplacian, dynamic range, ridge,
and gauge fixing. Thus this baseline has a fixed-confidence-style stopping
interface but is an empirical certificate heuristic; zero observed error is
not presented as a proof of $\delta$-correctness.
\item \textbf{UniformPairwiseBT-BordaPlugIn-FC.} It uses the same balanced
pair-coordinate sampling as the MLE baseline, but avoids solving the BT MLE. It
estimates each pairwise probability with add-$1/2$ smoothing,
\[
\widehat p_{ijr}=\frac{W_{ijr}+1/2}{N_{ijr}+1},
\]
forms the plug-in Borda estimate
$\widehat B_{ir}=(K-1)^{-1}\sum_{j\ne i}\widehat p_{ijr}$, and applies the same
strict-Pareto plug-in set and gap certificate in Borda space.
\end{itemize}
""".strip()


def figure_grid(
    title: str,
    label: str,
    entries: list[tuple[str, str]],
    caption_tail: str,
    placement: str = "H",
) -> str:
    lines = [rf"\begin{{figure}}[{placement}]", r"\centering"]
    for idx, (filename, subcaption) in enumerate(entries):
        lines.extend(
            [
                r"\begin{minipage}{0.48\textwidth}",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{figures/{filename}}}",
                rf"\caption*{{({chr(97 + idx)}) {tex_escape(subcaption)}}}",
                r"\end{minipage}",
            ]
        )
        if idx % 2 == 0:
            lines.append(r"\hfill")
        else:
            lines.append(r"\vspace{0.8em}")
    lines.extend(
        [
            rf"\caption{{{tex_escape(title)} {caption_tail}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )
    return "\n".join(lines)


def figure_grid_three(title: str, label: str, entries: list[tuple[str, str]], caption_tail: str) -> str:
    lines = [r"\begin{figure}[H]", r"\centering"]
    for idx, (filename, subcaption) in enumerate(entries):
        width = "0.48" if idx < 2 else "0.52"
        lines.extend(
            [
                rf"\begin{{minipage}}{{{width}\textwidth}}",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{figures/{filename}}}",
                rf"\caption*{{({chr(97 + idx)}) {tex_escape(subcaption)}}}",
                r"\end{minipage}",
            ]
        )
        if idx == 0:
            lines.append(r"\hfill")
        elif idx == 1:
            lines.append(r"\vspace{0.8em}")
    lines.extend(
        [
            rf"\caption{{{tex_escape(title)} {caption_tail}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )
    return "\n".join(lines)


def figure_single(title: str, label: str, filename: str, caption_tail: str) -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.92\textwidth]{{figures/{filename}}}",
            rf"\caption{{{title} {caption_tail}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )


def copy_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for dest, src in FIGURES_TO_COPY.items():
        if not src.exists():
            missing.append(str(src))
            continue
        shutil.copy2(src, FIG_DIR / dest)
    if missing:
        raise FileNotFoundError("Missing figures:\n" + "\n".join(missing))


def max_error(rows: list[dict[str, str]]) -> float:
    return max(as_float(row.get("error_rate"), 0.0) for row in rows)


def vb_loglog_slope(
    rows: list[dict[str, str]], experiment_id: str, x_column: str
) -> float:
    points = []
    for row in rows:
        if (
            row.get("experiment_id") != experiment_id
            or row.get("algorithm") != "VB-EGE-practical"
        ):
            continue
        x = as_float(row.get(x_column))
        y = as_float(row.get("mean_tau"))
        if x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y):
            points.append((math.log(x), math.log(y)))
    if len(points) < 2:
        return float("nan")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return float("nan")
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def order_report_sections(parts: list[str]) -> list[str]:
    """Arrange the report around the empirical argument rather than run chronology."""
    end_document = r"\end{document}"
    cleaned = [part for part in parts if part not in {r"\clearpage", end_document}]
    markers = [
        (index, part)
        for index, part in enumerate(cleaned)
        if part.startswith(r"\section{") or part.startswith(r"\section*{")
    ]
    if not markers:
        return parts

    prefix = cleaned[: markers[0][0]]
    sections: dict[str, list[str]] = {}
    for marker_index, (start, marker) in enumerate(markers):
        stop = markers[marker_index + 1][0] if marker_index + 1 < len(markers) else len(cleaned)
        sections[marker] = cleaned[start:stop]

    desired = [
        r"\section{Protocol and Empirical Error Evidence}",
        r"\section{Constant Sensitivity}",
        r"\section{Fixed-Confidence Scaling}",
        r"\section{Confidence-Scaling Quantiles}",
        r"\section{Fixed-Confidence Benchmark Suite}",
        r"\section{Sensitivity to Pareto-Set Size}",
        r"\section{Sanity Check: Sampling-Mechanism Diagnostics}",
        r"\section{Takeaways}",
        r"\section*{Reproducibility Notes}",
    ]
    missing = [marker for marker in desired if marker not in sections]
    if missing:
        raise ValueError(f"Missing report sections: {missing}")

    ordered = list(prefix)
    for marker in desired:
        ordered.extend(sections[marker])
    ordered.append(end_document)
    return ordered


def generate_tex() -> str:
    scaling_rows = read_csv(SUMMARY_DIR / "fixed_confidence_scaling_summary.csv")
    bench_rows = read_csv(SUMMARY_DIR / "fixed_confidence_benchmarks_summary.csv")
    bench_paired_rows = read_csv(SUMMARY_DIR / "fixed_confidence_benchmarks_summary_paired.csv")
    conf_rows = read_csv(SUMMARY_DIR / "confidence_scaling_quantile_summary.csv")
    calib_rows = read_csv(SUMMARY_DIR / "constants_calibration_summary.csv")
    pareto_rows = read_csv(SUMMARY_DIR / "pareto_size_ablation_summary.csv")
    theory_rows = read_csv(SUMMARY_DIR / "theory_constants_sanity_summary.csv")
    bench_rows_by_exp = group_by_experiment(bench_rows)
    bench_paired_rows_by_exp = group_by_experiment(bench_paired_rows)
    total_scaling_reps = sum(
        int(as_float(r.get("n_reps")))
        for r in scaling_rows
        if r["algorithm"] == "VB-EGE-practical"
    )
    total_bench_reps = sum(
        int(as_float(r.get("n_reps")))
        for r in bench_rows
        if r["algorithm"] == "VB-EGE-practical"
    )
    vb_k_slope = vb_loglog_slope(scaling_rows, "fc_K_scaling_d4", "K")
    vb_d_slope = vb_loglog_slope(scaling_rows, "fc_d_scaling_K64", "d")
    scaling_max_err = max_error(scaling_rows)
    bench_max_err = max_error(bench_rows)

    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{caption}",
        r"\usepackage{float}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage{longtable}",
        r"\usepackage{xcolor}",
        r"\usepackage{url}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\newcolumntype{Y}{>{\raggedright\arraybackslash}X}",
        r"\setlength{\parskip}{0.45em}",
        r"\setlength{\parindent}{0pt}",
        r"\title{Synthetic Fixed-Confidence Experiments for Binary Pairwise Pareto Set Identification}",
        r"\author{Experiment report generated from \texttt{Exp\_Synthetic}}",
        rf"\date{{{REPORT_DATE}}}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        (
            "This report summarizes the completed synthetic experiments for fixed-confidence "
            "binary pairwise Pareto set identification. The experiments compare VB-EGE-practical "
            "against three fixed-confidence benchmarks: focal Borda, BT-MLE-Cert, and "
            "pairwise Borda plug-in. The MLE "
            "method uses a fixed-confidence-style empirical certificate and is not claimed to "
            "have a formal delta-correct confidence region. "
            f"The main runs contain {total_scaling_reps} scaling replications and {total_bench_reps} "
            "benchmark replications per algorithm across all experiment cells. Across all completed main "
            f"settings, the empirical error rate is {scaling_max_err:.1f} for scaling and "
            f"{bench_max_err:.1f} for benchmarks; therefore the main comparison is sample "
            "complexity. Mean stopping time with an instance-clustered standard error for "
            "hierarchical cells and an ordinary replication standard error otherwise is the primary "
            "summary throughout; paired mean slowdown ratios and tail quantiles provide complementary "
            "diagnostics. The experiments are organized in "
            "logical order: protocol and empirical error evidence, constant sensitivity, "
            "fixed-confidence scaling, confidence-scaling quantiles, the fixed-confidence "
            "benchmark suite, sensitivity to Pareto-set size, and sampling-mechanism sanity checks."
        ),
        r"\end{abstract}",
        r"\section{Protocol and Empirical Error Evidence}",
        (
            "The task is fixed-confidence identification of the strict Pareto set under binary "
            "pairwise comparison feedback. Each run stops adaptively when the algorithm can return "
            "a Pareto set certificate at the target failure probability. The default target is "
            r"$\delta=0.05$, except in the confidence-scaling sweep where $\delta$ itself is varied. "
            "The reported stopping time counts coordinate-specific pairwise comparison pulls. "
            "Lower stopping time is better when the empirical error rate is controlled."
        ),
        (
            "Empirical error is the fraction of runs whose returned set differs from the true "
            r"Pareto set. All completed main fixed-confidence cells have zero observed errors. "
            r"A zero count is summarized by the 95\% Wilson upper bound rather than interpreted "
            r"as proof of zero risk. For example, 500 zero-error replications give a Wilson upper "
            r"bound of about 0.0076, so the $\delta=0.001$ sweep supports the stopping-time trend "
            r"in $\log(1/\delta)$ but cannot directly validate a one-in-a-thousand error rate."
        ),
        algorithm_table(),
        protocol_definitions(),
        (
            "In the benchmark runs, the BT-MLE baseline uses balanced random pair-coordinate allocation, ridge parameter "
            r"$10^{-8}$, fallback ridge parameter $10^{-4}$, maximum absolute latent score 20, "
            "and MLE tolerance $10^{-9}$ in the benchmark run. Raw outputs are stored as CSV files "
            "because the local parquet engine was unavailable; this does not change the numerical "
            "results."
        ),
        r"\paragraph{Metrics and legends.}",
        (
            r"Mean stopping time $\bar{\tau}$ with its standard error is the primary stopping-time "
            r"summary in every section. The heterogeneous benchmark suite additionally reports paired "
            r"per-replication mean slowdown ratios, clustered by latent instance when complete IDs "
            r"identify repeated observations. Across the main "
            "fixed-confidence runs the empirical Pareto-set error is zero; this is reported with "
            "Wilson upper bounds rather than uninformative zero-error floor curves. The formal baselines in the report are the fixed-confidence "
            "baselines, not capped fixed-budget variants. Plot legends use the short labels "
            r"\textsc{VB-EGE}, Focal Borda, BT-MLE, and Borda plug-in. In line plots these are "
            "encoded respectively by solid circles, dashed open squares, dash-dot triangles, and "
            "solid diamonds, so nearly coincident curves remain distinguishable without relying only on color. "
            r"Every uncertainty interval "
            "is drawn as a translucent shaded region: a ribbon around a line or a darker interval "
            "block over a bar. No whisker-style error bars are used."
        ),
        r"\paragraph{Plot computation conventions.}",
        (
            r"Whenever the same axis label appears later, it uses the same computation. "
            r"Mean stopping time is $\bar{\tau}=n^{-1}\sum_s\tau_s$ over replications. "
            r"For every plot whose center is a mean, the shaded uncertainty band is one standard "
            r"error, $[\bar\tau-\widehat{\mathrm{SE}}(\bar\tau),"
            r"\bar\tau+\widehat{\mathrm{SE}}(\bar\tau)]$. When an instance has "
            r"multiple observation replications, $\widehat{\mathrm{SE}}$ is computed from "
            r"the latent-instance means and the effective sample size is the number of "
            r"instances; otherwise it is the ordinary replication standard error. "
            r"Every stopping-time curve and bar uses this mean $\pm$ SE interval as its shaded region. "
            r"The q95 stopping time retained in the confidence section is the empirical 95th percentile of the same "
            r"$\{\tau_s\}$ values. When an interval is narrower than the plotted line width, its "
            r"ribbon visually coincides with the center line and is not artificially enlarged. In particular, "
            r"several dyadic scaling baselines have exactly zero within-cell stopping-time variance, so their "
            r"SE ribbon has zero width; the nonzero VB-EGE ribbons can also be visually narrow on a log axis. "
            r"Empirical error is "
            r"$n^{-1}\sum_s\mathbf{1}\{\widehat P_s\ne P_s\}$; when no failures are observed, "
            r"error-floor plots show $\log_{10}(0.5/n)$ instead of $-\infty$. The Wilson upper "
            r"bound is the upper endpoint of the two-sided 95\% Wilson interval for this binomial "
            r"error rate. A main-benchmark paired ratio first computes "
            r"$R_s=\tau_{\mathrm{baseline},s}/\tau_{\mathrm{VB},s}$ on the same instance and "
            r"observation replicate, then reports $\bar R=n^{-1}\sum_s R_s$ with one standard error. "
            r"Where hierarchical instance IDs are available, that standard error is computed from latent-instance "
            r"means; legacy fixed-instance cells use the ordinary replication standard error. Other ratio plots use the explicitly stated "
            r"ratio of summary statistics. "
            r"Pair-cell coverage is averaged over runs after computing "
            r"$\tau_s/C$ for each pairwise run, where $C=dK(K-1)/2$. Mean Hamming distance is "
            r"the average symmetric-difference size $|\widehat P_s\triangle P_s|$. Final phase is "
            r"the last dyadic phase index $m$ used by the algorithm."
        ),
        r"\section{Fixed-Confidence Scaling}",
        r"\paragraph{Definitions.}",
        (
            r"This section studies instance scaling under a fixed-confidence stopping rule: the "
            r"target failure probability is held fixed at $\delta=0.05$ throughout. The three sweeps "
            r"vary the number of arms $K$, objective dimension $d$, or latent gap $\Delta$, holding "
            r"the remaining instance parameters fixed. Dependence on $\delta$ is studied only in "
            r"the next section, Confidence-Scaling Quantiles. "
            r"Every line and bar reports mean stopping time $\bar\tau$. Its translucent ribbon or "
            r"darker bar overlay is $\bar\tau\pm\widehat{\mathrm{SE}}(\bar\tau)$ as defined in Section 1. "
            "Axes are logarithmic when all plotted values are positive; lower values use fewer pulls."
        ),
        scaling_settings_table(),
        scaling_result_table(scaling_rows),
        (
            "The scaling tasks isolate how stopping time changes with the number of arms, the "
            "dimension, and the latent gap on the symmetric hard instance at fixed confidence. "
            "In this generator, there is a single true Pareto arm and all other arms are dominated "
            "by the same coordinate gap. This makes the setting useful for checking whether the "
            "fixed-confidence rules behave monotonically and whether pairwise baselines pay a "
            "large cost for broad pair-coordinate coverage."
        ),
        figure_grid_three(
            "Scaling curves for mean stopping time.",
            "fig:scaling-curves",
            [
                ("scale_tau_K.pdf", "K scaling"),
                ("scale_tau_d.pdf", "d scaling"),
                ("scale_tau_gap.pdf", "Gap scaling"),
            ],
            r"Shaded ribbons are mean $\pm$ one standard error; dark center lines show the means.",
        ),
        figure_grid_three(
            "Scaling results as grouped bar plots.",
            "fig:scaling-bars",
            [
                ("scale_bar_K.pdf", "K scaling"),
                ("scale_bar_d.pdf", "d scaling"),
                ("scale_bar_gap.pdf", "Gap scaling"),
            ],
            r"Darker overlays show mean $\pm$ one standard error.",
        ),
        (
            f"Conclusion. For VB-EGE, descriptive log--log slopes are {vb_k_slope:.2f} "
            f"in K and {vb_d_slope:.2f} in d, consistent with near-linear leading "
            "dependence. VB-EGE and Focal Borda nearly coincide because every dominated "
            "arm has the same gap, leaving little heterogeneity for elimination to exploit. "
            "The pairwise methods become more expensive as K and d grow because they spread "
            "samples over pair-coordinate cells. Larger gaps reduce cost; flat segments are "
            "expected when several gaps stop in the same dyadic phase."
        ),
        r"\section{Confidence-Scaling Quantiles}",
        r"\paragraph{Definitions.}",
        (
            r"The x-axis is $x=\log(1/\delta)$ and the center statistic is mean stopping time "
            r"$\bar\tau$. Shaded ribbons and darker bar overlays are $\bar\tau\pm"
            r"\widehat{\mathrm{SE}}(\bar\tau)$ for "
            r"that mean. The table additionally reports $q_{0.95}(\tau)$, the empirical 95th "
            "percentile, to retain tail information without adding another figure."
        ),
        confidence_quantile_settings_table(),
        confidence_quantile_result_table(conf_rows),
        (
            "This experiment repeats fixed-confidence runs over a wider grid of target failure "
            r"probabilities. Within each experiment and replication, all $\delta$ values reuse "
            "the same synthetic instance, so the confidence effect is not confounded with "
            "between-instance hardness variation. The line legends are the four algorithms; the "
            "curves and bars report mean stopping time with one-standard-error bands, while "
            "the accompanying table retains tail information through q95. The x-axis is "
            r"$\log(1/\delta)$, so approximately linear growth indicates the expected logarithmic "
            "confidence cost. Normalized and tau/log diagnostic plots are omitted here because they "
            "repeat the same ordering without adding a new empirical claim."
        ),
        figure_grid(
            "Confidence-scaling mean stopping times.",
            "fig:confidence-mean-ci",
            [
                ("conf_mean_ci_symmetric.pdf", "Symmetric K64 d10"),
                ("conf_mean_ci_arena10.pdf", "Arena-10 medium"),
            ],
            r"Shaded ribbons are mean $\pm$ one standard error; dark center lines show the means.",
        ),
        figure_grid(
            "Confidence-scaling grouped bar plots.",
            "fig:confidence-bars",
            [
                ("conf_bar_ci_symmetric.pdf", "Symmetric K64 d10"),
                ("conf_bar_ci_arena10.pdf", "Arena-10 medium"),
            ],
            r"Darker overlays show mean $\pm$ one standard error.",
        ),
        (
            "Result interpretation. On the symmetric setting, VB-EGE and the focal Borda baseline "
            "remain close because the certificate is essentially one-dimensional after symmetry. "
            "The arena setting separates the methods more clearly: VB-EGE keeps a smaller high "
            "quantile stopping time than the pairwise baselines because it does not need broad "
            "pair-coordinate coverage at every confidence level. The final-phase diagnostic should "
            "move in steps rather than smoothly, which is expected from phase doubling. A direct "
            r"fit $\bar\tau=a+b\log(1/\delta)$ gives $R^2\approx0.99997$ on the symmetric task "
            r"and $R^2\approx0.9978$ on Arena-10. This supports logarithmic confidence cost in "
            "stopping time; it does not empirically validate a 0.001 error probability from only "
            "500 replications."
        ),
        r"\clearpage",
        r"\section{Fixed-Confidence Benchmark Suite}",
        r"\paragraph{Definitions.}",
        (
            r"Each benchmark bar reports mean stopping time $\bar\tau$ on a log scale. "
            r"The darker interval overlay spans $\bar\tau\pm\widehat{\mathrm{SE}}(\bar\tau)$. Baseline ratios "
            r"in the table are paired per-run mean ratios to VB-EGE, with instance-clustered "
            r"standard errors when complete instance IDs are available. "
            "Lower bars and ratios above one favor VB-EGE. Convex-3D and Arena-10 use "
            "50 latent instances with ten observation replications each; the other settings "
            "use 500 independent latent instances."
        ),
        bench_settings_table(bench_rows_by_exp),
        benchmark_result_table(bench_rows_by_exp, bench_paired_rows_by_exp),
        (
            "The benchmark suite stresses different Pareto geometries beyond the symmetric hard "
            "case. Convex-2D and Convex-3D test clean low-dimensional trade-off frontiers. Arena settings use "
            "mutually non-dominated anchors and dominated alternatives offset from a random anchor. "
            "Witness settings create dominated arms that are certified by a unique Pareto witness, "
            "which tests whether algorithms can avoid unnecessary global pairwise coverage. "
            "Two-group-10 is a high-dimensional separation task with many low-quality arms and a "
            "smaller high-quality group."
        ),
        figure_single(
            "Stopping times across all eight fixed-confidence benchmark settings.",
            "fig:bench-tau-all",
            "bench_group_all.pdf",
            r"Darker overlays show mean $\pm$ one standard error.",
        ),
        (
            "Conclusion. VB-EGE-practical has the smallest mean stopping time in every "
            "benchmark setting. Its advantage is smallest on unique-witness instances and "
            "largest on arena and two-group geometries. BT-MLE estimates pairwise signs "
            "accurately but pays for broad pair-cell coverage; the pairwise Borda plug-in is "
            "the slowest method throughout."
        ),
        r"\clearpage",
        r"\section{Constant Sensitivity}",
        r"\paragraph{Definitions.}",
        (
            r"The sample constant $c_s$ multiplies the phase sample count, the threshold constant "
            r"$c_\theta$ scales the certificate margin, and $c_{\log}$ scales the logarithmic safety "
            r"term. The paper choice is $(c_s,c_\theta,c_{\log})=(2,4,4)$; the explicit "
            r"conservative-constant check uses $(8,16,4)$. Heatmap colors encode mean stopping time and each "
            r"cell prints mean $\pm$ one standard error. In the two bar panels, the darker overlay "
            r"also spans mean $\pm$ one standard error."
        ),
        arena10_consistency_table(bench_rows, conf_rows, calib_rows),
        (
            "The earlier cross-section discrepancy came from independent random instance banks: "
            "rare near-boundary Arena instances created very large stopping-time outliers, so raw "
            "means differed by orders of magnitude. The revised "
            "benchmark now extends the shared bank to 50 instances, while constant sensitivity "
            "retains the first 30; their common subset uses the same ten observation replications "
            "and random-number construction, but the aggregate means need not agree exactly. The "
            "confidence sweep keeps its original independent bank because its within-replication "
            r"pairing across $\delta$ is the design needed for the confidence-scaling claim; its "
            "mean remains informative within that paired bank but is not treated as an exact "
            "cross-section identity with the shared benchmark bank."
        ),
        calibration_settings_table(),
        calibration_result_table(calib_rows),
        (
            r"The sensitivity sweep varies the VB-EGE sample constant $c_s$ and threshold constant "
            r"$c_\theta$. In the sensitivity figures, each point is one constant pair; the x-axis is "
            "mean stopping time and the y-axis is either empirical error or Wilson upper confidence "
            "bound. Heatmaps use sample constant on one axis and threshold constant on the other; "
            "darker or lighter cells encode the plotted metric named in the title. The default "
            r"paper setting is $(c_s,c_\theta)=(2,4)$."
        ),
        figure_grid(
            "Constant-sensitivity error diagnostics.",
            "fig:calibration-frontiers",
            [
                ("calib_error_frontier.pdf", "Empirical error frontier"),
                ("calib_wilson_frontier.pdf", "Wilson upper frontier"),
            ],
            r"All tested cells have zero observed error; panel (b) shows the corresponding Wilson upper bounds.",
            placement="H",
        ),
        figure_grid_three(
            "Constant-sensitivity mean stopping-time heatmaps.",
            "fig:calibration-heatmaps",
            [
                ("calib_heatmap_tau_symmetric.pdf", "Symmetric K64 d10"),
                ("calib_heatmap_tau_arena10.pdf", "Arena-10 medium"),
                ("calib_heatmap_tau_witness10.pdf", "Witness-10"),
            ],
            r"Color encodes mean stopping time; cell text reports mean $\pm$ one standard error.",
        ),
        (
            "Conclusion. The grid identifies stopping-time sensitivity, not a reliability "
            "frontier, because no failures are observed. Smaller constants are faster, while "
            "the pre-specified paper constants remain the common main-run setting."
        ),
        theory_constants_result_table(theory_rows),
        figure_grid(
            "Paper and conservative constants on small instances.",
            "fig:theory-constants",
            [
                ("theory_constants_symmetric.pdf", "Symmetric K16 d4"),
                ("theory_constants_arena4.pdf", "Arena-4 K16"),
            ],
            r"Darker overlays span mean $\pm$ one standard error; the $(8,16,4)$ sensitivity profile is deliberately more conservative.",
            placement="H",
        ),
        r"\clearpage",
        r"\section{Sensitivity to Pareto-Set Size}",
        r"\paragraph{Definitions.}",
        (
            r"The x-axis is the true Pareto-front size $|P|=s$. Each dark line is mean stopping "
            r"time $\bar\tau$, and its translucent ribbon spans $\bar\tau\pm"
            r"\widehat{\mathrm{SE}}(\bar\tau)$ across replications. The result table "
            r"reports ratios of mean stopping times to VB-EGE at the same $(d,s)$ cell; ratios above "
            "one favor VB-EGE."
        ),
        pareto_size_settings_table(),
        pareto_size_result_table(pareto_rows),
        (
            "This sensitivity study holds the arena geometry fixed while increasing the true Pareto frontier "
            r"size $|P|$. Arena-4 and Arena-10 retain the same generator families and margin ranges "
            "while the requested number of frontier arms changes."
        ),
        figure_grid(
            "Stopping time versus Pareto-set size.",
            "fig:pareto-size-sensitivity",
            [
                ("pareto_tau_arena4.pdf", "Arena-4 tau"),
                ("pareto_tau_arena10.pdf", "Arena-10 tau"),
            ],
            r"Shaded ribbons are mean $\pm$ one standard error; dark center lines show the means.",
        ),
        (
            "Conclusion. Stopping cost increases with the frontier size, but every tested "
            "baseline/VB-EGE ratio remains above one. The adaptive advantage therefore "
            "persists while more arms must be accepted."
        ),
        r"\section{Sanity Check: Sampling-Mechanism Diagnostics}",
        r"\paragraph{Definitions.}",
        (
            r"For pairwise methods, $C=dK(K-1)/2$ is the number of pair-coordinate cells; VB-EGE "
            r"uses $Kd$ focal arm-coordinate cells. The structural multiplier is $C/(Kd)=(K-1)/2$. "
            r"The per-cell burden ratio is $(\tau_{\mathrm{pair}}/C)/(\tau_{\mathrm{VB}}/(Kd))$. "
            r"The allocation diagnostic plots each arm's Borda certificate gap $\Delta_i^B$ against "
            r"its total VB-EGE allocation $\sum_r N_{ir}$."
        ),
        figure_grid(
            "Pairwise total-cost decomposition.",
            "fig:pair-coverage",
            [
                ("scale_pair_coverage.pdf", "Scaling tasks"),
                ("bench_pair_coverage.pdf", "Benchmark tasks"),
            ],
            r"Their product is the total stopping-time ratio $\tau_{\mathrm{pair}}/\tau_{\mathrm{VB}}$.",
            placement="H",
        ),
        (
            "This decomposition separates the number of basic cells from the average evidence "
            "collected in each cell. Pairwise methods pay for both a larger cell set and, depending "
            "on the certificate, a different per-cell burden."
        ),
        figure_single(
            "Arm allocation versus Borda certificate gap.",
            "fig:allocation-gap",
            "allocation_vs_gap.pdf",
            r"Smaller-gap arms receive more VB-EGE samples in the representative runs.",
        ),
        (
            "The allocation plot is the direct adaptivity diagnostic: VB-EGE concentrates its "
            "sampling on arms close to the decision boundary instead of maintaining uniform "
            "coverage over all arms or all pair-coordinate cells."
        ),
        r"\section{Takeaways}",
        r"\begin{enumerate}",
        r"\item All completed main fixed-confidence synthetic experiments achieved zero empirical Pareto-set error.",
        r"\item VB-EGE-practical loses essentially nothing to uniform focal Borda on symmetric instances and is substantially more sample-efficient on heterogeneous Pareto geometries.",
        r"\item Pairwise BT-MLE can estimate latent pairwise signs accurately, but fixed-confidence Pareto certification makes its broad pair-coordinate coverage expensive.",
        r"\item The largest gains occur in high-dimensional arena and two-group settings, where adaptive Vector-Borda certificates avoid many unnecessary pairwise comparisons.",
        r"\item Arena-10 benchmark and constant sensitivity share a versioned bank on the calibration subset; the benchmark extends it to 50 instances, while confidence scaling retains a separate paired-$\delta$ bank.",
        r"\item The BT-MLE comparator is an empirical fixed-confidence-style certificate rather than a method with a formal delta-correct confidence region.",
        r"\end{enumerate}",
        r"\section*{Reproducibility Notes}",
        r"\begingroup\small\sloppy",
        (
            r"The source summaries used in this report are "
            r"\path{results/summary/fixed_confidence_scaling_summary.csv} and "
            r"\path{results/summary/fixed_confidence_benchmarks_summary.csv}, plus "
            r"\path{results/summary/fixed_confidence_benchmarks_summary_paired.csv}, "
            r"\path{results/summary/confidence_scaling_quantile_summary.csv}, "
            r"\path{results/summary/constants_calibration_summary.csv}, "
            r"the Pareto-set-size sensitivity summary, and "
            r"\path{results/summary/theory_constants_sanity_summary.csv}. "
            r"The raw completed main outputs are "
            r"\path{results/raw/fixed_confidence_scaling.csv}, "
            r"\path{results/raw/fixed_confidence_benchmarks.csv}, and the corresponding "
            r"CSV fallbacks under \path{results/raw/} for the diagnostic extensions. "
            "Smoke runs are excluded from the main conclusions because they were used only to "
            "validate the implementation path before the full runs. Revised randomized benchmarks "
            r"store \texttt{instance\_id}, \texttt{theta\_hash}, instance index, and observation "
            "replicate in every raw row."
        ),
        r"\endgroup",
        r"\end{document}",
    ]
    return "\n\n".join(order_report_sections(parts))


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    copy_figures()
    tex = generate_tex()
    outpath = REPORT / "synthetic_fixed_confidence_report.tex"
    outpath.write_text(tex, encoding="utf-8")
    print(outpath)


if __name__ == "__main__":
    main()
