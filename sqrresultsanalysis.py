import os
import json
import glob
import argparse
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


METRICS_OF_INTEREST = ["losses", "coverage", "complexity", "time_all", "time_fit"]
METRIC_LABELS = {
    "losses": "Normalised Quantile Loss",
    "coverage": "Absolute Coverage Error",
    "complexity": "Parsimony",
    "time_all": "Total Time",
    "time_fit": "Fit Time",
}

# mapping from internal model keys to user-friendly legend names
Models = {
    "SQR": "SQR",
    "LightGBM": "LGBM",
    "DecisionTree": "QDT",
    "LinearQuantile": "LQR",
}

# canonical ordering for legends
MODEL_ORDER = ["SQR", "LGBM", "QDT", "LQR"]

# fixed publication color palette by display name
COLOR_PALETTE = {
    "SQR": "#377eb8",
    "LGBM": "#ff7f00",
    "QDT": "#008080",
    "LQR": "#e41a1c",
}

# legend styling: small, upper-right, consistent everywhere
LEGEND_KWARGS = dict(
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    frameon=True,
    fontsize=8,
    title_fontsize=8,
    borderaxespad=0.2,
    handlelength=1.2,
    handletextpad=0.4,
    labelspacing=0.2,
)

# figure size modes for publication-quality plots
FIG_SIZES = {
    "single": (6.0, 3.7),  # standard single-column figure
    "half": (4.0, 2.6),    # half-width (e.g. inset) figure
}

# current mode; set by ``main`` using CLI flag
FIG_MODE = "single"


def _set_style():
    sns.set_theme(style="whitegrid")


def _set_fig_mode(mode):
    global FIG_MODE
    if mode not in FIG_SIZES:
        raise ValueError(f"invalid fig_mode '{mode}'")
    FIG_MODE = mode


def _get_figsize():
    return FIG_SIZES.get(FIG_MODE, FIG_SIZES["single"])


def _get_palette():
    """
    Always return the full palette mapping for seaborn/matplotlib.
    Seaborn will ignore keys not present in the data, but colors stay consistent.
    """
    return dict(COLOR_PALETTE)


def _safe_tau_str(tau):
    return str(tau).replace(".", "_") if tau is not None else "none"


def _place_small_legend(ax, title="Model"):
    handles, labels = ax.get_legend_handles_labels()
    if not handles or not labels:
        return

    existing = ax.get_legend()
    if existing is not None:
        existing.remove()

    label_set = set(labels)
    ordered_labels = [m for m in MODEL_ORDER if m in label_set]
    if not ordered_labels:
        ordered_labels = labels

    handle_map = {lab: h for h, lab in zip(handles, labels)}
    ordered_handles = [handle_map[lab] for lab in ordered_labels]

    ax.legend(ordered_handles, ordered_labels, title=title, **LEGEND_KWARGS)


def _prepare_models(df):
    df = df.copy()
    df["Models"] = df["model"].map(Models).fillna(df["model"])
    present = [m for m in MODEL_ORDER if m in df["Models"].unique()]
    if present:
        df["Models"] = pd.Categorical(df["Models"], categories=present, ordered=True)
    return df, present


def read_summary_stats(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.rename(columns={"dataset": "dataset"})
    return df.set_index("dataset")


def collect_results(results_dir, summary_df):
    rows = []
    files = glob.glob(os.path.join(results_dir, "*.json"))
    for f in files:
        with open(f, "r") as fh:
            data = json.load(fh)

        for model_name, model_dict in data.items():
            tau = model_dict.get("tau", None)
            for metric in METRICS_OF_INTEREST:
                if metric not in model_dict:
                    continue
                metric_map = model_dict[metric]
                for ds_name, values in metric_map.items():
                    if ds_name not in summary_df.index:
                        continue

                    if isinstance(values, list) and len(values) > 0:
                        vals = [v for v in values if v is not None]
                        if len(vals) == 0:
                            continue
                        v = float(np.mean(vals))
                    else:
                        try:
                            v = float(values)
                        except Exception:
                            continue

                    n_instances = (
                        int(summary_df.loc[ds_name, "n_instances"])
                        if "n_instances" in summary_df.columns
                        else None
                    )
                    rows.append(
                        {
                            "dataset": ds_name,
                            "n_instances": n_instances,
                            "model": model_name,
                            "metric": metric,
                            "tau": tau,
                            "value": v,
                            "source_file": os.path.basename(f),
                        }
                    )

    return pd.DataFrame(rows)


def _write_bin_summary(out_path, edges, labels, counts, zero_count=None):
    base = os.path.splitext(out_path)[0]
    table_path = base + "_bins.csv"
    rows = []
    if zero_count is not None:
        rows.append({"Size": "0", "range": "0", "#datasets": zero_count})
    for i, label in enumerate(labels):
        low = edges[i]
        high = edges[i + 1]
        rows.append({"Size": label, "range": f"{low}-{high}", "#datasets": counts.get(label, 0)})
    pd.DataFrame(rows).to_csv(table_path, index=False)


# --- output paths ---------------------------------------------------------

def _make_lineplot_out_path(out_dir, metric, subdirs, tau):
    path = os.path.join(out_dir, metric, *subdirs)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"all_models_tau_{_safe_tau_str(tau)}.png")


def _make_boxplot_out_path(out_dir, metric, boxplot_group, tau):
    """
    Store ALL boxplots in one dedicated folder tree:

      <out_dir>/boxplots/<boxplot_group>/<metric>/boxplot_tau_<tau>.png

    Where boxplot_group is e.g. 'by_instances', 'by_features', 'by_categorical_features'.
    """
    safe_tau = _safe_tau_str(tau)
    path = os.path.join(out_dir, "boxplots", boxplot_group, metric)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"boxplot_tau_{safe_tau}.png")


# --- binning --------------------------------------------------------------

def _quantile_bins(series, n_bins=4):
    vals = series.dropna()
    if vals.empty:
        return None, None
    min_val = vals.min()
    max_val = vals.max()
    if min_val == max_val:
        return None, None

    quantiles = vals.quantile(np.linspace(0, 1, n_bins + 1)).tolist()
    edges = sorted(set(quantiles))
    if len(edges) - 1 < n_bins:
        edges = list(np.linspace(min_val, max_val, n_bins + 1))

    labels = ["S", "M", "L", "XL"][: len(edges) - 1]
    return edges, labels


def _categorical_bins(series):
    vals = series.dropna()
    if vals.empty:
        return None, None
    pos = vals[vals > 0]
    if pos.empty:
        return None, None

    edges, _ = _quantile_bins(pos, n_bins=2)
    if edges is None:
        return None, None
    labels = ["S", "L"][: len(edges) - 1]
    return edges, labels


# --- plotting -------------------------------------------------------------

def _plot_distribution(
    df,
    out_dir,
    count_col,
    xlabel,
    boxplot_group,   # e.g. 'by_instances'
    bin_func,
    include_zero=False,
    logx=False,      # kept for API compat; not used by boxplots
):
    df, model_order = _prepare_models(df)
    palette = _get_palette()

    for metric, metric_df in df.groupby("metric"):
        for tau, tau_df in metric_df.groupby("tau"):
            sub = tau_df.dropna(subset=[count_col])
            if sub.empty:
                continue

            edges, labels = bin_func(sub[count_col])
            zero_count = None

            if edges is None:
                if include_zero:
                    zero_count = int((sub[count_col] == 0).sum())
                    labels = ["0"]
                    edges = []
                else:
                    continue

            sub = sub.copy()
            if include_zero:
                pos_mask = sub[count_col] > 0
                if edges:
                    sub.loc[~pos_mask, "bin"] = "0"
                    sub.loc[pos_mask, "bin"] = pd.cut(
                        sub.loc[pos_mask, count_col],
                        bins=edges,
                        labels=labels,
                        include_lowest=True,
                    )
                else:
                    sub["bin"] = "0"
            else:
                sub["bin"] = pd.cut(sub[count_col], bins=edges, labels=labels, include_lowest=True)

            counts = sub["bin"].value_counts().to_dict()
            counts = {str(k): v for k, v in counts.items()}

            plt.figure(figsize=_get_figsize())
            ax = sns.boxplot(
                x="bin",
                y="value",
                hue="Models",
                hue_order=model_order,
                palette=palette,
                data=sub,
                showcaps=True,
                showfliers=False,
                whiskerprops={"linewidth": 0.5},
            )

            _place_small_legend(ax, title="Model")

            plt.xlabel(xlabel)

            if edges:
                tick_labels = []
                for i, lbl in enumerate(labels):
                    low = edges[i]
                    high = edges[i + 1]
                    tick_labels.append(f"{lbl} ({low:.0f}-{high:.0f})")
            else:
                tick_labels = labels
            plt.xticks(range(len(tick_labels)), tick_labels)

            metric_label = METRIC_LABELS.get(metric, metric)
            plt.ylabel(metric_label)
            title_tau = f" (τ={tau})" if tau is not None else ""
            plt.title(f"{boxplot_group.replace('_',' ').title()} — {metric_label}{title_tau}")

            out_path = _make_boxplot_out_path(out_dir, metric, boxplot_group, tau)
            _write_bin_summary(out_path, edges or [], labels, counts, zero_count=zero_count)

            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()


def _plot_comparison(
    df,
    out_dir,
    count_col,
    xlabel,
    subdir,          # e.g. 'by_instances'
    bin_func,
    include_zero=False,
    logx=True,
):
    df, model_order = _prepare_models(df)
    palette = _get_palette()

    for metric, metric_df in df.groupby("metric"):
        for tau, tau_df in metric_df.groupby("tau"):
            sel = tau_df.dropna(subset=[count_col])
            if sel.empty:
                continue

            edges, labels = bin_func(sel[count_col])
            zero_count = None

            if edges is None:
                if include_zero:
                    sel = sel.copy()
                    sel["bin"] = sel[count_col].apply(lambda x: "0")
                    zero_count = int((sel["bin"] == "0").sum())
                else:
                    continue
            else:
                sel = sel.copy()
                if include_zero:
                    pos_mask = sel[count_col] > 0
                    sel.loc[~pos_mask, "bin"] = "0"
                    sel.loc[pos_mask, "bin"] = pd.cut(
                        sel.loc[pos_mask, count_col],
                        bins=edges,
                        labels=labels,
                        include_lowest=True,
                    )
                else:
                    sel["bin"] = pd.cut(sel[count_col], bins=edges, labels=labels, include_lowest=True)

            counts = sel["bin"].value_counts().to_dict()
            zero_count = counts.pop("0", 0) if "0" in counts else zero_count
            counts = {str(k): v for k, v in counts.items()}

            agg = (
                sel.groupby(["model", "bin"])
                .agg(value=("value", "median"), **{count_col: (count_col, "median")})
                .reset_index()
            )
            agg["bin_center"] = agg[count_col]
            agg["Models"] = agg["model"].map(Models).fillna(agg["model"])

            present = [m for m in MODEL_ORDER if m in agg["Models"].unique()]
            if present:
                agg["Models"] = pd.Categorical(agg["Models"], categories=present, ordered=True)

            plt.figure(figsize=_get_figsize())
            ax = plt.gca()

            for model_disp in present:
                mdf = agg[agg["Models"] == model_disp]
                if mdf.empty:
                    continue
                ax.plot(
                    mdf["bin_center"],
                    mdf["value"],
                    marker="o",
                    label=model_disp,
                    color=palette.get(model_disp),
                )

            ax.set_xlabel(xlabel)
            metric_label = METRIC_LABELS.get(metric, metric)
            ax.set_ylabel(metric_label)
            title_tau = f" (τ={tau})" if tau is not None else ""
            ax.set_title(f"Model Comparison — {metric_label}{title_tau}")

            if logx:
                ax.set_xscale("log")

            _place_small_legend(ax, title="Model")

            plt.tight_layout()
            out_path = _make_lineplot_out_path(out_dir, metric, [subdir, "comparison_binned"], tau)
            _write_bin_summary(out_path, edges or [], labels, counts, zero_count=zero_count)
            plt.savefig(out_path, dpi=150)
            plt.close()


# --- public plot wrappers --------------------------------------------------

def plot_metric_by_instances(df, out_dir, logx=True, kind="line"):
    _plot_distribution(
        df,
        out_dir,
        "n_instances",
        "Number of Instances (range)",
        "by_instances",
        lambda s: _quantile_bins(s, n_bins=4),
        include_zero=False,
        logx=logx,
    )


def plot_all_models_together(df, out_dir, metric, tau=None, logx=True):
    sel = df[df["metric"] == metric]
    if tau is not None:
        sel = sel[sel["tau"] == tau]
    if sel.empty:
        return None
    return _plot_comparison(
        sel,
        out_dir,
        "n_instances",
        "Number of Instances (bin center)",
        "by_instances",
        lambda s: _quantile_bins(s, n_bins=4),
        include_zero=False,
        logx=logx,
    )


def plot_metric_by_features(df, out_dir, logx=True, kind="line"):
    df, model_order = _prepare_models(df)
    palette = _get_palette()

    for metric, metric_df in df.groupby("metric"):
        for tau, tau_df in metric_df.groupby("tau"):
            sub = tau_df.dropna(subset=["n_features"])
            if sub.empty:
                continue

            edges, labels = _quantile_bins(sub["n_features"], n_bins=4)
            if edges is None:
                continue

            sub = sub.copy()
            sub["feat_bin"] = pd.cut(sub["n_features"], bins=edges, labels=labels, include_lowest=True)

            counts = sub["feat_bin"].value_counts().to_dict()
            counts = {str(k): v for k, v in counts.items()}

            plt.figure(figsize=_get_figsize())
            ax = sns.boxplot(
                x="feat_bin",
                y="value",
                hue="Models",
                hue_order=model_order,
                palette=palette,
                data=sub,
                showcaps=True,
                showfliers=False,
                whiskerprops={"linewidth": 0.5},
            )

            _place_small_legend(ax, title="Model")

            plt.xlabel("Number of Features (bin label / range)")
            feat_tick_labels = []
            for i, lbl in enumerate(labels):
                low = edges[i]
                high = edges[i + 1]
                feat_tick_labels.append(f"{lbl} ({low:.0f}-{high:.0f})")
            plt.xticks(range(len(feat_tick_labels)), feat_tick_labels)

            metric_label = METRIC_LABELS.get(metric, metric)
            plt.ylabel(metric_label)
            title_tau = f" (τ={tau})" if tau is not None else ""
            plt.title(f"By Features — {metric_label}{title_tau}")

            out_path = _make_boxplot_out_path(out_dir, metric, "by_features", tau)
            _write_bin_summary(out_path, edges, labels, counts)

            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()


def plot_all_models_together_features(df, out_dir, metric, tau=None, logx=True):
    sel = df[df["metric"] == metric]
    if tau is not None:
        sel = sel[sel["tau"] == tau]
    sel = sel.dropna(subset=["n_features"])
    if sel.empty:
        return None

    edges, labels = _quantile_bins(sel["n_features"], n_bins=4)
    if edges is None:
        return None

    sel = sel.copy()
    sel["feat_bin"] = pd.cut(sel["n_features"], bins=edges, labels=labels, include_lowest=True)

    bin_labels = pd.cut(sel["n_features"], bins=edges, labels=labels, include_lowest=True)
    counts = bin_labels.value_counts().to_dict()
    counts = {str(k): v for k, v in counts.items()}

    agg = (
        sel.groupby(["model", "feat_bin"])
        .agg(value=("value", "median"), n_features=("n_features", "median"))
        .reset_index()
    )
    agg["bin_center"] = agg["n_features"]
    agg["Models"] = agg["model"].map(Models).fillna(agg["model"])

    present = [m for m in MODEL_ORDER if m in agg["Models"].unique()]
    if present:
        agg["Models"] = pd.Categorical(agg["Models"], categories=present, ordered=True)

    palette = _get_palette()

    plt.figure(figsize=_get_figsize())
    ax = plt.gca()

    for model_disp in present:
        mdf = agg[agg["Models"] == model_disp]
        if mdf.empty:
            continue
        ax.plot(
            mdf["bin_center"],
            mdf["value"],
            marker="o",
            label=model_disp,
            color=palette.get(model_disp),
        )

    if logx:
        ax.set_xscale("log")

    _place_small_legend(ax, title="Model")

    plt.tight_layout()
    safe_tau = _safe_tau_str(tau)
    out_subdir = os.path.join(out_dir, metric, "by_features", "comparison_binned")
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
    _write_bin_summary(out_path, edges, labels, counts)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_metric_by_categorical_features(df, out_dir, logx=False, kind="line"):
    df, model_order = _prepare_models(df)
    palette = _get_palette()

    for metric, metric_df in df.groupby("metric"):
        for tau, tau_df in metric_df.groupby("tau"):
            sub = tau_df.dropna(subset=["n_categorical_features"])
            if sub.empty:
                continue

            edges, labels = _categorical_bins(sub["n_categorical_features"])
            if edges is None:
                # only zeros or insufficient variation: plot only zero bin, no hue
                sub = sub.copy()
                sub["cat_bin"] = sub["n_categorical_features"].apply(lambda x: "0" if x == 0 else "")
                zero_count = int((sub["cat_bin"] == "0").sum())

                plt.figure(figsize=_get_figsize())
                ax = sns.boxplot(
                    x="cat_bin",
                    y="value",
                    data=sub[sub["cat_bin"] == "0"],
                    showcaps=True,
                    showfliers=False,
                    whiskerprops={"linewidth": 0.5},
                )

                plt.xlabel("Number of Categorical Features")
                metric_label = METRIC_LABELS.get(metric, metric)
                plt.ylabel(metric_label)
                title_tau = f" (τ={tau})" if tau is not None else ""
                plt.title(f"By Categorical Features — {metric_label}{title_tau}")

                out_path = _make_boxplot_out_path(out_dir, metric, "by_categorical_features", tau)
                _write_bin_summary(out_path, [], [], {}, zero_count=zero_count)

                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()
                continue

            sub = sub.copy()
            pos_mask = sub["n_categorical_features"] > 0
            sub.loc[~pos_mask, "cat_bin"] = "0"
            sub.loc[pos_mask, "cat_bin"] = pd.cut(
                sub.loc[pos_mask, "n_categorical_features"],
                bins=edges,
                labels=labels,
                include_lowest=True,
            )

            counts = sub["cat_bin"].value_counts().to_dict()
            zero_count = counts.pop("0", 0)

            plt.figure(figsize=_get_figsize())
            ax = sns.boxplot(
                x="cat_bin",
                y="value",
                hue="Models",
                hue_order=model_order,
                palette=palette,
                data=sub,
                showcaps=True,
                showfliers=False,
                whiskerprops={"linewidth": 0.5},
            )

            _place_small_legend(ax, title="Model")

            plt.xlabel("Number of Categorical Features (bin label / range)")
            cat_tick_labels = ["0"]
            for i, lbl in enumerate(labels):
                low = edges[i]
                high = edges[i + 1]
                cat_tick_labels.append(f"{lbl} ({low:.0f}-{high:.0f})")
            plt.xticks(range(len(cat_tick_labels)), cat_tick_labels)

            metric_label = METRIC_LABELS.get(metric, metric)
            plt.ylabel(metric_label)
            title_tau = f" (τ={tau})" if tau is not None else ""
            plt.title(f"By Categorical Features — {metric_label}{title_tau}")

            out_path = _make_boxplot_out_path(out_dir, metric, "by_categorical_features", tau)
            _write_bin_summary(out_path, edges, labels, counts, zero_count=zero_count)

            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()


def plot_all_models_together_categorical_features(df, out_dir, metric, tau=None, logx=False):
    sel = df[df["metric"] == metric]
    if tau is not None:
        sel = sel[sel["tau"] == tau]
    if sel.empty:
        return None

    sel = sel.dropna(subset=["n_categorical_features"])
    edges, labels = _categorical_bins(sel["n_categorical_features"])

    if edges is None:
        sel = sel.copy()
        sel["cat_bin"] = sel["n_categorical_features"].apply(lambda x: "0" if x == 0 else "")
        zero_count = int((sel["cat_bin"] == "0").sum())

        agg = (
            sel[sel["cat_bin"] == "0"]
            .groupby(["model", "cat_bin"])
            .agg(value=("value", "median"), n_cat=("n_categorical_features", "median"))
            .reset_index()
        )
        agg["bin_center"] = agg["n_cat"]
        counts = {}
    else:
        sel = sel.copy()
        pos_mask = sel["n_categorical_features"] > 0
        sel.loc[~pos_mask, "cat_bin"] = "0"
        sel.loc[pos_mask, "cat_bin"] = pd.cut(
            sel.loc[pos_mask, "n_categorical_features"],
            bins=edges,
            labels=labels,
            include_lowest=True,
        )

        counts = sel["cat_bin"].value_counts().to_dict()
        zero_count = counts.pop("0", 0)

        agg = (
            sel.groupby(["model", "cat_bin"])
            .agg(value=("value", "median"), n_cat=("n_categorical_features", "median"))
            .reset_index()
        )
        agg["bin_center"] = agg["n_cat"]

    agg["Models"] = agg["model"].map(Models).fillna(agg["model"])
    present = [m for m in MODEL_ORDER if m in agg["Models"].unique()]
    if present:
        agg["Models"] = pd.Categorical(agg["Models"], categories=present, ordered=True)

    palette = _get_palette()

    plt.figure(figsize=_get_figsize())
    ax = plt.gca()

    for model_disp in present:
        mdf = agg[agg["Models"] == model_disp]
        if mdf.empty:
            continue
        ax.plot(
            mdf["bin_center"],
            mdf["value"],
            marker="o",
            label=model_disp,
            color=palette.get(model_disp),
        )

    ax.set_xlabel("Number of Categorical Features (bin center)")
    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(metric_label)
    title_tau = f" (τ={tau})" if tau is not None else ""
    ax.set_title(f"Model Comparison — {metric_label}{title_tau}")

    if logx:
        ax.set_xscale("log")

    _place_small_legend(ax, title="Model")

    plt.tight_layout()
    safe_tau = _safe_tau_str(tau)
    out_subdir = os.path.join(out_dir, metric, "by_categorical_features", "comparison_binned")
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")

    if edges is None:
        _write_bin_summary(out_path, [], [], {}, zero_count=zero_count)
    else:
        _write_bin_summary(out_path, edges, labels, counts, zero_count=zero_count)

    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# --- HTML index ------------------------------------------------------------

def generate_html_index(out_dir, metrics):
    """
    Simple index.
    - Boxplots now live under: <out_dir>/boxplots/<group>/<metric>/boxplot_tau_<tau>.png
    - Line/comparison plots remain under: <out_dir>/<metric>/<...>/comparison_binned/
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
\t<meta charset="UTF-8">
\t<meta name="viewport" content="width=device-width, initial-scale=1.0">
\t<title>SQR Results — Plot Overview</title>
\t<style>
\t\tbody { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
\t\th1 { color: #2c3e50; }
\t\t.metric-section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; }
\t\t.plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
\t\t.plot-item { background: #f9f9f9; padding: 12px; border-radius: 5px; border: 1px solid #e0e0e0; }
\t\t.plot-item a { display: inline-block; margin-top: 6px; padding: 8px 12px; background: #3498db; color: white; text-decoration: none; border-radius: 3px; }
\t</style>
</head>
<body>
\t<h1>SQR Analysis — Plot Overview</h1>
\t<div class="nav">
\t\t<strong>Quick Navigation:</strong>
"""
    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        html_content += f'\t\t<a href="#{metric}">{metric_label}</a>\n'
    html_content += "\t</div>\n"

    for metric in metrics:
        metric_label = METRIC_LABELS.get(metric, metric)
        metric_dir = os.path.join(out_dir, metric)

        html_content += f'\t<div class="metric-section">\n\t\t<h2 id="{metric}">{metric_label}</h2>\n'

        # Boxplots (new)
        html_content += "\t\t<h3>Boxplots</h3>\n"
        html_content += '\t\t<div class="plot-grid">\n'
        for grp in ["by_instances", "by_features", "by_categorical_features"]:
            box_dir = os.path.join(out_dir, "boxplots", grp, metric)
            if not os.path.exists(box_dir):
                continue
            for png_file in sorted(os.listdir(box_dir)):
                if png_file.endswith(".png"):
                    rel_path = os.path.relpath(os.path.join(box_dir, png_file), out_dir)
                    html_content += (
                        '\t\t\t<div class="plot-item">\n'
                        f"\t\t\t\t<h4>{grp} — {png_file}</h4>\n"
                        f'\t\t\t\t<a href="{rel_path}">View Plot</a>\n'
                        "\t\t\t</div>\n"
                    )
        html_content += "\t\t</div>\n"

        # Comparison plots (existing structure)
        if os.path.exists(metric_dir):
            for section in ["by_instances", "by_features", "by_categorical_features"]:
                sec_dir = os.path.join(metric_dir, section, "comparison_binned")
                if not os.path.exists(sec_dir):
                    continue
                html_content += f"\t\t<h3>Comparison (binned) — {section}</h3>\n"
                html_content += '\t\t<div class="plot-grid">\n'
                for png_file in sorted(os.listdir(sec_dir)):
                    if png_file.endswith(".png"):
                        rel_path = os.path.relpath(os.path.join(sec_dir, png_file), out_dir)
                        html_content += (
                            '\t\t\t<div class="plot-item">\n'
                            f"\t\t\t\t<h4>{png_file}</h4>\n"
                            f'\t\t\t\t<a href="{rel_path}">View Plot</a>\n'
                            "\t\t\t</div>\n"
                        )
                html_content += "\t\t</div>\n"

        html_content += "\t</div>\n"

    html_content += "</body>\n</html>\n"

    index_path = os.path.join(out_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return index_path


# --- main -----------------------------------------------------------------

def main(results_dir="results", summary_tsv="all_summary_stats.tsv", out_dir="plots", metrics=None, models=None, fig_mode="single"):
    _set_style()
    _set_fig_mode(fig_mode)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    summary_df = read_summary_stats(summary_tsv)
    df = collect_results(results_dir, summary_df)

    if df.empty:
        print("No results found to plot.")
        return

    if metrics is None:
        metrics = sorted(df["metric"].unique())

    if models is not None:
        df = df[df["model"].isin(models)]

    # Boxplots by instances (go to out_dir/boxplots/...)
    plot_metric_by_instances(df[df["metric"].isin(metrics)], out_dir)

    # Boxplots by features
    if "n_features" in summary_df.columns:
        df = df.merge(summary_df[["n_features"]], left_on="dataset", right_index=True, how="left")
        plot_metric_by_features(df[df["metric"].isin(metrics)], out_dir)
        for metric in metrics:
            for tau in df[df["metric"] == metric]["tau"].dropna().unique():
                plot_all_models_together_features(df, out_dir, metric, tau=tau)

    # Boxplots by categorical features
    if "n_categorical_features" in summary_df.columns:
        df = df.merge(summary_df[["n_categorical_features"]], left_on="dataset", right_index=True, how="left")
        plot_metric_by_categorical_features(df[df["metric"].isin(metrics)], out_dir)
        for metric in metrics:
            for tau in df[df["metric"] == metric]["tau"].dropna().unique():
                plot_all_models_together_categorical_features(df, out_dir, metric, tau=tau)

    # Comparisons by instances (line plots)
    for metric in metrics:
        for tau in df[df["metric"] == metric]["tau"].dropna().unique():
            plot_all_models_together(df, out_dir, metric, tau=tau)

    index_path = generate_html_index(out_dir, metrics)
    print(f"Plots saved to: {os.path.abspath(out_dir)}")
    print(f"HTML index generated: {os.path.abspath(index_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results per metric/model/tau against dataset size.")
    parser.add_argument("--results_dir", default="results", help="Directory with result JSON files")
    parser.add_argument("--summary_tsv", default="all_summary_stats.tsv", help="TSV with dataset metadata (n_instances)")
    parser.add_argument("--out_dir", default="plots", help="Output directory for plots")
    parser.add_argument("--metrics", nargs="*", default=None, help="Specific metrics to plot")
    parser.add_argument("--models", nargs="*", default=None, help="Specific models to plot")
    parser.add_argument(
        "--fig_mode",
        choices=["single", "half"],
        default="single",
        help="Figure size mode for publication (single or half width)",
    )
    args = parser.parse_args()

    main(
        results_dir=args.results_dir,
        summary_tsv=args.summary_tsv,
        out_dir=args.out_dir,
        metrics=args.metrics,
        models=args.models,
        fig_mode=args.fig_mode,
    )