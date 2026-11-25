"""Shared matplotlib style for the figures of the paper (scripts/plot_*.py and analyse_*.py).

Figures are drawn at their printed size, so the fonts below are the fonts in the PDF:
FULL_W is the JAIR text width (450 pt = 6.25 in), and PART_W is the width of figures placed at 0.7 of it.
"""
FULL_W = 6.25
PART_W = 0.7 * FULL_W

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.handlelength": 1.6,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "axes.linewidth": 0.6,
    "axes.titlepad": 3,
    "axes.labelpad": 2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# One color and line style per training variant, used by every control figure.
VARIANTS = {
    "base": dict(label="Baseline", color="#1f77b4", ls="-", marker="o"),
    "cov": dict(label="+Cov", color="#ff7f0e", ls="--", marker="o"),
    "ctrl": dict(label="+Ctrl", color="#2ca02c", ls="--", marker="o"),
    "both": dict(label="+Aux", color="#d62728", ls="-", marker="o"),
}



def minor_log_label(y, _pos=None):
    """Label minor log ticks only at mantissa 2, 3 and 5 (avoids crowded tick labels on short log axes)."""
    import math
    if y <= 0:
        return ""
    e = math.floor(math.log10(y) + 1e-9)
    mant = round(y / 10 ** e, 6)
    if mant not in (2.0, 3.0, 5.0):
        return ""
    return f"${int(mant)}\\times10^{{{e}}}$"


def control_panels(left, right, n_offset, ylabel, path, errors=None, legend_loc=("best", "best")):
    """Two-panel control figure (left: versus training samples m; right: versus latent dimension n).

    left, right: DataFrames indexed by m and by n_enc, with columns among Baseline, +Cov, +Ctrl, +Aux.
    errors: optional (left_sd, right_sd) DataFrames of the same shape, drawn as error bars.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {v["label"]: v for v in VARIANTS.values()}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 2, figsize=(PART_W, 2.0), layout="constrained")
        for k, (panel, x0) in enumerate(((left, 0), (right, n_offset))):
            for a in [c for c in names if c in panel.columns]:
                v = names[a]
                x = panel.index.values + x0
                lab = a
                if errors is not None:
                    ax[k].errorbar(x, panel[a], yerr=errors[k][a], color=v["color"], ls=v["ls"], marker=v["marker"],
                                   ms=3, lw=1.1, capsize=1.5, elinewidth=0.6, label=lab)
                else:
                    ax[k].plot(x, panel[a], color=v["color"], ls=v["ls"], marker=v["marker"], ms=3, lw=1.1, label=lab)
            ax[k].set_xscale("log", base=10 if k == 0 else 2)
            ax[k].set_xlabel("Training samples $m$" if k == 0 else "Latent dimension $n$")
            ax[k].set_ylabel(ylabel)
            ax[k].grid(True, which="major", ls="--")
        h, lab = ax[0].get_legend_handles_labels()
        fig.legend(h, lab, ncol=len(lab), loc="outside upper center", frameon=False, columnspacing=1.2, handlelength=2.2)
        fig.savefig(path)
        plt.close(fig)
