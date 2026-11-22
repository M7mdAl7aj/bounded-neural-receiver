#!/usr/bin/env python3
"""Draw the two architecture diagrams that the paper's Fig. 1 and Fig. 2 show
(the published versions were re-rendered for the final layout). The
as-measured architecture:
  Fig. 1 - SIMO 1x2 primary chain, PN+IQI impairments, receivers R1/R3/R5.
  Fig. 2 - R5 = inference-time clip of trained R3 onto L2-real +/- Dmax.

Aspect ratios: 2.726 (Fig. 1) and 2.231 (Fig. 2).

Outputs: figures/fig00a_system_architecture.png, figures/fig00b_r5_block.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

BLUE = "#e8f4fd"; BLUE2 = "#d5e8f7"; YELL = "#fdf6e3"; PINK = "#fde8e8"
EDGE = "#2c3e50"


def box(ax, x, y, w, h, text, fc=BLUE, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=EDGE, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=EDGE, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=12, color=color, lw=lw))


def fig1_system(path):
    # authored near single-column width so labels stay legible when placed at ~3.3 in
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")

    yc, hh = 22.0, 13.0
    box(ax, 1.0, yc, 17.0, hh, "LDPC enc.\n256-QAM\n(MCS 27)", fs=5.6)
    box(ax, 20.5, yc, 13.5, hh, "OFDM\n51 PRB", fs=5.6)
    box(ax, 36.5, yc, 13.5, hh, "gNB\n1 TX", fc=BLUE2, bold=True, fs=5.8)
    box(ax, 52.5, yc, 15.0, hh, "CDL-C/D/E\n(TR 38.901)", fc=YELL, fs=5.6)
    box(ax, 70.0, yc, 12.5, hh, "UE\n2 RX", fc=BLUE2, bold=True, fs=5.8)
    box(ax, 85.0, yc, 14.0, hh, "Rx + LDPC\nR1 | R3 | R5", fc=PINK, fs=5.4, bold=True)

    for x0, x1 in ((18.0, 20.5), (34.0, 36.5), (50.0, 52.5), (67.5, 70.0), (82.5, 85.0)):
        arrow(ax, x0, yc + hh / 2, x1, yc + hh / 2, lw=1.0)
    ax.text(51.2, yc + hh / 2 + 1.6, r"$x(k)$", fontsize=5.4, ha="center", va="bottom")
    ax.text(83.7, yc + hh / 2 + 1.6, r"$y(k)$", fontsize=5.4, ha="center", va="bottom")

    # impairment boxes feeding the channel/UE (arrows land on box tops, clear of x(k)/y(k))
    box(ax, 44.0, 41.0, 21.0, 10.0, "Phase noise\n(TR 38.803)", fc=YELL, fs=5.4)
    box(ax, 68.0, 41.0, 24.0, 10.0, "IQ imbalance\nIRR = 25 dB", fc=YELL, fs=5.4)
    arrow(ax, 59.5, 41.0, 59.5, yc + hh, color="#8e6d1a", lw=0.9)   # -> CDL box top
    arrow(ax, 76.0, 41.0, 76.0, yc + hh, color="#8e6d1a", lw=0.9)   # -> UE box top

    ax.text(50.0, 11.5, "Primary: SIMO 1×2 (one stream, two UE chains).",
            fontsize=5.4, ha="center")
    ax.text(50.0, 5.0,
            "Secondary rank-2: 256-element gNB, wideband SVD-precoding surrogate\n"
            "for the DTR-AS-MBWC beamformer (Sec. II-F).",
            fontsize=4.9, ha="center", color="#555555")
    fig.savefig(path, dpi=400, bbox_inches=None)
    plt.close(fig)


def fig2_r5(path):
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")

    # inputs: mirror feeds R3 only (top); grid fans to both R3 and L2 (no crossing)
    box(ax, 1.0, 52.0, 23.0, 13.0, "Conjugate-mirror\n" + r"$Y^*[N{-}k]$", fc=BLUE2, fs=5.4)
    box(ax, 1.0, 30.0, 23.0, 14.0, "Received grid\nRe/Im, DMRS,\nnoise level", fs=5.4)

    # processing (narrower so the label gap before the clip is clear)
    box(ax, 33.0, 48.0, 28.0, 15.0,
        "Trained R3\n11×ResBlock, 128 ch\n(unconstrained)", fc=BLUE2, fs=5.2, bold=True)
    box(ax, 33.0, 25.0, 28.0, 14.0,
        "L2-real baseline\nLS → IQI inv. →\nLMMSE → demap", fc=YELL, fs=5.2, bold=True)

    # clip + output  (title line + equation line; font sized to fit inside the box)
    box(ax, 73.0, 38.0, 24.0, 13.0,
        "Clip\n" + r"$L_{R5}{=}\mathrm{clip}(L_{R3},L_c{\pm}\Delta_{\max})$", fc=PINK, fs=4.3, bold=True)

    # arrows (no crossings): mirror->R3 ; grid->R3 ; grid->L2 ; labels centred in the gap
    arrow(ax, 24.0, 58.0, 33.0, 57.0, lw=0.9)                 # mirror -> R3
    arrow(ax, 24.0, 38.0, 33.0, 53.0, lw=0.9)                 # grid -> R3 (up)
    arrow(ax, 24.0, 35.0, 33.0, 32.0, lw=0.9)                 # grid -> L2 (down)
    arrow(ax, 61.0, 54.0, 73.0, 47.0, lw=0.9); ax.text(67.0, 53.0, r"$L_{R3}$", fontsize=5.0, ha="center", va="center")
    arrow(ax, 61.0, 33.0, 73.0, 42.0, lw=0.9); ax.text(67.0, 34.0, r"$L_c$", fontsize=5.0, ha="center", va="center")
    arrow(ax, 97.0, 44.5, 100.0, 44.5, lw=0.9); ax.text(98.6, 48.5, r"$L_{R5}$", fontsize=5.0, ha="center", va="center")

    ax.text(50.0, 13.0,
            r"Theorem 1:  $\Delta_{\max}{=}0\Rightarrow L_{R5}{=}L_c$  (exact fallback);  "
            r"$\Delta_{\max}{\to}\infty\Rightarrow L_{R5}{=}L_{R3}$;",
            fontsize=5.0, ha="center", color="#8e2020")
    ax.text(50.0, 8.0, r"$\Vert L_{R5}-L_c\Vert_\infty \leq \Delta_{\max}$ for all inputs.",
            fontsize=5.0, ha="center", color="#8e2020")
    ax.text(50.0, 2.5, "Inference-time projection; no residual training.",
            fontsize=4.9, ha="center", color="#555555")
    fig.savefig(path, dpi=400, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    import os
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    fig1_system(os.path.join(d, "fig00a_system_architecture.png"))
    fig2_r5(os.path.join(d, "fig00b_r5_block.png"))
    print("written fig00a_system_architecture.png + fig00b_r5_block.png")
