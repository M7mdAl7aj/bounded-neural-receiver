#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graphical abstract of the paper.

Every number and phrase is taken from the paper; mathtext is used
for every expression so subscripts, +/- and the infinity norm are typeset (no
literal underscores in the output). IEEE aspect 660:295, rendered at 4x PNG
(2640x1180) and vector PDF with embedded fonts.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                            'font.family': 'DejaVu Sans',
                            'mathtext.fontset': 'dejavusans'})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W_IN, H_IN, DPI = 6.60, 2.95, 400          # 660:295 at 4x = 2640x1180 px
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

INK, EDGE = '#1a1a2e', '#4a4a6a'
C1, C2, C3 = '#eaf1fb', '#fdeeee', '#eefaf0'
STRIP, RES = '#f4f0fa', '#fff8e8'
MX = 0.045                                   # >=4% margins

def box(x, y, w, h, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008',
                 fc=fc, ec=EDGE, lw=0.9, mutation_aspect=W_IN/H_IN))

# ---- title band (the paper's title, verbatim) ----
ax.text(0.5, 0.955, 'A Bounded Neural Receiver with Exact Classical Fallback for 6G cmWave:',
        ha='center', va='center', fontsize=8.6, weight='bold', color=INK)
ax.text(0.5, 0.885, 'Fair-Baseline Evaluation',
        ha='center', va='center', fontsize=8.6, weight='bold', color=INK)

# ---- three-stage flow ----
BY, BH = 0.545, 0.27
BW = 0.272; GAP = (1 - 2*MX - 3*BW) / 2
xs = [MX, MX + BW + GAP, MX + 2*(BW + GAP)]
box(xs[0], BY, BW, BH, C1)
ax.text(xs[0]+BW/2, BY+BH-0.055, 'Fair classical baseline', ha='center', va='center',
        fontsize=7.4, weight='bold', color=INK)
ax.text(xs[0]+BW/2, BY+BH-0.115, r'$L_{c}$ (L2-real)', ha='center', va='center', fontsize=7.0, color=INK)
ax.text(xs[0]+BW/2, BY+0.075, 'DMRS-estimated IQ compensation,\nno oracle knowledge',
        ha='center', va='center', fontsize=6.0, color=INK)
box(xs[1], BY, BW, BH, C2)
ax.text(xs[1]+BW/2, BY+BH-0.055, 'Neural receiver R3', ha='center', va='center',
        fontsize=7.4, weight='bold', color=INK)
ax.text(xs[1]+BW/2, BY+0.10, 'DeepRx-style CNN\n7.125 GHz, 51 PRB, 30 kHz SCS,\n256-QAM, two receive chains',
        ha='center', va='center', fontsize=6.0, color=INK)
box(xs[2], BY, BW, BH, C3)
ax.text(xs[2]+BW/2, BY+BH-0.055, 'Bounded receiver R5', ha='center', va='center',
        fontsize=7.4, weight='bold', color=INK)
ax.text(xs[2]+BW/2, BY+0.10,
        r'$L_{R5} = L_{c} + \mathrm{clip}(L_{R3} - L_{c},\, \pm\Delta_{max})$',
        ha='center', va='center', fontsize=6.6, color=INK)
for a in range(2):
    ax.add_patch(FancyArrowPatch((xs[a]+BW+0.004, BY+BH/2), (xs[a+1]-0.004, BY+BH/2),
                 arrowstyle='-|>', mutation_scale=9, color=EDGE, lw=1.0))

# ---- property strip ----
PY_, PH = 0.365, 0.135
box(MX, PY_, 1-2*MX, PH, STRIP)
props = [r'$\Delta_{max} = 0$: exact classical fallback',
         r'$\Delta_{max} \to \infty$: recovers R3',
         r'$\Vert L_{R5} - L_{c} \Vert_{\infty} \leq \Delta_{max}$ (Theorem 1)',
         'per-bit LLR-deviation certificate,\nnot a BLER guarantee']
pw = (1-2*MX)/4
for k, s in enumerate(props):
    ax.text(MX + pw*(k+0.5), PY_+PH/2, s, ha='center', va='center', fontsize=5.9, color=INK)

# ---- results strip ----
RY, RH = 0.185, 0.135
box(MX, RY, 1-2*MX, RH, RES)
res = ['1.44 dB gain over L2-real\nat 10% BLER',
       r'$\Delta_{max} = \ln 4$ captures 63%,' + '\n' + r'$\ln 16$ captures 99%',
       'benign under channel shift,\nprotective under impairment shift']
rw = (1-2*MX)/3
for k, s in enumerate(res):
    ax.text(MX + rw*(k+0.5), RY+RH/2, s, ha='center', va='center', fontsize=6.1, color=INK)

# ---- footer ----
ax.text(0.5, 0.10, 'Fair-baseline evaluation · exact classical fallback · auditable LLR-deviation envelope',
        ha='center', va='center', fontsize=6.4, style='italic', color=EDGE)

import os
out = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out, 'figures', 'graphical_abstract_preview.png'), dpi=DPI)
fig.savefig(os.path.join(out, '..', '..', 'neural_receiver_graphical_abstract_ieee.pdf'))
print('rendered PNG (%dx%d px) + PDF' % (W_IN*DPI, H_IN*DPI))
