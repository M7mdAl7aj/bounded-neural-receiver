# Bounded neural receiver

A neural receiver for 6G cmWave whose output never strays more than a fixed amount, $\Delta_{\max}$, from a classical
receiver's answer, bit by bit. Set $\Delta_{\max} = 0$ and the classical receiver comes back exactly, whatever the network's
weights; let $\Delta_{\max} \to \infty$ and it is the network alone.

$$L_{\mathrm{R5}} = L_{\mathrm{c}} + \mathrm{clip}\left(L_{\mathrm{R3}} - L_{\mathrm{c}},\ \pm\Delta_{\max}\right)$$

where $L_{\mathrm{c}}$ is the classical receiver's log-likelihood ratio for a coded bit, $L_{\mathrm{R3}}$ the trained
network's, and $L_{\mathrm{R5}}$ the bounded output passed to the LDPC decoder.

**Paper:** <https://doi.org/10.1109/OJCOMS.2026.3756446> (IEEE Open Journal of the Communications Society, open access).

**Try it:** <https://m7mdal7aj.github.io/bounded-neural-receiver/>. Drag $\Delta_{\max}$, switch between six channel and
impairment conditions, and hover any chart for its values. The page shows measured results computed in advance; nothing is
simulated in your browser. To run the real decoder yourself, see [the notebook](#the-notebook).

## What is here

- `docs/index.html`: the interactive page, one self-contained file (it works offline).
- `site/`: how the page is built. `build_data.py` reads `results/` and stops unless every operating point and its
  cross-seed spread equals the published Table V (`tests/reference_values.json`); `build_page.js` then assembles the page;
  `check_page.py` opens it with the network blocked and checks the equations and the hover on every chart.
- `results/`: the measured block-error-rate curves and 10% operating points behind the page (three seeds per point), and one
  sample slot of real LLRs per condition.
- `receiver/`: the link-level simulator with its RF impairments (TR 38.803 phase noise, IQ imbalance), the classical
  receivers, the neural receivers (network definitions and evaluation; the training code is not included), the bound, and
  the offline tools that decode stored LLRs.
- `notebook/`: a notebook that decodes real stored LLRs at any bound with the paper's LDPC decoder, on a CPU; its two
  samples are in `notebook/data/`.
- `scripts/`: the figure scripts.

## The notebook

[`notebook/bounded_receiver_cpu.ipynb`](notebook/bounded_receiver_cpu.ipynb) runs the bound on real data with the real 5G NR
LDPC decoder. It checks that $\Delta_{\max} = 0$ returns the classical receiver's LLRs bit for bit, decodes two bundled
samples (one from the condition the network was trained on, one from a harsher IQ imbalance) at the paper's six bounds and
confirms the outcome of every block matches the recorded decode, decodes them at any other bound, and shows how to put your
own receiver's LLRs through the bound. GitHub shows it with its outputs. To run it yourself (no GPU needed; it takes a few
minutes on a CPU):

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r notebook/requirements.txt
jupyter notebook notebook/bounded_receiver_cpu.ipynb
```

## Running the code

The code runs with Python 3.10, TensorFlow 2.16.2 with tf-keras 2.16.0 (set `TF_USE_LEGACY_KERAS=1`), Sionna 1.2.2 and
numpy 1.26.4, in a conda environment named `paper3`. The TensorRT latency scripts in `receiver/latency/` also need TensorRT
(see `receiver/latency/README.md`).

To rebuild the page from `results/`:

```bash
python site/build_data.py build/data.json
node site/build_page.js site/template.html build/data.json docs/index.html
```

## Citing

Please cite the paper:

> A. Elnashar and M. AlHaj, "A Bounded Neural Receiver with Exact Classical Fallback for 6G cmWave: Fair-Baseline Evaluation," *IEEE Open Journal of the Communications Society*, early access, 2026, doi: [10.1109/OJCOMS.2026.3756446](https://doi.org/10.1109/OJCOMS.2026.3756446).

The software is by Mohammed AlHaj; see `CITATION.cff`.

## Licence

MIT, copyright 2026 Mohammed AlHaj (see `LICENSE`). Included third-party files: KaTeX (MIT, `site/vendor/katex/LICENSE`),
and the Inter and IBM Plex Mono fonts (SIL Open Font License, `site/vendor/fonts/`).
