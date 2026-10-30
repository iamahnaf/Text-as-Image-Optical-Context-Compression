# 📄 Text-as-Image Optical Context Compression for Vision-Language Models
### *How Far Can Rendered Text Replace Text Tokens in Long-Context Retrieval?*

[![Paper](https://img.shields.io/badge/IEEE%20Format-Research%20Paper-blue.svg)](paper/main.pdf)
[![Model](https://img.shields.io/badge/Model-Gemma--4%20(31B)-8A2BE2.svg)](https://ollama.com/library/gemma4)
[![Benchmark](https://img.shields.io/badge/Benchmark-RULER%20Synthetic%20NIAH-orange.svg)](https://arxiv.org/abs/2404.06654)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Executive Summary

Modern Large Language Models (LLMs) face steep computational bottlenecks when ingesting long document contexts. Because transformer self-attention scales quadratically ($\mathcal{O}(N^2)$) with token count, feeding raw text tokens leads to massive memory footprints, high latency, and steep inference costs. 

**Optical Context Compression (Text-as-Image)** offers an alternative paradigm: render long contextual passages into high-contrast 2D images and allow the Vision Encoder of a multimodal foundation model to compress thousands of text tokens into a compact sequence of visual tokens.

This repository hosts the official experimental framework, codebase, benchmarks, and research paper exploring **Text-as-Image Compression** using **Gemma 4 (31B)** under a controlled **RULER needle-in-a-haystack (NIAH)** evaluation grid.

```
                           [ Long Text Context (500 - 1500 tokens) ]
                                            │
                ┌───────────────────────────┴───────────────────────────┐
                ▼                                                       ▼
      [ Condition A: Baseline ]                               [ Condition B: Optical ]
         Raw Text Tokens (m)                                 Render to 2D Canvas Image
                │                                                       │
         + Query Tokens (q)                                   Pass to Vision Encoder
                │                                                       │
        639 - 1,647 Tokens                                   Fixed ~292 Visual Tokens (k)
                │                                                       │
                ▼                                                       ▼
      Ollama: gemma4:31b (T=0)                                Ollama: gemma4:31b (T=0)
  (100% Accuracy | High Memory)                        (Up to 82.3% Token Savings | CR 5.7x)
```

---

## 🔍 The Problem & Research Motivation

| Traditional Text-Only Context | Optical Context Compression (Text-as-Image) |
| :--- | :--- |
| **High Prompt Token Cost**: Grows linearly at $\approx 1.35\times$ words. | **Fixed Visual Token Footprint**: Ingested in $\approx 292$ tokens constant footprint. |
| **Quadratic Attention Bottleneck**: Long prefill sequences strain GPU VRAM. | **Massive Compression Ratios**: Delivers $2.17\times$ to $5.92\times$ token reductions. |
| **100% Text Character Fidelity**: Exact lexical match preserved. | **Optical Degradation Cliff**: Visual token density induces OCR misreadings at high CR. |

### Research Questions (RQs)
1. **RQ1 (Compression Efficacy):** How efficiently can textual context be compressed into visual tokens without losing retrieval accuracy?
2. **RQ2 (Model Transferability):** How does compression resilience differ across multimodal architectures (Gemma 4 vs. Qwen 3.5)? *(Planned for Future Work)*
3. **RQ3 (Degradation Cliff):** At what document length and Compression Ratio (CR) does retrieval fidelity collapse?
4. **RQ4 (Latency Trade-offs):** Does visual token reduction yield real-world inference speedups in standard serving runtimes?

---

## 🖼️ Visualizations & Key Findings

### 1. Token Scaling, Compression Curves, and Operational Regimes
The foundational empirical findings and mathematical regression models are synthesized below:

<p align="center">
  <img src="results/threshold_analysis.png" alt="Threshold Analysis Dashboard" width="95%"/>
</p>

* **Figure (a) Prompt Token Scaling:** Raw text prompt tokens scale as $T_{\text{text}}(W) \approx 1.3524 \cdot W + 47.16$ ($R^2 > 0.999$), while visual representation maintains an invariant footprint of $T_{\text{image}} \approx 292.2 \pm 4.5$ tokens. The break-even crossing occurs at **$W \approx 181$ words**.
* **Figure (b) Compression Ratio (CR) Curve:** $\mathrm{CR}(W) = \frac{T_{\text{text}}(W)}{T_{\text{image}}}$. Operational milestones are achieved at $\mathrm{CR}=1.0$ (181 words), $\mathrm{CR}=2.0$ (397 words), $\mathrm{CR}=3.0$ (613 words), and $\mathrm{CR}=4.0$ (829 words).
* **Figure (c) The Optimal Sweet Spot:** An empirical sweet spot exists between **397 and 434 words**, where optical compression achieves **$\ge 50\%$ prompt token savings** while retaining **100% needle retrieval accuracy**.
* **Figure (d) Operational Guidelines:** Five distinct operational zones delineate when to employ raw text vs. image rendering.

---

### 2. Retrieval Accuracy Across Budgets & Degradation Heatmap

<p align="center">
  <img src="results/plot_1_heatmap.png" alt="Retrieval Accuracy Heatmap" width="48%"/>
  <img src="results/plot_2_degradation_cliff.png" alt="Degradation Cliff" width="48%"/>
</p>

* **Heatmap Analysis:**
  * **500 Tokens:** 100% Retrieval accuracy across all 5 visual budget canvas configurations ($k=70$ to $k=1120$).
  * **1000 Tokens:** 80% Retrieval accuracy (only $k=560$ suffered a minor single-character transcription error: `CIPHER8-8028-OMEGA`).
  * **1500 Tokens:** 20% Retrieval accuracy. 4 out of 5 settings collapsed due to illegible rendering density. Only the ultra-wide $k=1120$ canvas ($1400\text{px}$, $22\text{pt}$) successfully extracted the needle.
* **Degradation Cliff:** Retrieval performance does not fall off a sudden cliff at $\mathrm{CR} \approx 4.5\times$; rather, it exhibits a progressive degradation corridor between $\mathrm{CR} = 3.6\times$ and $5.3\times$.

---

### 3. Token Savings & Budget Scaling Dynamics

<p align="center">
  <img src="results/plot_3_token_savings.png" alt="Prompt Token Savings" width="48%"/>
  <img src="results/plot_4_budgets_curves.png" alt="Multi-Budget Curves" width="48%"/>
</p>

* **Token Savings:** Image representation cuts prompt token size by **53.5%** (500 tokens), **72.9%** (1000 tokens), and **82.3%** (1500 tokens).
* **Nominal vs. Physical Token Budgets:** Across all 15 grid runs, the serving engine mapped canvas images to roughly **$\approx 292$ tokens**, regardless of nominal canvas width ($400\text{px}$ to $1400\text{px}$). Thus, nominal budget settings govern **drawing density and font readability**, rather than true token slot allocations.

---

## 📊 Summary of Experimental Results

Evaluated on **Gemma 4 (31B)** via Ollama API at deterministic decoding (`temperature = 0.0`):

| Context Length (Target) | Document Words ($W$) | Text Tokens ($m+q$) | Image Tokens ($k+q$) | Compression Ratio ($\mathrm{CR}$) | Token Savings (\%) | Text Acc. (\%) | Image Acc. (\%) | Mean Speedup ($t_A/t_B$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **500 tokens** | 434 | 639 | 297 | **$2.17\times$** | **53.5%** | 100% | **100.0%** | $0.61\times$ |
| **1000 tokens** | 768 | 1077 | 294 | **$3.69\times$** | **72.9%** | 100% | **80.0%** | $0.92\times$ |
| **1500 tokens** | 1180 | 1647 | 293 | **$5.65\times$** | **82.3%** | 100% | **20.0%** | $0.42\times$ |

> ⚠️ **Key Latency Takeaway (RQ4):** Contrary to theoretical expectations, fewer prompt tokens did not translate to lower end-to-end wall-clock latency in our serving stack ($t_A / t_B = 0.65\times$ mean speedup; image was slower in 13 of 15 runs). The computational overhead of vision encoder patch extraction and raster processing dominated at these context scales.

---

## 📐 Operational Decision Matrix (When to use Text-as-Image?)

Based on our empirical regression model, we partition document lengths into five actionable operational zones:

| Zone | Word Range ($W$) | Target CR | Retrieval Acc. | Practical Guideline |
| :--- | :---: | :---: | :---: | :--- |
| **Zone I: Negative Economy** | $< 181$ words | $< 1.0\times$ | 100% | **Use Raw Text.** Image rendering introduces a net token penalty. |
| **Zone II: Marginal Gain** | $181 - 397$ words | $1.0 - 2.0\times$ | High | **Use Raw Text.** Token savings ($<50\%$) fail to offset vision encoding latency. |
| **Zone III: The Sweet Spot** | **$397 - 434$ words** | **$2.0 - 2.2\times$** | **100%** | **Optimal for Text-as-Image.** $\ge 50\%$ prompt savings with zero accuracy loss. |
| **Zone IV: Extended Zone** | $434 - 768$ words | $2.2 - 3.7\times$ | $\approx 80\%$ | **Viable with Caution.** Suitable for approximate retrieval or multi-candidate matching. |
| **Zone V: Degradation Cliff** | $> 768$ words | $> 3.7\times$ | $\le 20\%$ | **Avoid Optical Compression.** Dense packing overwhelms OCR patch resolution. |

---

## 🔬 Rasterizer Canvas Configurations

The text-to-image engine standardizes rendering into 5 discrete canvas budget geometries:

| Visual Budget ($k$) | Canvas Width | Font Size | Line Spacing | Wrap Width | Outer Padding |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **70** | $400 - 450\text{ px}$ | $14\text{ pt}$ | $4\text{ px}$ | $55 - 60\text{ chars}$ | $20\text{ px}$ |
| **140** | $600 - 650\text{ px}$ | $16\text{ pt}$ | $5\text{ px}$ | $70 - 72\text{ chars}$ | $25\text{ px}$ |
| **280** | $800 - 850\text{ px}$ | $18\text{ pt}$ | $6\text{ px}$ | $80 - 85\text{ chars}$ | $30\text{ px}$ |
| **560** | $1100\text{ px}$ | $20\text{ pt}$ | $7\text{ px}$ | $95\text{ chars}$ | $35\text{ px}$ |
| **1120** | $1400\text{ px}$ | $22\text{ pt}$ | $8\text{ px}$ | $115\text{ chars}$ | $40\text{ px}$ |

<p align="center">
  <img src="paper/figures/budget_70_rendered.png" alt="k=70 Canvas" width="40%"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="paper/figures/budget_1120_rendered.png" alt="k=1120 Canvas" width="52%"/>
</p>
<p align="center"><em>Visual comparison: 500-token context rendered under k=70 (left, narrow) vs. k=1120 (right, wide high-res).</em></p>

---

## 🛠️ Repository Architecture

```bash
DIP-project/
├── paper/                                  # Research publication artifacts
│   ├── main.tex                            # Complete IEEE-format conference paper LaTeX source
│   ├── main.pdf                            # Compiled research publication PDF
│   └── figures/                            # High-resolution benchmark plots and charts
├── src/                                    # Modular experiment codebase
│   ├── datasets/
│   │   └── ruler_loader.py                 # RULER synthetic Needle-in-a-Haystack generator
│   ├── renderer/
│   │   ├── text_normalizer.py              # NFKC normalization and whitespace cleanup
│   │   └── rasterizer.py                   # Canvas layout engine with discrete token budgets
│   └── evaluation/
│       ├── metrics.py                      # CR, exact match substring accuracy, latency speedup
│       ├── runner.py                       # Single-condition Ollama API runner
│       ├── grid_runner.py                  # Full 3x5 grid evaluation pipeline
│       └── plot_grid.py                    # Scientific plotting script for dashboards & heatmaps
├── results/                                # Benchmark CSV outputs and generated dashboards
│   ├── full_grid_benchmark_results.csv     # Raw 15-run grid evaluation log
│   ├── threshold_analysis.png              # 4-panel mathematical threshold dashboard
│   ├── plot_1_heatmap.png                  # Grid accuracy heatmap
│   ├── plot_2_degradation_cliff.png        # Degradation cliff plot
│   ├── plot_3_token_savings.png            # Token reduction bar chart
│   └── plot_4_budgets_curves.png           # Multi-budget trajectory curves
├── Full_experiment_Gemma4_RULER.ipynb      # Interactive end-to-end research notebook
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed and [Ollama](https://ollama.com/) running locally with the Gemma 4 vision model pulled:

```bash
ollama pull gemma4:31b
ollama serve
```

### 2. Environment Setup
Clone the repository and install the dependencies:

```bash
git clone https://github.com/iamahnaf/Testing-dip.git
cd Testing-dip

# Create virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
# source .venv/bin/activate

# Install required packages
pip install pillow pandas matplotlib ollama requests
```

### 3. Run the Text-to-Image Rasterizer
You can test the rasterizer on any custom string or document:

```python
from src.renderer.rasterizer import TextRasterizer

rasterizer = TextRasterizer()
document_text = "Special authorization key CIPHER-8562-SIGMA was recorded in emergency protocol ledger section 9."

# Render to PIL Image at budget k=560 (1100px width, 20pt font)
image = rasterizer.render(text=document_text, token_budget=560)
image.save("rendered_doc.png")
```

### 4. Run the Full Benchmark Grid
Execute the end-to-end evaluation across all context lengths and visual token budgets:

```bash
# Open and run the interactive Jupyter notebook:
jupyter notebook Full_experiment_Gemma4_RULER.ipynb
```

Or execute grid evaluation modularly:

```python
from src.evaluation.grid_runner import run_full_grid
from src.evaluation.plot_grid import generate_all_plots

# Runs 3 lengths (500, 1000, 1500) x 5 canvas budgets (70, 140, 280, 560, 1120)
results_df = run_full_grid(model_name="gemma4:31b")
generate_all_plots("results/full_grid_benchmark_results.csv")
```

---

## 📝 Citation & Academic Work

If you find this research pipeline, benchmark grid, or dataset generator helpful in your work, please cite our conference paper:

```bibtex
@inproceedings{tahmid2026opticalcontext,
  title={Text-as-Image Compression for Vision-Language Models: How Far Can Rendered Text Replace Text Tokens in Long-Context Retrieval?},
  author={Ahnaf Tahmid and Quazi Nazib Zaman and Fouzia Jahan Dhola},
  booktitle={IEEE Conference Proceedings},
  year={2026},
  organization={United International University}
}
```

---

## 👥 Contributors & Contact

* **Ahnaf Tahmid** - *Dept. of CSE, United International University* ([mtahmid2310203@bscse.uiu.ac.bd](mailto:mtahmid2310203@bscse.uiu.ac.bd))
* **Quazi Nazib Zaman** - *Dept. of CSE, United International University* ([qzaman2320230@bscse.uiu.ac.bd](mailto:qzaman2320230@bscse.uiu.ac.bd))
* **Fouzia Jahan Dhola** - *Dept. of CSE, United International University* ([fdhola2320182@bscse.uiu.ac.bd](mailto:fdhola2320182@bscse.uiu.ac.bd))

---
*Developed as part of the Digital Image Processing & Vision-Language Research Initiative.*
