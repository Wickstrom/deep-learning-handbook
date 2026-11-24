# Deep Learning

Lecture notebooks for the course **FYS-3033 Deep Learning** at UiT The Arctic
University of Norway.

The notebooks are the lecture slides. They are written as
[Marimo](https://marimo.io) apps: the text is organised as slides, and the
figures are live Python — matrices, gradients and training loops you can inspect
and change.

> **🚧 Under construction.** This repository currently covers the first two
> lectures. Exercise notebooks and the remaining lectures are not written yet,
> and the material is still being revised — expect changes.

## 📑 Content

### 1. Introduction to deep learning

- What deep learning is, and where it is used
- A short history: hand-crafted features, ImageNet, and the deep learning moment
- Why deep learning works: data, compute, algorithms and learned representations
- From labelled to unlabelled data
- The not-so-good part: black boxes, unreliability and the need for theory
- Course overview and practical information

[![Notebook](https://img.shields.io/badge/-Notebook-blue?logo=marimo&style=flat&labelColor=gray)](notebooks/00/intro.py)

### 2. Neural networks

- The supervised learning setup
- The artificial neuron and activation functions
- Stacking neurons: from linear boundaries to hidden representations
- Multi-layer networks, softmax and cross-entropy
- Universal approximation
- Training by gradient descent
- Backpropagation through a computation graph and through a layer
- Putting it together: a NumPy MLP trained on a two-arm spiral

[![Notebook](https://img.shields.io/badge/-Notebook-blue?logo=marimo&style=flat&labelColor=gray)](notebooks/01/neural_networks.py)

## 💻 Run the notebooks locally

The repository is packaged with [uv](https://docs.astral.sh/uv/), a single
static binary that installs Python and the project dependencies.

Install uv once, then from the repository root:

```bash
uv sync
uv run marimo edit notebooks/00/intro.py
uv run marimo edit notebooks/01/neural_networks.py
```

`uv sync` reads `pyproject.toml` and creates a project-local virtualenv with
`marimo`, `numpy` and `matplotlib`. Each notebook also declares its own
dependencies in a PEP 723 header, so running a single notebook needs no prior
`uv sync`.

## 🎞 Build the slide decks

The notebooks can be exported as self-contained WebAssembly pages:

```bash
./scripts/build_decks.sh
python3 -m http.server --directory build 8000
```

Then open <http://localhost:8000>. The build script writes the decks to
`build/00/intro/` and `build/01/neural_networks/`, copies the figures, and adds
`index.html` as a landing page.

A GitHub Actions workflow (`.github/workflows/build-decks.yml`) builds the same
artifacts on every push, publishes them to GitHub Pages, and uploads them as a
downloadable archive.

## 🌐 Online slides

The decks are deployed to GitHub Pages:

**<https://wickstrom.github.io/deep-learning-handbook/>**

> **Note on privacy.** The repository is private, but GitHub Pages sites are
> publicly readable. Links to the deployed slides can be shared with anyone,
> and search engines may index them. Keep that in mind before the material is
> ready for students outside the course.

## 🗂 Repository layout

```
notebooks/
  00/intro.py                  # Lecture 1: Introduction
  01/neural_networks.py        # Lecture 2: Neural networks
  */layouts/*.slides.json      # Marimo slide layout
  */media/                     # Figures used by the notebooks
slide_pdfs/                    # Original lecture slides (PDF)
scripts/build_decks.sh         # Export both decks to build/
index.html                     # Landing page for the exported decks
```

## 📚 Reading

The course follows the optional textbook *Deep Learning* by Goodfellow, Bengio
and Courville (MIT Press, 2016), available free at
[deeplearningbook.org](https://www.deeplearningbook.org).

- Lecture 1: no required reading; the lecture is an overview.
- Lecture 2: Chapter 6, *Deep Feedforward Networks*.
- Next lecture: Chapter 9, *Convolutional Networks*.

## 🙏 Acknowledgements

The structure of this repository and its slide workflow are inspired by the
[Pattern Recognition handbook](https://github.com/Wickstrom/pattern-recognition-handbook)
(FYS-3012) and the
[Time Series Analysis with Python handbook](https://github.com/FilippoMB/python-time-series-handbook).

---

Course material for FYS-3033. Please do not redistribute without permission.
