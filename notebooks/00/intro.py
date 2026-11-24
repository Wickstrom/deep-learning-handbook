# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Marimo version of Lecture 1: Introduction.
#
# The notebook is the slide deck. Each cell is one slide. Run it locally with
#
#     uv run marimo edit notebooks/00/intro.py
#
# or export it to a self-contained WebAssembly page with
#
#     uv run marimo export html-wasm notebooks/00/intro.py -o build/00/intro \
#         --mode run --no-show-code

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/intro.slides.json",
)

with app.setup:
    import os
    from pathlib import Path

    # Make `media/...` resolve no matter which directory marimo was started
    # from. In the WASM export `__file__` may be unset, in which case the page
    # already sits next to its `media/` directory.
    if "__file__" in globals() and __file__:
        try:
            os.chdir(Path(__file__).resolve().parent)
        except OSError:
            pass

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    ACCENT = "#2563eb"
    MUTED = "#94a3b8"
    WARN = "#d97706"
    GREEN = "#059669"

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "figure.dpi": 110,
        }
    )

    def footer(n: int, total: int = 19) -> str:
        """Small slide counter, appended to every markdown slide."""
        return (
            '<div style="position:fixed;bottom:10px;right:18px;font-size:12px;'
            'color:#9ca3af;font-family:system-ui,sans-serif;">'
            f"{n} / {total}</div>"
        )


@app.cell
def _():
    mo.md(
        r"""
    <style>
    .katex { font-size: 1.15em; }
    .katex-display .katex { font-size: 1.3em; }
    </style>

    # Deep Learning

    ### Lecture 1 — Introduction

    **FYS-3033 · UiT The Arctic University of Norway**

    <br>

    These notebooks are the lecture slides. In the exported deck use
    <kbd>←</kbd> / <kbd>→</kbd> to move between slides; run
    `marimo edit` locally to change the code and the figures.

    <div style="margin-top:2.2rem;padding:10px 14px;border-left:4px solid #2563eb;background:#eff6ff;font-size:0.92rem;">
    This repository currently covers the first two lectures only. Exercise
    notebooks are intentionally not included.
    </div>
    """
        + footer(1)
    )
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                r"""
    ## Deep learning is everywhere

    - **Computer vision** — recognising, locating and segmenting objects.
    - **Language and speech** — transcription, translation, assistants.
    - **Reinforcement learning** — games, robotics and control.
    - **Science and medicine** — imaging, proteins, weather, materials.

    The same recipe shows up again and again:

    > a flexible model with many parameters,
    > trained on data by gradient descent,
    > on fast parallel hardware.

    The results are often surprising: one architecture family can caption an
    image, translate a sentence and beat a world champion at Go.
    """
                + footer(2)
            ),
            mo.vstack(
                [
                    mo.image(src="media/dl_everywhere.png", width="100%"),
                    mo.md(
                        "<div style='font-size:0.75rem;color:#6b7280;'>"
                        "Image captioning: a vision network encodes the image and a "
                        "language model writes a description.</div>"
                    ),
                ],
                gap=1,
            ),
        ],
        widths=[1, 1.15],
        gap=2.5,
        align="center",
    )
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                r"""
    ## Computer vision

    - **Classification** — *what is in this image?*
    - **Object detection** — *what is where?* Bounding boxes and labels.
    - **Semantic / instance segmentation** — a class for every pixel.
    - **Video** — tracking, action recognition, summarisation.
    - **Captioning** — images to sentences.
    - **Generation** — super-resolution, editing, deepfakes.

    A single autonomous-driving frame exercises most of the list at once:
    lane markings, cars, pedestrians, signs and cyclists, all under real-time
    constraints.
    """
                + footer(3)
            ),
            mo.vstack(
                [
                    mo.image(src="media/self_driving.png", width="100%"),
                    mo.md(
                        "<div style='font-size:0.75rem;color:#6b7280;'>"
                        "Detections overlaid on a driving scene (course slides)."
                        "</div>"
                    ),
                ],
                gap=1,
            ),
        ],
        widths=[1, 1.1],
        gap=2.5,
        align="center",
    )
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                r"""
    ## Language and speech

    - **Speech recognition** — audio to text, now the default on phones.
    - **Machine translation** — one sequence into another.
    - **Conversational models** — dialogue, summarisation, code.
    - **Large language models** — one model, many tasks, prompted in natural
      language.

    The gains were not subtle. On a standard speech-recognition benchmark
    (Dahl et al., 2012):

    | System | RT03S | FSH |
    |---|---:|---:|
    | Hand-crafted features | 27.4 % | 23.6 % |
    | Deep neural network | 18.5 % | 16.1 % |

    Word error rate, lower is better. A roughly one-third relative reduction
    from swapping the front end for a learned one.
    """
                + footer(4)
            ),
            mo.md(
                r"""
    ### Why sequence models matter

    - Speech and text are **sequences**: the order carries information.
    - The same input can have many valid outputs (many translations are
      correct).
    - Later in the course: recurrent networks, attention and transformers.

    <div style="margin-top:1rem;padding:10px 14px;border-left:4px solid #059669;background:#ecfdf5;font-size:0.9rem;">
    A recurring theme: replace hand-designed representations with learned ones,
    and let the loss decide what is useful.
    </div>
    """
            ),
        ],
        widths=[1.25, 1],
        gap=2.5,
        align="center",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Reinforcement learning and science

    - **Games** — AlphaGo, Atari, StarCraft. Learn a policy from reward.
    - **Robotics** — perception plus control, often learned end to end.
    - **Scientific discovery** — protein structure, weather, particle physics,
      materials.
    - **Medicine** — mammography screening, tumour detection, segmentation.

    <br>

    ### What these have in common

    - A **parametric model** with millions (or billions) of parameters.
    - A **differentiable objective** that measures how wrong the model is.
    - **Gradient-based optimisation** on lots of data and compute.
    - **Representations** that are learned rather than designed by hand.
    """
        + footer(5)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Before deep learning: hand-crafted features

    For decades, computer vision meant **designing features** and then training
    a shallow classifier (often an SVM) on top:

    | Feature family | Core idea | Weakness |
    |---|---|---|
    | SIFT | Local gradient histograms at keypoints | Fails on texture-poor or blurry images |
    | HOG | Histogram of oriented gradients on a grid | Rigid, sensitive to pose |
    | Textons | Quantised local texture responses | Limited vocabulary |
    | GIST | Coarse global scene statistics | Ignores small objects |

    <br>

    The pipeline looked like this:

    ```
    image  ->  hand-designed features  ->  classifier  ->  label
    ```

    Every arrow was engineered by a human, and every design choice was tuned
    for a particular dataset. **The features did not improve when you gave them
    more data.**
    """
        + footer(6)
    )
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                r"""
    ## ImageNet (2010)

    - Images collected from the internet.
    - Labels from Amazon Mechanical Turk.
    - **1.2 million training images**, 1 000 classes.
    - An annual challenge: the ImageNet Large Scale Visual Recognition
      Challenge (ILSVRC).

    <br>

    This was the first dataset large and diverse enough that **representation
    learning** could show its advantage over hand-crafted features. The
    benchmark made progress measurable, and the measurement changed the field.
    """
                + footer(7)
            ),
            mo.vstack(
                [
                    mo.image(src="media/imagenet.png", width="100%"),
                    mo.md(
                        "<div style='font-size:0.75rem;color:#6b7280;'>"
                        "A sample of ImageNet classes (Karpathy, Stanford).</div>"
                    ),
                ],
                gap=1,
            ),
        ],
        widths=[1, 1.1],
        gap=2.5,
        align="center",
    )
    return


@app.cell
def _():
    def imagenet_chart():
        models = [
            "2010\nshallow",
            "2011\nshallow",
            "2012\nAlexNet (8)",
            "2013\nZFNet (8)",
            "2014\nVGG (19)",
            "2014\nGoogLeNet (22)",
            "2015\nResNet (152)",
        ]
        errors = [28.2, 25.8, 16.4, 11.7, 7.3, 6.7, 3.57]
        colors = [MUTED, MUTED, ACCENT, ACCENT, ACCENT, ACCENT, WARN]

        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        bars = ax.bar(range(len(errors)), errors, color=colors, width=0.68)
        for bar, value in zip(bars, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.7,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#111827",
            )
        ax.set_xticks(range(len(models)), models, fontsize=9)
        ax.set_ylabel("top-5 error (%)")
        ax.set_ylim(0, 33)
        ax.set_title("ImageNet classification: five years from hand-crafted to deep")
        ax.annotate(
            "deep networks",
            xy=(3.0, 27.5),
            xytext=(4.4, 30.0),
            fontsize=9,
            color="#374151",
            arrowprops=dict(arrowstyle="->", color="#9ca3af"),
        )
        ax.axhline(25.8, color="#e5e7eb", linewidth=0.8, zorder=0)
        fig.text(0.99, 0.01, "8 / 19", ha="right", fontsize=9, color="#9ca3af")
        fig.tight_layout()
        return fig
    mo.as_html(imagenet_chart())
    return

@app.cell
def _(mo):
    mo.md(
        r"""
    ## The ImageNet moment

    - **2012:** AlexNet cuts the top-5 error from 25.8 % to 16.4 %.
    - **2014:** GoogLeNet reaches 6.7 % with 22 layers.
    - **2015:** ResNet reaches 3.57 % with 152 layers.
    - **Depth and data scale together** — and so does the compute.

    <br>

    The interesting part is not the leaderboard. It is that the winning
    systems learned **general-purpose features** that transferred to other
    tasks: detection, segmentation, medical imaging, remote sensing. A network
    trained on one million images turned out to be a useful starting point for
    almost any vision problem.
    """
        + footer(9)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## What happened?

    Three things changed at once, and they reinforced each other:

    1. **Data.** Labelled datasets became large and diverse (ImageNet, then
       much more).
    2. **Compute.** GPUs made the matrix algebra of neural networks cheap;
       training moved from weeks to hours.
    3. **Algorithms.** ReLU, dropout, batch normalisation, better
       initialisation and optimisers made deep networks trainable.

    <br>

    A fourth ingredient arrived later and matters just as much:

    4. **Software.** Automatic differentiation and deep-learning frameworks
       made experiments reproducible and fast to write.

    <div style="margin-top:0.8rem;padding:10px 14px;border-left:4px solid #d97706;background:#fffbeb;font-size:0.9rem;">
    No single trick caused the breakthrough. The whole stack improved together,
    which is why progress looked sudden from the outside.
    </div>
    """
        + footer(10)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## From labelled to unlabelled data

    Labels are expensive. Raw data is not.

    ```
    pretrain on unlabelled data  ->  adapt to a task  ->  deploy
       (self-supervised)              (few labels)
    ```

    - **Self-supervised learning** — hide part of the input and predict it:
      masked words, masked image patches, next token.
    - **Transfer learning** — fine-tune a pretrained network on a small
      labelled dataset.
    - **Zero-shot and few-shot** — describe the task in text and let a large
      model generalise without new labels.

    <br>

    This is the dominant paradigm today: **pretrain once, adapt many times.**
    The frontier has moved from labelled classification to multimodal models
    trained on images, text, audio and video together.
    """
        + footer(11)
    )
    return


@app.cell
def _():
    def hierarchy_chart():
        stages = [
            ("pixels", "raw intensities", "#e5e7eb"),
            ("edges", "oriented gradients", "#bfdbfe"),
            ("textures", "repeated patterns", "#93c5fd"),
            ("parts", "eyes, wheels, leaves", "#60a5fa"),
            ("objects", "cats, cars, faces", "#2563eb"),
        ]
        fig, ax = plt.subplots(figsize=(9.2, 1.9))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.axis("off")
        for i, (name, sub, color) in enumerate(stages):
            x = 0.15 + i * 2.0
            box = plt.Rectangle(
                (x, 0.55), 1.55, 0.9, facecolor=color, edgecolor="none", alpha=0.9
            )
            ax.add_patch(box)
            ax.text(
                x + 0.775,
                1.0,
                name,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if i >= 3 else "#1f2937",
            )
            ax.text(
                x + 0.775, 0.28, sub, ha="center", va="center", fontsize=8, color="#4b5563"
            )
            if i < len(stages) - 1:
                ax.annotate(
                    "",
                    xy=(x + 1.95, 1.0),
                    xytext=(x + 1.7, 1.0),
                    arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.2),
                )
        ax.text(
            9.85,
            1.75,
            "increasing abstraction  →",
            ha="right",
            va="center",
            fontsize=9,
            color="#6b7280",
        )
        fig.text(0.99, 0.01, "12 / 19", ha="right", fontsize=9, color="#9ca3af")
        fig.tight_layout(pad=0.2)
        return fig
    mo.as_html(hierarchy_chart())
    return

@app.cell
def _(mo):
    mo.md(
        r"""
    ## Why does deep learning work?

    A deep network composes simple transformations:

    $$x \;\longrightarrow\; h_1 \;\longrightarrow\; h_2 \;\longrightarrow\; \cdots \;\longrightarrow\; \hat{y}$$

    - Early layers detect **edges and colours**.
    - Middle layers combine them into **textures and parts**.
    - Late layers represent **objects and concepts**.

    <br>

    Two consequences:

    - **Reuse.** A feature learned for one task is useful for many others, so
      depth gives an exponential saving in parameters compared with a shallow
      network that must memorise every combination.
    - **Composition.** Complex functions are built from simple ones, which is
      exactly what the chain rule is good at differentiating.

    This is the *representation learning* view: the network learns the features,
    not just the classifier on top of them.
    """
        + footer(13)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Learned features are visualisable

    - Filters in the first layers look like **Gabor filters**: oriented edges
      at different scales.
    - Deeper filters respond to **textures, object parts and whole objects**.
    - The same architecture trained on different data learns different
      features.
    - Visualisation methods (saliency, Grad-CAM, feature inversion) let us
      inspect what a network is looking at.

    <br>

    But be careful: a pretty heatmap is **not** an explanation. Visualisation
    is evidence, not proof. Later in the course we look at visualisation and
    uncertainty quantification more rigorously.

    <div style="margin-top:1rem;padding:10px 14px;border-left:4px solid #6b7280;background:#f9fafb;font-size:0.9rem;">
    "Why does it work?" is still an open research question. Depth, data and
    optimisation interact in ways we do not fully understand.
    </div>
    """
        + footer(14)
    )
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                r"""
    ## Neural art and generative models

    The same learned features can be used to **create** images, not only to
    classify them.

    - **Style transfer** (Gatys et al., 2015): match the content of one image
      to the style statistics of another.
    - **DeepDream**: amplify whatever the network detects.
    - **Diffusion models**: learn to reverse a noising process and sample new
      images.

    <br>

    Recognition and generation are two sides of the same coin: both depend on a
    representation rich enough to capture what matters in the data.
    """
                + footer(15)
            ),
            mo.vstack(
                [
                    mo.image(src="media/neural_art.png", width="100%"),
                    mo.md(
                        "<div style='font-size:0.75rem;color:#6b7280;'>"
                        "Style transfer applied to a building (course slides)."
                        "</div>"
                    ),
                ],
                gap=1,
            ),
        ],
        widths=[1, 1.1],
        gap=2.5,
        align="center",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The not-so-good part

    - **Black box.** A decision can be hard to explain to a user, a regulator
      or a doctor.
    - **Unreliable.** Performance degrades under distribution shift, and small
      adversarial perturbations can flip a prediction.
    - **Overconfident.** A softmax score is not a calibrated probability.
    - **Expensive.** Training and inference consume energy, data and hardware.
    - **Need for theory.** Empirical progress has outpaced our understanding.

    <br>

    A well-known failure mode: an object that is rare in training data, seen
    from an unusual angle, can be confidently misclassified. The system has no
    way to say *"I have not seen this before"* unless we build that in.

    <div style="margin-top:1rem;padding:10px 14px;border-left:4px solid #dc2626;background:#fef2f2;font-size:0.9rem;">
    Part of this course is about closing that gap: uncertainty, robustness,
    evaluation and the theory behind the tricks that work.
    </div>
    """
        + footer(16)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Course overview

    **Aim**

    - Introduce the techniques that underlie modern deep learning.
    - Give practical experience: implement the pieces, not just call them.
    - Build theoretical understanding and intuition.
    - Cover recent concepts, not only the classical ones.

    **Curriculum**

    - Optional textbook: *Deep Learning*, Goodfellow, Bengio & Courville
      (2016), free at [deeplearningbook.org](https://www.deeplearningbook.org).
    - Recent papers for the topics where the book is out of date.
    - These notebooks.

    **Roadmap**

    | Block | Topics |
    |---|---|
    | Foundations | Neurons, backpropagation, convolutional networks |
    | Training | Initialisation, optimisation, generalisation |
    | Sequence models | RNNs, attention, transformers |
    | Generative models | Autoencoders, VAEs, GANs, diffusion |
    | Analysis | Visualisation, uncertainty, robustness |
    """
        + footer(17)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Practical information

    **Prerequisites**

    - Calculus, linear algebra and basic statistics.
    - Image analysis, machine learning or pattern recognition.
    - Programming (Python).

    **Assessment**

    | Component | When | Notes |
    |---|---|---|
    | Exercises | Weekly | Formative, not graded |
    | Mandatory assignment | Weeks 8–9 | No deep-learning frameworks |
    | Home exam | Weeks 13–17 | Frameworks allowed |
    | Oral exam | Weeks 20–21 | Discussion of the curriculum |

    <br>

    The mandatory assignment asks you to implement layers and gradients
    yourself. That is deliberate: once you have written backpropagation by
    hand, the frameworks stop being magic.
    """
        + footer(18)
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Next lecture

    ### Neural networks

    - The artificial neuron and its activation function.
    - Stacking neurons into layers and networks.
    - Softmax and the cross-entropy loss.
    - Training by gradient descent.
    - Backpropagation, from the chain rule to a working algorithm.

    <br>

    *Reading:* Goodfellow, Bengio & Courville, Chapter 6.
    """
        + footer(19)
    )
    return


if __name__ == "__main__":
    app.run()
