# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# Marimo version of Lecture 2: Neural Networks.
#
# The notebook is the slide deck. Each cell is one slide. Interactive figures
# use tabs with precomputed settings: marimo does not allow reading a widget's
# value in the cell that creates it, and tabs keep each demo on one slide.
#
#     uv run marimo edit notebooks/01/neural_networks.py
#
#     uv run marimo export html-wasm notebooks/01/neural_networks.py \
#         -o build/01/neural_networks --mode run --no-show-code

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/neural_networks.slides.json",
)

with app.setup:
    import os
    from pathlib import Path

    if "__file__" in globals() and __file__:
        try:
            os.chdir(Path(__file__).resolve().parent)
        except OSError:
            pass

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    ACCENT = "#2563eb"
    MUTED = "#94a3b8"
    WARN = "#d97706"
    GREEN = "#059669"
    RED = "#dc2626"

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

    def footer(n: int, total: int = 20) -> str:
        return (
            '<div style="position:fixed;bottom:10px;right:18px;font-size:12px;'
            'color:#9ca3af;font-family:system-ui,sans-serif;">'
            f"{n} / {total}</div>"
        )

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def softmax(z):
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)

    ACTS = {
        "linear": (lambda z: z, lambda z: np.ones_like(z)),
        "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1.0 - sigmoid(z))),
        "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
        "ReLU": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0).astype(float)),
        "leaky ReLU": (
            lambda z: np.where(z > 0, z, 0.05 * z),
            lambda z: np.where(z > 0, 1.0, 0.05),
        ),
    }

    def make_spiral(n=180, noise=0.12, seed=0):
        rng = np.random.default_rng(seed)
        xs, ys = [], []
        for k in range(2):
            t = np.linspace(0.35, 3.0, n)
            angle = 2.0 * t + k * np.pi
            xs.append(
                np.c_[t * np.cos(angle), t * np.sin(angle)]
                + rng.normal(0, noise, (n, 2))
            )
            ys.append(np.full(n, k))
        data = np.vstack(xs)
        labels = np.concatenate(ys)
        return (data - data.mean(0)) / data.std(0), labels


@app.cell
def _():
    mo.md(
        r"""
    <style>
    .katex { font-size: 1.15em; }
    .katex-display .katex { font-size: 1.3em; }
    </style>

    # Neural Networks

    ### Lecture 2 — Neurons, layers, losses and backpropagation

    **FYS-3033 · UiT The Arctic University of Norway**

    <br>

    Today we build a neural network from the inside out:

    neuron → activation → layer → network → loss → gradient → backpropagation.

    <div style="margin-top:1.6rem;padding:10px 14px;border-left:4px solid #2563eb;background:#eff6ff;font-size:0.9rem;">
    Several figures on the following slides are interactive: switch the tabs
    under a figure to compare settings. Run <code>marimo edit</code> on this
    notebook to see the code behind each one.
    </div>

    <div style="margin-top:0.8rem;padding:10px 14px;border-left:4px solid #d97706;background:#fffbeb;font-size:0.9rem;">
    🚧 <strong>Under construction</strong> — only the first two lectures are
    written so far. Exercises and the remaining lectures are on the way, and
    these slides are still being revised.
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
    ## Supervised learning

    Given a dataset of input–label pairs

    $$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}, \qquad x_i \in \mathbb{R}^{d},$$

    we want a parametric model $f_\theta$ that predicts $y$ from $x$.

    <br>

    | Ingredient | What it is |
    |---|---|
    | **Data** | Examples and their labels |
    | **Model** | $f_\theta$, with parameters $\theta$ |
    | **Loss** | $\mathcal{L}(\hat{y}, y)$, a measure of error |
    | **Optimiser** | A rule for updating $\theta$ |

    <br>

    Training means solving

    $$\theta^\star = \arg\min_\theta \; \frac{1}{N}\sum_{i=1}^{N}
      \mathcal{L}\big(f_\theta(x_i), y_i\big).$$
    """
                + footer(2)
            ),
            mo.md(
                r"""
    ### Two kinds of problem

    **Regression** — the label is continuous.

    - Predict a temperature, a depth map, a pixel value.
    - Typical loss: squared error.

    **Classification** — the label is a category.

    - Predict *cat* vs *dog*, or one of 1 000 ImageNet classes.
    - Typical loss: cross-entropy.

    <br>

    Everything in this lecture applies to both. The differences are the
    **output activation** and the **loss**, which have to match each other.
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
    def neuron_slide():
        presets = {
            "boundary through the origin": (1.0, 1.0, 0.0),
            "rotated and shifted": (2.0, -0.5, -1.0),
            "steep, far from the origin": (-1.5, 1.0, 2.0),
        }
        xx, yy = np.meshgrid(np.linspace(-3, 3, 260), np.linspace(-3, 3, 260))
        tabs = {}
        for name, (w1, w2, b) in presets.items():
            zz = w1 * xx + w2 * yy + b
            pp = sigmoid(zz)

            fig, ax = plt.subplots(figsize=(5.6, 4.2))
            cf = ax.contourf(
                xx, yy, pp, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.85
            )
            ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=2)
            ax.axhline(0, color="white", linewidth=0.6, alpha=0.6)
            ax.axvline(0, color="white", linewidth=0.6, alpha=0.6)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_title(
                rf"$z = {w1:.1f}x_1 + {w2:.1f}x_2 + {b:.1f}$"
                "\nblack line: $z = 0$"
            )
            fig.colorbar(cf, ax=ax, shrink=0.82, label="$P(y=1\\,|\\,x)$")
            fig.tight_layout()
            tabs[name] = mo.as_html(fig)
            plt.close(fig)

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## The artificial neuron

    $$\text{pre-activation: } z = w^\top x + b,
      \qquad \text{activation: } a = \varphi(z)$$

    - $w$ is the **normal vector** of the decision boundary $z = 0$.
    - $b$ shifts the boundary away from the origin.
    - $\varphi$ turns the score into an output; with the sigmoid it is a
      probability.

    Switch tabs to rotate and translate the boundary.
    """
                ),
                mo.ui.tabs(tabs),
                mo.md(footer(3)),
            ],
            gap=1,
            align="start",
        )

    neuron_slide()

@app.cell
def _(mo):
    def activation_slide():
        properties = {
            "linear": "Range $(-\\infty,\\infty)$; derivative $1$. Used only in a regression output.",
            "sigmoid": "Range $(0,1)$; derivative at most $1/4$. Binary output and gates.",
            "tanh": "Range $(-1,1)$; derivative at most $1$. Recurrent networks.",
            "ReLU": "Range $[0,\\infty)$; derivative $0$ or $1$. Hidden layers in CNNs and MLPs.",
            "leaky ReLU": "Range $(-\\infty,\\infty)$; derivative $1$ or $0.05$. Avoids dead units.",
        }
        z = np.linspace(-4, 4, 400)
        tabs = {}
        for name, (fn, dfn) in ACTS.items():
            fig, ax = plt.subplots(figsize=(6.0, 3.2))
            ax.plot(z, fn(z), color=ACCENT, linewidth=2.4, label=r"$\varphi(z)$")
            ax.plot(
                z,
                dfn(z),
                color=WARN,
                linewidth=2.0,
                linestyle="--",
                label=r"$\varphi'(z)$",
            )
            ax.axhline(0, color="#e5e7eb", linewidth=0.8)
            ax.axvline(0, color="#e5e7eb", linewidth=0.8)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-2.2, 4.2)
            ax.set_xlabel("$z$")
            ax.set_title(f"{name}")
            ax.legend(frameon=False, fontsize=9, loc="upper left")
            fig.tight_layout()
            tabs[name] = mo.vstack(
                [mo.md(properties[name]), mo.as_html(fig)], gap=0.5
            )
            plt.close(fig)

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Activation functions

    The activation decides what a neuron can express. Two properties matter
    most:

    - **Range and shape** — does it produce negative values? Is it bounded?
    - **Derivative** — how much gradient survives a layer?

    **Saturation is the enemy.** Sigmoid and tanh flatten out for large $|z|$,
    so their derivatives go to zero. ReLU keeps a derivative of exactly one for
    all positive inputs, which is why it made deep networks trainable.
    """
                ),
                mo.ui.tabs(tabs),
                mo.md(footer(4)),
            ],
            gap=1,
            align="start",
        )

    activation_slide()

@app.cell
def _(mo):
    def linearity_slide():
        return mo.md(
            r"""
    ## Why we need a non-linearity

    Stack two linear layers:

    $$a^{(1)} = W^{(1)}x + b^{(1)}, \qquad
      a^{(2)} = W^{(2)}a^{(1)} + b^{(2)}.$$

    Substitute the first into the second:

    $$a^{(2)} = \underbrace{W^{(2)}W^{(1)}}_{W'}x
      + \underbrace{W^{(2)}b^{(1)} + b^{(2)}}_{b'}.$$

    The result is **another linear function**. No matter how many linear layers
    you stack, the composition collapses to a single matrix.

    <br>

    A non-linear activation between the layers breaks this collapse. That is
    what allows a deep network to represent curved decision boundaries and
    complicated functions.

    <div style="margin-top:0.9rem;padding:10px 14px;border-left:4px solid #d97706;background:#fffbeb;font-size:0.9rem;">
    Depth without non-linearity is a waste of parameters. The non-linearity is
    what makes composition useful.
    </div>
    """
            + footer(5)
        )

    linearity_slide()

@app.cell
def _(mo):
    def xor_slide():
        grid = np.linspace(-0.6, 1.6, 300)
        gx, gy = np.meshgrid(grid, grid)
        xor_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        xor_labels = np.array([0, 1, 1, 0])

        fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2))

        ax = axes[0]
        for label_value, color, marker in [(0, ACCENT, "o"), (1, WARN, "s")]:
            mask = xor_labels == label_value
            ax.scatter(
                xor_points[mask, 0],
                xor_points[mask, 1],
                c=color,
                marker=marker,
                s=110,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
                label=f"class {label_value}",
            )
        ax.set_title("XOR")
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        ax = axes[1]
        zz = gx + gy - 1.0
        ax.contourf(gx, gy, sigmoid(zz), levels=12, cmap="RdBu_r", alpha=0.65)
        ax.contour(gx, gy, zz, levels=[0], colors="black", linewidths=1.6)
        for label_value, color, marker in [(0, ACCENT, "o"), (1, WARN, "s")]:
            mask = xor_labels == label_value
            ax.scatter(
                xor_points[mask, 0],
                xor_points[mask, 1],
                c=color,
                marker=marker,
                s=90,
                edgecolor="white",
                linewidth=1.1,
                zorder=3,
            )
        ax.set_title("one neuron: not separable")
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_aspect("equal")

        ax = axes[2]
        h1 = (gx + gy > 0.5).astype(float)
        h2 = (-gx - gy > -1.5).astype(float)
        out = (h1 + h2 > 1.5).astype(float)
        ax.contourf(gx, gy, out, levels=[-0.5, 0.5, 1.5], colors=["#dbeafe", "#fde68a"])
        ax.contour(gx, gy, h1, levels=[0.5], colors=ACCENT, linewidths=1.2)
        ax.contour(gx, gy, h2, levels=[0.5], colors=WARN, linewidths=1.2)
        for label_value, color, marker in [(0, ACCENT, "o"), (1, WARN, "s")]:
            mask = xor_labels == label_value
            ax.scatter(
                xor_points[mask, 0],
                xor_points[mask, 1],
                c=color,
                marker=marker,
                s=90,
                edgecolor="white",
                linewidth=1.1,
                zorder=3,
            )
        ax.set_title("two hidden units: solved")
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_aspect("equal")
        fig.tight_layout()

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Stacking neurons

    A single neuron draws **one straight line**. XOR cannot be separated by
    one line, so no single neuron can solve it, no matter how long you train.

    Two hidden units, however, can carve the input space into regions:

    - $h_1 = \mathbf{1}[x_1 + x_2 > 0.5]$ (an OR),
    - $h_2 = \mathbf{1}[x_1 + x_2 < 1.5]$ (a NAND),
    - output $= \mathbf{1}[h_1 + h_2 > 1.5]$ (AND of the two).

    The output neuron now only has to separate **two regions**, which is a
    linear problem again. This is the core idea of a hidden layer: change the
    representation so that the final problem becomes easy.
    """
                ),
                mo.as_html(fig),
                mo.md(footer(6)),
            ],
            gap=1,
            align="start",
        )
        plt.close(fig)

    xor_slide()

@app.cell
def _(mo):
    def mlp_slide():
        return mo.md(
            r"""
    ## Multi-layer networks

    Write one layer as an affine map followed by a non-linearity:

    $$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}, \qquad
      a^{(l)} = \varphi\big(z^{(l)}\big), \qquad a^{(0)} = x.$$

    Shapes for a batch of $B$ examples:

    | Quantity | Shape |
    |---|---|
    | $a^{(l-1)}$ | $n_{l-1} \times B$ |
    | $W^{(l)}$ | $n_l \times n_{l-1}$ |
    | $b^{(l)}$ | $n_l \times 1$ (broadcast over the batch) |
    | $z^{(l)}, a^{(l)}$ | $n_l \times B$ |

    <br>

    A **multi-layer perceptron** (MLP) is this composition, with a suitable
    output layer at the end. Parameter count for 784 → 128 → 10:

    $$(784 \cdot 128 + 128) + (128 \cdot 10 + 10) = 101\,770.$$

    The forward pass is a sequence of matrix multiplications, which is exactly
    what a GPU is built to do.
    """
            + footer(7)
        )

    mlp_slide()

@app.cell
def _(mo):
    def softmax_slide():
        presets = {
            "confident cat": (3.0, 0.5, -1.0),
            "uncertain": (0.3, 0.2, 0.0),
            "dog wins": (-1.0, 2.5, 0.5),
            "first three, shifted by +1": (4.0, 1.5, 0.0),
        }
        tabs = {}
        for name, logits in presets.items():
            probs = softmax(np.array(logits))
            fig, ax = plt.subplots(figsize=(5.8, 3.2))
            bars = ax.bar(
                ["cat", "dog", "bird"], probs, color=[ACCENT, GREEN, WARN], width=0.6
            )
            for bar, value in zip(bars, probs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.03,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("probability")
            ax.set_title(
                f"logits = ({logits[0]:.1f}, {logits[1]:.1f}, {logits[2]:.1f})"
                f"  —  sum = {probs.sum():.3f}"
            )
            fig.tight_layout()
            tabs[name] = mo.as_html(fig)
            plt.close(fig)

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Output activation and softmax

    For $C$ classes, the network produces one **logit** $z_c$ per class. The
    softmax turns logits into a categorical distribution:

    $$p_c = \frac{\exp(z_c)}{\sum_{j=1}^{C}\exp(z_j)},
      \qquad p_c > 0, \quad \sum_c p_c = 1.$$

    - It is **shift invariant**: adding the same constant to every logit leaves
      the probabilities unchanged. Compare the first and last tab.
    - It preserves the **ranking**: the most likely class is the one with the
      largest logit.
    - It is **not** a calibrated probability. A softmax score of 0.99 does not
      mean the model is right 99 % of the time.
    """
                ),
                mo.ui.tabs(tabs),
                mo.md(footer(8)),
            ],
            gap=1,
            align="start",
        )

    softmax_slide()

@app.cell
def _(mo):
    def uat_slide():
        rng = np.random.default_rng(0)
        xg = np.linspace(-1, 1, 400)
        target = np.sin(2 * np.pi * xg) + 0.4 * np.cos(5 * np.pi * xg)
        slopes = rng.normal(0, 3.0, 40)
        offsets = rng.uniform(-2, 2, 40)

        tabs = {}
        for n in [2, 4, 8, 20, 40]:
            feats = np.maximum(slopes[:n, None] * xg[None, :] + offsets[:n, None], 0).T
            feats = np.c_[feats, np.ones(len(xg))]
            coeffs, *_ = np.linalg.lstsq(feats, target, rcond=None)
            approx = feats @ coeffs
            mse = np.mean((approx - target) ** 2)

            fig, ax = plt.subplots(figsize=(7.2, 3.4))
            ax.plot(
                xg, target, color="#111827", linewidth=2, linestyle="--", label="target"
            )
            ax.plot(xg, approx, color=ACCENT, linewidth=2.2, label="network")
            ax.fill_between(xg, target, approx, color=ACCENT, alpha=0.12)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1.9, 1.9)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x)$")
            ax.set_title(f"{n} ReLU units — MSE = {mse:.4f}")
            ax.legend(frameon=False, fontsize=9, loc="upper right")
            fig.tight_layout()
            tabs[f"{n} units"] = mo.as_html(fig)
            plt.close(fig)

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Universal approximation

    > A single hidden layer with a linear output unit can approximate any
    > continuous function arbitrarily well, given enough hidden units.
    > — Hornik (1991)

    A hidden unit computes $\varphi(w^\top x + b)$; summing many of them builds
    a flexible basis. The tabs add units and refit only the output weights.

    Two important caveats:

    - The theorem is about **existence**, not about finding the weights by
      gradient descent.
    - The required width can be enormous. **Depth** is usually far more
      parameter-efficient than one very wide layer.
    """
                ),
                mo.ui.tabs(tabs),
                mo.md(footer(9)),
            ],
            gap=1,
            align="start",
        )

    uat_slide()

@app.cell
def _(mo):
    def loss_slide():
        pg = np.linspace(0.01, 0.999, 400)
        fig, ax = plt.subplots(figsize=(6.6, 3.4))
        ax.plot(pg, -np.log(pg), color=ACCENT, linewidth=2.2)
        for pv, label in [(0.05, "confident, wrong"), (0.50, "uncertain"), (0.95, "confident, right")]:
            loss = -np.log(pv)
            ax.scatter([pv], [loss], s=55, color=WARN, zorder=3)
            ax.annotate(
                f"{label}\np = {pv:.2f},  loss = {loss:.2f}",
                xy=(pv, loss),
                xytext=(pv + 0.05, loss + 0.7),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#6b7280", linewidth=0.8),
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5)
        ax.set_xlabel("$p_{y}$ (probability assigned to the true class)")
        ax.set_ylabel(r"$-\log p_{y}$")
        ax.set_title("Cross-entropy loss for one example")
        fig.tight_layout()

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Loss function: cross-entropy

    A classifier outputs $p_\theta(y \mid x)$. The likelihood of the dataset is
    $\prod_i p_\theta(y_i \mid x_i)$, so maximum likelihood means minimising the
    **negative log-likelihood**:

    $$\mathcal{L} = -\log p_\theta(y \mid x),
      \qquad
      J(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i \mid x_i).$$

    For a one-hot label only the true class survives, so this is the
    **cross-entropy** between the label distribution and the prediction.

    Being confidently wrong is expensive: as $p_y \to 0$, the loss goes to
    infinity. That is exactly the behaviour we want from a training signal.
    """
                ),
                mo.as_html(fig),
                mo.md(footer(10)),
            ],
            gap=1,
            align="start",
        )
        plt.close(fig)

    loss_slide()

@app.cell
def _(mo):
    def gd_slide():
        return mo.md(
            r"""
    ## Training by gradient descent

    We cannot solve $\min_\theta J(\theta)$ in closed form for a deep network.
    Instead, start somewhere and repeatedly step **downhill**:

    $$\theta \leftarrow \theta - \eta \,\nabla_\theta J(\theta).$$

    - $\eta > 0$ is the **learning rate**: the step length.
    - $\nabla_\theta J$ points in the direction of steepest **increase**, so we
      move in the opposite direction.
    - One pass over the data is an **epoch**.

    <br>

    | Variant | Gradient computed on |
    |---|---|
    | Batch gradient descent | All $N$ examples |
    | Stochastic gradient descent | One example |
    | Mini-batch (standard) | A random batch of 32–512 |

    Mini-batches trade a little gradient noise for much better hardware
    utilisation. The noise can even help escape bad regions.

    <div style="margin-top:0.8rem;padding:10px 14px;border-left:4px solid #2563eb;background:#eff6ff;font-size:0.9rem;">
    The hard part is not the update rule. It is computing
    $\nabla_\theta J$ for millions of parameters — that is what backpropagation
    does.
    </div>
    """
            + footer(11)
        )

    gd_slide()

@app.cell
def _(mo):
    def gd2d_slide():
        def loss2d(a, b):
            return 0.5 * (a**2 + 4.0 * b**2)

        grid = np.linspace(-3, 3, 240)
        gx, gy = np.meshgrid(grid, grid)
        Z = loss2d(gx, gy)
        tabs = {}

        for lr in [0.02, 0.18, 0.55]:
            path = [np.array([-2.6, 2.0])]
            for _ in range(20):
                point = path[-1]
                grad = np.array([point[0], 4.0 * point[1]])
                path.append(point - lr * grad)
            path = np.array(path)

            fig, ax = plt.subplots(figsize=(6.2, 3.9))
            levels = np.linspace(0, 20, 24)
            ax.contourf(
                gx, gy, np.clip(Z, 0, 20), levels=levels, cmap="Greys", alpha=0.55
            )
            ax.contour(
                gx,
                gy,
                np.clip(Z, 0, 20),
                levels=levels[::4],
                colors="#9ca3af",
                linewidths=0.6,
            )
            visible = path[np.all(np.abs(path) < 50, axis=1)]
            ax.plot(
                visible[:, 0],
                visible[:, 1],
                "-o",
                color=ACCENT,
                markersize=3.5,
                linewidth=1.6,
            )
            ax.scatter([path[0, 0]], [path[0, 1]], s=70, color=WARN, zorder=4)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_xlabel(r"$\theta_1$")
            ax.set_ylabel(r"$\theta_2$")
            diverged = not np.all(np.isfinite(path)) or np.max(np.abs(path)) > 1e3
            final = "diverged" if diverged else f"{loss2d(*path[-1]):.3f}"
            ax.set_title(
                rf"$\eta$ = {lr:.2f}   —   loss after 20 steps: {final}"
            )
            fig.tight_layout()
            label = {0.02: "too small", 0.18: "about right", 0.55: "too large"}[lr]
            tabs[f"η = {lr:.2f}  ({label})"] = mo.as_html(fig)
            plt.close(fig)

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Gradient descent in practice

    The loss here is $J(\theta) = \tfrac12(\theta_1^2 + 4\theta_2^2)$, a long
    narrow valley. Gradients point across the valley, not along it.

    - **Too small** $\eta$: slow progress.
    - **About right** $\eta$: steady descent.
    - **Too large** $\eta$: overshoot, oscillate, and eventually diverge.

    This sensitivity is why later lectures cover momentum, adaptive step sizes
    and learning-rate schedules.
    """
                ),
                mo.ui.tabs(tabs),
                mo.md(footer(12)),
            ],
            gap=1,
            align="start",
        )

    gd2d_slide()

@app.cell
def _(mo):
    def graph_slide():
        fig, ax = plt.subplots(figsize=(9.0, 3.5))
        ax.set_xlim(0, 9.6)
        ax.set_ylim(0, 4.4)
        ax.axis("off")

        def box(x, y, text, face="#eff6ff", edge=ACCENT):
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.78, y - 0.44),
                    1.56,
                    0.88,
                    boxstyle="round,pad=0.06,rounding_size=0.12",
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.4,
                )
            )
            ax.text(x, y, text, ha="center", va="center", fontsize=9.5)

        box(1.0, 3.5, "$x$\n2")
        box(1.0, 1.5, "$y$\n3")
        box(4.0, 2.5, "$q = x+y$\n5")
        box(4.0, 0.4, "$z$\n5")
        box(7.4, 1.45, "$f = qz$\n25", face="#fffbeb", edge=WARN)

        def arrow(p, q, text, tpos, color="#6b7280"):
            ax.annotate(
                "",
                xy=q,
                xytext=p,
                arrowprops=dict(
                    arrowstyle="->", color=color, linewidth=1.3, shrinkA=26, shrinkB=26
                ),
            )
            ax.text(tpos[0], tpos[1], text, fontsize=8.5, color=color, ha="center")

        arrow((1.0, 3.5), (4.0, 2.5), r"$\partial q/\partial x = 1$", (2.4, 3.35))
        arrow((1.0, 1.5), (4.0, 2.5), r"$\partial q/\partial y = 1$", (2.4, 1.65))
        arrow((4.0, 2.5), (7.4, 1.45), r"$\partial f/\partial q = 5$", (5.8, 2.5))
        arrow((4.0, 0.4), (7.4, 1.45), r"$\partial f/\partial z = 5$", (5.8, 0.35))

        ax.text(
            0.2,
            4.15,
            "Forward: compute and store values.",
            fontsize=10,
            fontweight="bold",
            color="#111827",
        )
        ax.text(
            0.2,
            0.0,
            r"Backward: $\partial f/\partial x = 5$, $\partial f/\partial y = 5$, "
            r"$\partial f/\partial z = 5$  (local derivative × upstream gradient)",
            fontsize=9,
            color=RED,
            va="bottom",
        )
        fig.tight_layout()

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Backpropagation: the computation graph

    Every network is a composition of simple operations. Draw it as a graph:

    - **Forward pass:** compute each node from its inputs and keep the values.
    - **Backward pass:** start with $\partial \mathcal{L}/\partial \mathcal{L}=1$
      and walk backwards. At each node, multiply the **upstream gradient** by
      the **local derivative** and pass it to the inputs.

    Here $f = (x+y)z$ with $x=2$, $y=3$, $z=5$. The forward pass gives $q=5$
    and $f=25$. The backward pass gives every partial derivative in one sweep.
    """
                ),
                mo.as_html(fig),
                mo.md(footer(13)),
            ],
            gap=1,
            align="start",
        )
        plt.close(fig)

    graph_slide()

@app.cell
def _(mo):
    def chain_slide():
        return mo.md(
            r"""
    ## The chain rule, done efficiently

    For a scalar composition $f(g(h(x)))$:

    $$\frac{df}{dx} = \frac{df}{dg}\cdot\frac{dg}{dh}\cdot\frac{dh}{dx}.$$

    With several paths from a variable to the loss, the contributions **add**:

    $$\frac{\partial \mathcal{L}}{\partial u}
      = \sum_{\text{paths } u \to \mathcal{L}} \;
        \prod_{\text{edges on the path}} \text{local derivative}.$$

    Enumerating paths explodes: a graph with nine paths needs nine products.
    Backpropagation instead computes **one gradient per node** and reuses it. In
    the example above, nine multiplications and eight additions collapse into
    four multiplications.

    <br>

    ### The vector case

    If $u \in \mathbb{R}^m$ and $v \in \mathbb{R}^n$, the local derivative is a
    Jacobian $J \in \mathbb{R}^{n \times m}$. We never form it. We only need
    the **vector–Jacobian product** (VJP)

    $$\bar{u} = J^\top \bar{v},$$

    which costs about the same as one forward pass through the layer.
    """
            + footer(14)
        )

    chain_slide()

@app.cell
def _(mo):
    def layer_slide():
        return mo.md(
            r"""
    ## Backpropagation through one layer

    Suppose we already know the gradient with respect to the layer output,
    $\bar{a}_{\text{out}} = \partial \mathcal{L}/\partial a_{\text{out}}$.
    For the layer $z = Wa + b$, $a_{\text{out}} = \varphi(z)$:

    $$\bar{z} = \bar{a}_{\text{out}} \odot \varphi'(z)$$

    $$\frac{\partial \mathcal{L}}{\partial W} = \bar{z}\,a^\top,
      \qquad
      \frac{\partial \mathcal{L}}{\partial b} = \bar{z}\mathbf{1}^\top,
      \qquad
      \bar{a} = W^\top \bar{z}.$$

    <br>

    Read the shapes: if $\bar{z}$ is $n_l \times B$ and $a$ is
    $n_{l-1} \times B$, then $\bar{z}a^\top$ is $n_l \times n_{l-1}$ — exactly
    the shape of $W$.

    **One matrix, two directions.** $W$ maps activations forward and gradients
    backward. This is why the backward pass costs roughly the same as the
    forward pass, and why GPUs love it.

    <div style="margin-top:0.8rem;padding:10px 14px;border-left:4px solid #059669;background:#ecfdf5;font-size:0.9rem;">
    In a batch, the bias gradient is summed over the batch dimension (the
    $\mathbf{1}^\top$ above), not averaged.
    </div>
    """
            + footer(15)
        )

    layer_slide()

@app.cell
def _(mo):
    def modules_slide():
        return mo.md(
            r"""
    ## Modules and automatic differentiation

    A layer is a **self-contained module** with two methods:

    ```python
    class Layer:
        def forward(self, x):
            self.x = x              # cache what backward needs
            return ...

        def backward(self, grad):
            self.dW = ...           # accumulate parameter gradients
            self.db = ...
            return grad_wrt_input   # pass the gradient to the child
    ```

    A network is a chain of such modules. Forward propagates values from the
    input to the loss; backward propagates gradients from the loss to the
    input, one module at a time.

    <br>

    **Automatic differentiation** builds this graph for you. PyTorch, TensorFlow
    and JAX all implement some variant of reverse-mode autodiff.

    **Gradient checking** is the unit test for your implementation: compare the
    analytic gradient with a finite difference,

    $$\frac{\partial \mathcal{L}}{\partial \theta_i}
      \approx \frac{\mathcal{L}(\theta + \epsilon e_i)
      - \mathcal{L}(\theta - \epsilon e_i)}{2\epsilon}.$$

    If the relative error is around $10^{-7}$ or smaller, your backward pass is
    probably correct. You will do this in the mandatory assignment.
    """
            + footer(16)
        )

    modules_slide()

@app.cell
def _(mo):
    def training_slide():
        tabs = {}
        for H in [2, 8, 24]:
            X, y = make_spiral()
            N = len(y)
            rng = np.random.default_rng(0)
            W1 = rng.normal(0, np.sqrt(2 / 2), (2, H))
            b1 = np.zeros(H)
            W2 = rng.normal(0, np.sqrt(2 / H), (H, 2))
            b2 = np.zeros(2)
            Y = np.eye(2)[y]
            lam = 1e-4
            losses = []
            for _ in range(1500):
                a1 = np.tanh(X @ W1 + b1)
                p = softmax(a1 @ W2 + b2)
                losses.append(
                    -np.mean(np.log(p[np.arange(N), y] + 1e-12))
                    + 0.5 * lam * (np.sum(W1 * W1) + np.sum(W2 * W2))
                )
                dz2 = (p - Y) / N
                dW2 = a1.T @ dz2 + lam * W2
                db2 = dz2.sum(0)
                dz1 = (dz2 @ W2.T) * (1 - a1**2)
                W1 -= 0.5 * (X.T @ dz1 + lam * W1)
                b1 -= 0.5 * dz1.sum(0)
                W2 -= 0.5 * dW2
                b2 -= 0.5 * db2

            xx, yy = np.meshgrid(np.linspace(-2.4, 2.4, 240), np.linspace(-2.4, 2.4, 240))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            posterior = softmax(np.tanh(grid_points @ W1 + b1) @ W2 + b2)[:, 1]
            posterior = posterior.reshape(xx.shape)
            accuracy = np.mean(p.argmax(1) == y)

            fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
            ax = axes[0]
            ax.contourf(
                xx, yy, posterior, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.65
            )
            ax.contour(xx, yy, posterior, levels=[0.5], colors="black", linewidths=1.3)
            ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, color=ACCENT, alpha=0.8)
            ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, color=WARN, alpha=0.8)
            ax.set_xlim(-2.4, 2.4)
            ax.set_ylim(-2.4, 2.4)
            ax.set_aspect("equal")
            ax.set_title(f"{H} hidden units — accuracy {accuracy:.1%}")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

            ax = axes[1]
            ax.plot(losses, color="#111827", linewidth=1.6)
            ax.set_yscale("log")
            ax.set_xlabel("iteration")
            ax.set_ylabel("loss")
            ax.set_title("training loss")
            fig.tight_layout()
            tabs[f"{H} hidden units"] = mo.as_html(fig)
            plt.close(fig)

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Putting it together: an MLP from scratch

    A 2–$H$–2 network with $\tanh$ hidden units and softmax output, trained on
    a two-arm spiral with full-batch gradient descent (1 500 steps, NumPy only):

    - forward: $z_1 = XW_1 + b_1$, $a_1 = \tanh(z_1)$,
      $z_2 = a_1W_2 + b_2$, $p = \mathrm{softmax}(z_2)$;
    - loss: cross-entropy plus a small L2 penalty;
    - backward: the four equations from the previous slide;
    - update: $\theta \leftarrow \theta - \eta \nabla_\theta J$.

    The spiral is not linearly separable, but a handful of hidden units is
    enough to bend the boundary around both arms. Compare two hidden units:
    the network can no longer represent the curve.
    """
                ),
                mo.ui.tabs(tabs),
                mo.md(footer(17)),
            ],
            gap=1,
            align="start",
        )

    training_slide()

@app.cell
def _(mo):
    def problems_slide():
        return mo.md(
            r"""
    ## Problems with deep networks

    **Optimisation is harder**

    - The loss is non-convex: many local minima and saddle points.
    - Gradients shrink or explode as they travel through many layers.
    - Saturating activations make this worse: the sigmoid derivative is at most
      $1/4$, so ten layers can shrink a gradient by $4^{-10} \approx 10^{-6}$.

    **Overfitting is a problem**

    - Millions of parameters can memorise the training set.
    - Generalisation depends on data, regularisation and inductive bias.

    **Resources**

    - Training needs data, compute and energy; inference has a cost too.

    <br>

    But the benefits are large: learned hierarchical features, transfer to new
    tasks, and models that scale with data and compute. The next lectures are
    about making these benefits actually materialise.
    """
            + footer(18)
        )

    problems_slide()

@app.cell
def _(mo):
    def summary_slide():
        return mo.md(
            r"""
    ## Summary

    **The model**

    $$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}, \qquad
      a^{(l)} = \varphi(z^{(l)}), \qquad
      p = \mathrm{softmax}(z^{(L)}).$$

    **The objective**

    $$J(\theta) = -\frac{1}{N}\sum_{i=1}^{N}
      \log p_\theta(y_i \mid x_i).$$

    **The algorithm**

    1. Forward pass: compute and cache activations.
    2. Backward pass: propagate $\partial \mathcal{L}/\partial a$ using
       $\bar{z} = \bar{a}\odot\varphi'(z)$, $\bar{W} = \bar{z}a^\top$,
       $\bar{a}_{\text{prev}} = W^\top\bar{z}$.
    3. Update: $\theta \leftarrow \theta - \eta\nabla_\theta J$.

    <br>

    That is a neural network. Everything else in this course is about making
    this loop **work better, faster and more reliably** at scale.
    """
            + footer(19)
        )

    summary_slide()

@app.cell
def _(mo):
    def next_slide():
        return mo.md(
            r"""
    ## Next lecture

    ### Convolutional neural networks

    - Why fully connected layers are a bad fit for images.
    - Convolution, padding, stride and pooling.
    - Weight sharing and translation equivariance.
    - Classic architectures: LeNet, AlexNet, VGG, ResNet.
    - Modern backbones and transfer learning.

    <br>

    *Reading:* Goodfellow, Bengio & Courville, Chapter 9.
    """
            + footer(20)
        )

    next_slide()


if __name__ == "__main__":
    app.run()