"""
Trajectory Classifier
----------------------
A small PyTorch multi-class classifier that identifies the type of a 2D
trajectory (ramp, sinusoid, circle, ellipse) from its raw sampled points.

Trajectories are generated synthetically (see generate_* functions, reused
from the trajectory-generator project) and represented as flattened
(x, y) / (time, value) vectors, so they can be fed into a simple linear
classifier.
"""

import math
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42

DURATION = 10.0
DT = 0.05  # sampling step; fixed across all trajectories so every sample
           # has the same number of points

SAMPLES_PER_CLASS = 50
CLASS_NAMES = {0: "ramp", 1: "sinusoid", 2: "circle", 3: "ellipse"}

TRAIN_SPLIT = 0.8
EPOCHS = 800
LEARNING_RATE = 0.1


# ---------------------------------------------------------------------------
# Trajectory generators
# ---------------------------------------------------------------------------

def generate_ramp(duration, dt, slope):
    t = 0.0
    times, values = [], []
    while t <= duration:
        values.append(slope * t)
        times.append(t)
        t += dt
    return times, values


def generate_sinusoid(duration, dt, amplitude, frequency):
    t = 0.0
    times, values = [], []
    while t <= duration:
        values.append(amplitude * math.sin(2 * math.pi * frequency * t))
        times.append(t)
        t += dt
    return times, values


def generate_circle(duration, dt, frequency, radius):
    t = 0.0
    x_values, y_values = [], []
    while t <= duration:
        x_values.append(radius * math.cos(2 * math.pi * frequency * t))
        y_values.append(radius * math.sin(2 * math.pi * frequency * t))
        t += dt
    return x_values, y_values


def generate_ellipse(duration, dt, frequency, radius_x, radius_y):
    t = 0.0
    x_values, y_values = [], []
    while t <= duration:
        x_values.append(radius_x * math.cos(2 * math.pi * frequency * t))
        y_values.append(radius_y * math.sin(2 * math.pi * frequency * t))
        t += dt
    return x_values, y_values


# ---------------------------------------------------------------------------
# Feature encoding: a trajectory is a list of (a, b) pairs; we flatten it
# into a single fixed-length vector so it can be used as model input.
# ---------------------------------------------------------------------------

def flatten(a_values, b_values):
    """Interleave two sequences into a single flat list: [a0, b0, a1, b1, ...]."""
    flat = []
    for a, b in zip(a_values, b_values):
        flat.append(a)
        flat.append(b)
    return flat


def unflatten(flat_values):
    """Inverse of flatten(): split a flat list back into (a_values, b_values)."""
    a_values, b_values = [], []
    for i, value in enumerate(flat_values):
        if i % 2 == 0:
            a_values.append(value)
        else:
            b_values.append(value)
    return a_values, b_values


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(samples_per_class=SAMPLES_PER_CLASS):
    features, labels = [], []

    for _ in range(samples_per_class):
        slope = random.uniform(-2.0, 2.0)
        times, values = generate_ramp(DURATION, DT, slope)
        features.append(flatten(times, values))
        labels.append(0)  # ramp

    for _ in range(samples_per_class):
        amplitude = random.uniform(0.5, 3.0)
        frequency = random.uniform(0.3, 1.0)
        times, values = generate_sinusoid(DURATION, DT, amplitude, frequency)
        features.append(flatten(times, values))
        labels.append(1)  # sinusoid

    for _ in range(samples_per_class):
        radius = random.uniform(0.5, 3.0)
        frequency = random.uniform(0.3, 1.0)
        x_values, y_values = generate_circle(DURATION, DT, frequency, radius)
        features.append(flatten(x_values, y_values))
        labels.append(2)  # circle

    for _ in range(samples_per_class):
        radius_x = random.uniform(0.5, 3.0)
        radius_y = radius_x * random.uniform(0.2, 0.6)  # keep ellipses visually
                                                          # distinct from circles
        frequency = random.uniform(0.3, 1.0)
        x_values, y_values = generate_ellipse(DURATION, DT, frequency, radius_x, radius_y)
        features.append(flatten(x_values, y_values))
        labels.append(3)  # ellipse

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def train_test_split(x, y, train_split=TRAIN_SPLIT):
    indices = torch.randperm(len(x))
    split = int(train_split * len(x))
    train_idx, test_idx = indices[:split], indices[split:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiClassifier(nn.Module):
    """Single linear layer: raw logits in, softmax handled by CrossEntropyLoss."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x)


def train(model, x_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")


def accuracy(model, x, y):
    with torch.no_grad():
        predictions = torch.argmax(model(x), dim=1)
        return (predictions == y).float().mean().item()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_predictions(model, x_test, y_test, n_examples=4):
    with torch.no_grad():
        predicted = torch.argmax(model(x_test), dim=1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i in range(n_examples):
        a_values, b_values = unflatten(x_test[i].tolist())
        true_label = y_test[i].item()
        pred_label = predicted[i].item()
        color = "green" if true_label == pred_label else "red"

        ax = axes[i]
        ax.plot(a_values, b_values)
        if true_label in (2, 3):  # circle / ellipse: keep aspect ratio correct
            ax.axis("equal")
        ax.set_title(
            f"true: {CLASS_NAMES[true_label]} | predicted: {CLASS_NAMES[pred_label]}",
            color=color,
        )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    x, y = generate_dataset()
    x_train, y_train, x_test, y_test = train_test_split(x, y)

    model = MultiClassifier(n_features=x.shape[1], n_classes=4)
    train(model, x_train, y_train)

    print(f"train accuracy: {accuracy(model, x_train, y_train):.4f}")
    print(f"test accuracy:  {accuracy(model, x_test, y_test):.4f}")

    plot_predictions(model, x_test, y_test)


if __name__ == "__main__":
    main()
