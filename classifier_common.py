import torch
import torch.nn as nn

import numpy as np

from utils import get_device

import cairocffi as cairo
import struct
from pathlib import Path

import gdown

import matplotlib.pyplot as plt


IMG_WIDTH = 48


class Model(nn.Module):
    def __init__(self, out_dim):
        super(Model, self).__init__()

        self.in_dim = (IMG_WIDTH, IMG_WIDTH)
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4608, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(100, self.out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def download_data(dataset_names):
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        dataset_path = datasets_dir / f"{dataset_name}.bin"
        if dataset_path.exists():
            continue

        url = f"https://storage.googleapis.com/quickdraw_dataset/full/binary/{dataset_name}.bin"
        try:
            result = gdown.download(url=url, output=str(dataset_path), quiet=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download dataset '{dataset_name}' from {url} to {dataset_path}."
            ) from exc

        if result is None or not dataset_path.exists():
            raise RuntimeError(
                f"Failed to download dataset '{dataset_name}' from {url} to {dataset_path}."
            )


def vector_to_raster(
    vector_images,
    side=28,
    line_diameter=16,
    padding=16,
    bg_color=(0, 0, 0),
    fg_color=(1, 1, 1),
):
    """Padding and line_diameter are relative to the original 256x256 image."""

    original_side = 256.0

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2.0 + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2.0, total_padding / 2.0)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.0
        offset = offset.reshape(-1, 1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)

    return raster_images


# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def unpack_drawing(file_handle):
    (key_id,) = struct.unpack("Q", file_handle.read(8))
    (country_code,) = struct.unpack("2s", file_handle.read(2))
    (recognized,) = struct.unpack("b", file_handle.read(1))
    (timestamp,) = struct.unpack("I", file_handle.read(4))
    (n_strokes,) = struct.unpack("H", file_handle.read(2))
    image = []
    for _ in range(n_strokes):
        (n_points,) = struct.unpack("H", file_handle.read(2))
        fmt = str(n_points) + "B"
        x = struct.unpack(fmt, file_handle.read(n_points))
        y = struct.unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        "key_id": key_id,
        "country_code": country_code,
        "recognized": recognized,
        "timestamp": timestamp,
        "image": image,
    }


def unpack_drawings(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def load_images(path, no_of_images=None):
    imgs_vector = [
        drawing["image"] for drawing in unpack_drawings(path) if drawing["recognized"]
    ]

    if no_of_images is not None:
        assert no_of_images <= len(imgs_vector), "Too few images!"
        imgs_vector = imgs_vector[:no_of_images]

    imgs = np.array(
        vector_to_raster(imgs_vector, side=IMG_WIDTH, line_diameter=12, padding=16)
    )
    imgs = (imgs.reshape(-1, IMG_WIDTH, IMG_WIDTH) / 255 - 0.5) / 0.5
    return imgs


def prepare_data(dataset_names, batch_size, no_of_batches):
    dataset = []

    for label, dataset_name in enumerate(dataset_names):
        imgs_vector = [
            drawing["image"]
            for drawing in unpack_drawings(f"datasets/{dataset_name}.bin")
            if drawing["recognized"]
        ]

        no_of_full_batches = len(imgs_vector) // batch_size
        assert no_of_batches <= no_of_full_batches, (
            f"Too few images in {dataset_name}! {no_of_batches=}, {no_of_full_batches=}"
        )

        imgs = imgs_vector[: no_of_batches * batch_size]
        imgs = np.array(
            vector_to_raster(imgs, side=IMG_WIDTH, line_diameter=12, padding=16)
        )
        imgs = (imgs.reshape(-1, IMG_WIDTH, IMG_WIDTH) / 255 - 0.5) / 0.5

        img_with_labels = [(img, label) for img in imgs]
        dataset.extend(img_with_labels)

        print(
            f"Taken {no_of_batches * batch_size} / {no_of_full_batches * batch_size} = {100 * no_of_batches / no_of_full_batches:.2f}% of {dataset_name}"
        )

    return dataset


def compute_error_rate(model, data_loader, device):
    model.eval()

    num_errs, num_examples = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.float().reshape(x.size(0), 1, IMG_WIDTH, IMG_WIDTH).to(device)
            y = y.to(device)
            out = model(x)

            _, pred = out.max(dim=1)
            num_errs += (pred != y.data).sum().item()
            num_examples += x.size(0)

    return num_errs / num_examples


def draw_images_with_score(img, size=(5, 10)):
    rows, cols = size
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))
    for row in range(rows):
        for col in range(cols):
            bear = torch.tensor(img[row * cols + col][0]).cpu()
            axes[row][col].imshow(bear)

            quality = img[row * cols + col][1]

            axes[row][col].title.set_text("%.4f" % quality)  # "%.2f" %
            axes[row][col].axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def train(model, no_of_epochs, data_loaders, opt, log=1000):
    model.train()
    device = get_device()

    try:
        it = 0
        for epoch in range(no_of_epochs):
            model.train()

            for x, y in data_loaders["train"]:
                x = x.float().reshape(x.size(0), 1, IMG_WIDTH, IMG_WIDTH).to(device)
                y = y.to(device)
                opt.zero_grad()
                it += 1

                out = model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                opt.step()

                _, pred = out.max(dim=1)
                batch_err = (pred != y).sum().item() / out.size(0)

                if it % log == 0:
                    print(f"it={it}, batch_err={batch_err * 100.0}")

            val_err = compute_error_rate(
                model, data_loader=data_loaders["valid"], device=device
            )
            print(f"Epoch {epoch + 1}: val_err = {100 * val_err:.2f}")
    except KeyboardInterrupt:
        pass
