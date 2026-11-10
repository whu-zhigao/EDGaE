#!/usr/bin/env python3
"""Losslessly wrap fig/EDGaEarch.png into a PDF container.

The approved artwork stays pixel-identical (matplotlib's PDF backend
embeds raster data with FlateDecode — no JPEG, no resampling); only the
container changes so the manuscript can reference a .pdf like every
other figure.  Page width is set to 3.5 in (IEEE column width), so the
raster lands at ~439 dpi at final size.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
PNG = HERE.parent / "fig" / "EDGaEarch.png"
PDF = HERE.parent / "fig" / "EDGaEarch.pdf"
PAGE_W_IN = 3.5  # IEEE column width


def main():
    im = Image.open(PNG)
    dpi = im.width / PAGE_W_IN
    w, h = im.width / dpi, im.height / dpi
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(im, interpolation="none")
    ax.set_xlim(0, im.width)
    ax.set_ylim(im.height, 0)
    fig.savefig(PDF, dpi=dpi)
    plt.close(fig)
    print(f"wrote {PDF} ({im.width}x{im.height}px @ {dpi:.0f}dpi, {w:.2f}x{h:.2f}in)")


if __name__ == "__main__":
    main()
