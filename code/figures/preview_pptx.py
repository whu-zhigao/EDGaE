#!/usr/bin/env python3
"""Render an approximate preview of EDGaEarch.pptx with matplotlib.

Reads the actual pptx via python-pptx (geometry, fills, line styles,
dash flags, text) and redraws it — used only for visual verification
since LibreOffice/qlmanage rendering is unavailable here.  Not a pixel
preview: fonts differ, rounded corners are approximated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
SRC = HERE / "EDGaEarch.pptx"
OUT = HERE / ".pptx-preview" / "pptx-preview.png"
OUT.parent.mkdir(exist_ok=True)

EMU = 914400.0


def rgb(color_format):
    try:
        c = color_format.rgb
        return (c[0] / 255, c[1] / 255, c[2] / 255)
    except Exception:
        return None


def is_dashed(shape):
    ln = shape.line._get_or_add_ln() if hasattr(shape, "line") else None
    if ln is None:
        return False
    d = ln.find(qn("a:prstDash"))
    return d is not None and d.get("val") in ("dash", "sysDash")


def shape_texts(shape):
    out = []
    if not shape.has_text_frame:
        return out
    for p in shape.text_frame.paragraphs:
        t = "".join(r.text for r in p.runs)
        if t.strip():
            align = {PP_ALIGN.CENTER: "center",
                     PP_ALIGN.RIGHT: "right"}.get(p.alignment, "left")
            size = max((r.font.size.pt for r in p.runs
                        if r.font.size is not None), default=10)
            bold = any(r.font.bold for r in p.runs)
            out.append((t, align, size, bold))
    return out


def main():
    prs = Presentation(str(SRC))
    slide = prs.slides[0]
    W = prs.slide_width / EMU
    H = prs.slide_height / EMU

    fig, ax = plt.subplots(figsize=(W, H), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # y down, like slide coords
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for shp in slide.shapes:
        x = shp.left / EMU
        y = shp.top / EMU
        w = shp.width / EMU
        h = shp.height / EMU

        if shp._element.tag == qn("p:cxnSp"):
            x1, y1 = shp.begin_x / EMU, shp.begin_y / EMU
            x2, y2 = shp.end_x / EMU, shp.end_y / EMU
            ec = rgb(shp.line.color) or (0.2, 0.2, 0.2)
            ax.plot([x1, x2], [y1, y2], color=ec, lw=1.6,
                    ls="--" if is_dashed(shp) else "-", zorder=4,
                    solid_capstyle="butt")
            # arrowhead at end
            import math
            dx, dy = x2 - x1, y2 - y1
            n = math.hypot(dx, dy) or 1
            ux, uy = dx / n, dy / n
            px, py = -uy, ux
            ax.fill([x2, x2 - 0.10 * ux + 0.04 * px, x2 - 0.10 * ux - 0.04 * px],
                    [y2, y2 - 0.10 * uy + 0.04 * py, y2 - 0.10 * uy - 0.04 * py],
                    color=ec, zorder=5)
            continue

        fc = None
        ec = None
        try:
            fc = rgb(shp.fill.fore_color)
        except Exception:
            pass
        try:
            ec = rgb(shp.line.color)
        except Exception:
            pass

        auto = None
        try:
            auto = shp.auto_shape_type
        except Exception:
            pass

        if "PARALLELOGRAM" in str(auto):
            sk = 0.25 * h
            poly = Polygon([(x + sk, y), (x + w, y),
                            (x + w - sk, y + h), (x, y + h)],
                           closed=True, facecolor=fc or "none",
                           edgecolor=ec or "none", lw=0.8, zorder=3)
            ax.add_patch(poly)
        else:
            rad = 0.03 if (fc or ec) else 0.0
            patch = FancyBboxPatch((x, y), w, h,
                                   boxstyle=f"round,pad=0,rounding_size={rad}",
                                   facecolor=fc or "none",
                                   edgecolor=ec or "none", lw=1.0, zorder=3,
                                   linestyle="--" if is_dashed(shp) else "-")
            ax.add_patch(patch)

        for (t, align, size, bold) in shape_texts(shp):
            ha = {"left": "left", "center": "center", "right": "right"}[align]
            tx = {"left": x + 0.07, "center": x + w / 2, "right": x + w - 0.07}[align]
            # crude vertical stack: first line near top, rest spread
            ny = y + h / 2
            if len(shape_texts(shp)) > 1:
                idx = [s[0] for s in shape_texts(shp)].index(t)
                ny = y + 0.25 + idx * 0.30
            ax.text(tx, ny, t, ha=ha, va="center",
                    fontsize=size * 0.9,
                    fontweight="bold" if bold else "normal",
                    color=(0.1, 0.1, 0.1), zorder=6,
                    family="serif")

    fig.savefig(OUT, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
