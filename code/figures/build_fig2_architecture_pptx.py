#!/usr/bin/env python3
"""Build an editable PowerPoint version of Fig. 2 (EDGaEarch.png).

Every element is a native PPT shape (rounded rectangles, connectors,
parallelogram/rectangle device glyphs, text boxes) so the figure can be
edited directly in PowerPoint / Keynote.  Geometry and palette follow the
original PNG (see paper/fig/EDGaEarch.png).

v2 changes vs the original EDGaEarch.pptx (which is preserved untouched):
  - "Privacy policies" governor input replaced by "Sampler spec & horizon"
  - "Data Plane (Server-Edge Substrate)" chip added between the planes
  - all text sizes raised ~1pt for print legibility

Run:  python3 paper/figsource/build_fig2_architecture_pptx.py
Out:  paper/figsource/EDGaEarch-v2.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
OUT = HERE / "EDGaEarch-v2.pptx"

# ── palette (sampled from the original figure) ──────────────────────────
CANVAS = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x00, 0x00, 0x00)
ARROW = RGBColor(0x33, 0x33, 0x33)

BLUE_FILL = RGBColor(0xD9, 0xE4, 0xF3)   # governor box + control-plane chip
BLUE_PANEL = RGBColor(0xDC, 0xE6, 0xF5)  # server-centric execution panel
BLUE_LINE = RGBColor(0x7A, 0x93, 0xB5)

GREEN_FILL = RGBColor(0xDE, 0xEB, 0xD8)  # manifest box
GREEN_CHIP = RGBColor(0xD8, 0xEA, 0xD3)  # execution-plane chip
GREEN_LINE = RGBColor(0x6E, 0x9E, 0x60)
GREEN_CHIP_LINE = RGBColor(0x8F, 0xB8, 0x8A)
GREEN_CHECK = RGBColor(0x4C, 0x8A, 0x3F)

LAV_FILL = RGBColor(0xE3, 0xDF, 0xF0)   # executors box
LAV_PANEL = RGBColor(0xE7, 0xE2, 0xF2)  # edge-local execution panel
LAV_LINE = RGBColor(0x8B, 0x7F, 0xC0)

KHAKI_FILL = RGBColor(0xE8, 0xEB, 0xD9)  # hybrid execution panel
KHAKI_LINE = RGBColor(0xA3, 0xAC, 0x7C)

CAP_FILL = RGBColor(0xEF, 0xEF, 0xED)    # bottom caption bar
CAP_LINE = RGBColor(0x9A, 0x9A, 0x9A)

SERVER_FILL = RGBColor(0x4A, 0x55, 0x60)  # server glyph body
SERVER_DETAIL = RGBColor(0x8A, 0x93, 0x9B)

DEV_TOP = RGBColor(0xC9, 0xCF, 0xD4)      # edge-device glyph top face
DEV_FRONT = RGBColor(0x9A, 0xA1, 0xA8)    # edge-device glyph front face
DEV_LINE = RGBColor(0x6B, 0x72, 0x78)

FONT = "Times New Roman"

SLIDE_W, SLIDE_H = 14.0, 8.0  # inches — same 1.75:1 aspect as the PNG


# ── helpers ─────────────────────────────────────────────────────────────
def _set_dash(shp):
    """Version-proof dashed line: raw <a:prstDash val='dash'/> on a:ln."""
    ln = shp.line._get_or_add_ln()
    d = ln.find(qn("a:prstDash"))
    if d is None:
        d = ln.makeelement(qn("a:prstDash"), {"val": "dash"})
        ln.append(d)
    else:
        d.set("val", "dash")


def rrect(slide, name, x, y, w, h, fill, line, lw=1.0, radius=0.12,
          dash=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    shp.name = name
    try:
        shp.adjustments[0] = radius
    except Exception:
        pass
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(lw)
    if dash:
        _set_dash(shp)
    shp.shadow.inherit = False
    return shp


def rect(slide, name, x, y, w, h, fill, line, lw=1.0):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    shp.name = name
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    return shp


def para(tf, first, text, size, bold=False, color=INK, align=PP_ALIGN.LEFT,
         space_before=0, space_after=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if space_before:
        p.space_before = Pt(space_before)
    if space_after:
        p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return p


def textbox(slide, name, x, y, w, h, anchor=MSO_ANCHOR.TOP, mgn=0.03):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(mgn)
    tf.margin_top = tf.margin_bottom = Inches(0.01)
    return tb, tf


def shape_text(shp, anchor=MSO_ANCHOR.MIDDLE, mgn=0.06):
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(mgn)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    return tf


def connector(slide, name, x1, y1, x2, y2, lw=2.0, dashed=False,
              arrow=False):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                      Inches(x1), Inches(y1),
                                      Inches(x2), Inches(y2))
    conn.name = name
    conn.line.color.rgb = ARROW
    conn.line.width = Pt(lw)
    if dashed:
        _set_dash(conn)
    if arrow:
        ln = conn.line._get_or_add_ln()
        tail = ln.find(qn("a:tailEnd"))
        if tail is None:
            tail = ln.makeelement(qn("a:tailEnd"),
                                  {"type": "triangle", "w": "med", "len": "med"})
            ln.append(tail)
        else:
            tail.set("type", "triangle")
    return conn


def server_stack(slide, tag, x, y_top):
    """Three stacked dark server units with a front-panel detail each."""
    for i in range(3):
        y = y_top + i * 0.31
        rrect(slide, f"icon-{tag}-server{i+1}", x, y, 0.72, 0.26,
              SERVER_FILL, None, radius=0.18)
        rect(slide, f"icon-{tag}-detail{i+1}", x + 0.06, y + 0.07,
             0.10, 0.12, SERVER_DETAIL, None)


def edge_device(slide, tag, x, y):
    """One flat edge-device glyph: parallelogram top + rectangle front."""
    top = slide.shapes.add_shape(MSO_SHAPE.PARALLELOGRAM,
                                 Inches(x - 0.05), Inches(y - 0.11),
                                 Inches(0.44), Inches(0.11))
    top.name = f"icon-{tag}-top"
    top.fill.solid()
    top.fill.fore_color.rgb = DEV_TOP
    top.line.color.rgb = DEV_LINE
    top.line.width = Pt(0.75)
    top.shadow.inherit = False
    front = rect(slide, f"icon-{tag}-front", x, y, 0.44, 0.17,
                 DEV_FRONT, DEV_LINE, lw=0.75)


# ── build ───────────────────────────────────────────────────────────────
def main():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

    # canvas background
    bg = rect(slide, "canvas", 0, 0, SLIDE_W, SLIDE_H, CANVAS, None)
    bg.name = "canvas-background"

    # ═══ Control Plane container ═══
    rrect(slide, "plane-control", 0.07, 0.16, 13.86, 3.36,
          None, BLUE_LINE, lw=2.0, radius=0.04, dash=True)
    chip = rrect(slide, "chip-control", 0.14, 0.16, 3.64, 0.52,
                 BLUE_FILL, BLUE_LINE, lw=1.0, radius=0.25)
    tf = shape_text(chip, mgn=0.12)
    para(tf, True, "Control Plane (Server-Side)", 14, bold=True)

    # B — Graph + Constraints
    b = rrect(slide, "box-input", 0.35, 0.88, 2.94, 2.40, CANVAS, BLUE_LINE)
    tf = shape_text(b, anchor=MSO_ANCHOR.TOP)
    para(tf, True, "Graph + Constraints", 14, bold=True, space_after=8)
    for t in ("Topology & features", "Memory limits",
              "Bandwidth & latency", "Sampler spec & horizon"):
        para(tf, False, "•  " + t, 11.5, space_after=5)

    # C — Placement Governor (with 3 nested steps)
    c = rrect(slide, "box-governor", 3.78, 0.64, 3.01, 2.56,
              BLUE_FILL, BLUE_LINE)
    tf = shape_text(c, anchor=MSO_ANCHOR.TOP)
    para(tf, True, "Placement Governor", 14, bold=True,
         align=PP_ALIGN.CENTER, space_after=6)
    for i, (t, y) in enumerate((("Cost Model", 1.12),
                                ("Feasibility Check", 1.72),
                                ("Placement Decision", 2.32))):
        inner = rrect(slide, f"box-governor-step{i+1}", 4.05, y, 2.47, 0.44,
                      CANVAS, BLUE_LINE, lw=0.75, radius=0.18)
        tfi = shape_text(inner, mgn=0.03)
        para(tfi, True, t, 12, align=PP_ALIGN.CENTER)

    # D — Placement Manifest
    d = rrect(slide, "box-manifest", 7.49, 0.64, 3.08, 2.56,
              GREEN_FILL, GREEN_LINE)
    tf = shape_text(d, anchor=MSO_ANCHOR.TOP)
    para(tf, True, "Placement Manifest", 13.5, bold=True,
         align=PP_ALIGN.CENTER)
    para(tf, False, "(Execution Plan)", 13.5, bold=True,
         align=PP_ALIGN.CENTER, space_after=8)
    for t in ("Server / Edge placement", "Data partition & cache plan",
              "Runtime configuration"):
        p = tf.add_paragraph()
        p.space_after = Pt(5)
        r1 = p.add_run()
        r1.text = "✓  "
        r1.font.name = FONT
        r1.font.size = Pt(11.5)
        r1.font.bold = True
        r1.font.color.rgb = GREEN_CHECK
        r2 = p.add_run()
        r2.text = t
        r2.font.name = FONT
        r2.font.size = Pt(11.5)
        r2.font.color.rgb = INK

    # E — Mode-Aware Executors
    e = rrect(slide, "box-executors", 11.34, 0.64, 2.38, 2.56,
              LAV_FILL, LAV_LINE)
    tf = shape_text(e, anchor=MSO_ANCHOR.TOP)
    para(tf, True, "Mode-Aware Executors", 13.5, bold=True,
         align=PP_ALIGN.CENTER, space_after=8)
    for t in ("Server-Centric", "Hybrid", "Edge-Local"):
        para(tf, False, t, 12, align=PP_ALIGN.CENTER, space_after=6)

    # ═══ pipeline arrows ═══
    connector(slide, "arrow-input-governor", 3.29, 2.08, 3.78, 2.08,
              arrow=True)
    connector(slide, "arrow-governor-manifest", 6.79, 2.08, 7.49, 2.08,
              arrow=True)
    connector(slide, "arrow-manifest-executors", 10.57, 2.08, 11.34, 2.08,
              dashed=True, arrow=True)

    # ═══ feedback loop (dashed, orthogonal) ═══
    connector(slide, "arrow-feedback-v1", 12.25, 3.20, 12.25, 3.84,
              lw=1.75, dashed=True)
    connector(slide, "arrow-feedback-h", 12.25, 3.84, 4.83, 3.84,
              lw=1.75, dashed=True)
    connector(slide, "arrow-feedback-v2", 4.83, 3.84, 4.83, 3.20,
              lw=1.75, dashed=True, arrow=True)
    _, tf = textbox(slide, "label-feedback", 6.42, 3.50, 3.40, 0.30)
    para(tf, True, "Feedback (metrics, state)", 12,
         align=PP_ALIGN.CENTER)

    # dispatch down into the execution plane
    connector(slide, "arrow-dispatch-edge", 12.74, 3.20, 12.74, 4.80,
              lw=1.75, dashed=True, arrow=True)

    # ═══ Execution Plane container ═══
    rrect(slide, "plane-execution", 0.07, 4.16, 13.86, 3.12,
          None, GREEN_CHIP_LINE, lw=2.0, radius=0.04, dash=True)
    chip = rrect(slide, "chip-execution", 0.14, 4.16, 3.85, 0.48,
                 GREEN_CHIP, GREEN_CHIP_LINE, lw=1.0, radius=0.25)
    tf = shape_text(chip, mgn=0.12)
    para(tf, True, "Execution Plane (Distributed)", 14, bold=True)

    # ═══ Data Plane chip (inter-plane band, left of the feedback label) ═══
    dp = rrect(slide, "chip-data-plane", 0.14, 3.66, 4.60, 0.44,
               CAP_FILL, CAP_LINE, lw=1.0, radius=0.25)
    tf = shape_text(dp, mgn=0.10)
    para(tf, True, "Data Plane (Server–Edge Substrate)", 11.5, bold=True)

    # G — Server-Centric panel
    g = rrect(slide, "panel-server", 0.35, 4.80, 3.99, 1.92,
              BLUE_PANEL, BLUE_LINE)
    _, tf = textbox(slide, "panel-server-title", 0.35, 4.86, 3.99, 0.30)
    para(tf, True, "Server-Centric", 14, bold=True, align=PP_ALIGN.CENTER)
    server_stack(slide, "server", 0.60, 5.42)
    gp = rrect(slide, "panel-server-text", 1.75, 5.44, 2.31, 1.20,
               CANVAS, BLUE_LINE, lw=0.75, radius=0.10)
    tf = shape_text(gp, mgn=0.08)
    para(tf, True, "•  Full computation on server", 11, space_after=4)
    para(tf, False, "•  Ship mini-batches from server", 11)

    # H — Hybrid panel
    rrect(slide, "panel-hybrid", 4.55, 4.80, 4.41, 1.92,
          KHAKI_FILL, KHAKI_LINE)
    _, tf = textbox(slide, "panel-hybrid-title", 4.55, 4.86, 4.41, 0.30)
    para(tf, True, "Hybrid", 14, bold=True, align=PP_ALIGN.CENTER)
    server_stack(slide, "hybrid", 4.85, 5.42)
    _, tf = textbox(slide, "panel-hybrid-plus", 5.62, 5.52, 0.40, 0.40)
    para(tf, True, "+", 20, bold=True, align=PP_ALIGN.CENTER)
    for i, (x, y) in enumerate(((6.20, 5.42), (6.38, 5.72), (6.56, 6.02))):
        edge_device(slide, f"hybrid-dev{i+1}", x, y)
    hp = rrect(slide, "panel-hybrid-text", 7.14, 5.44, 1.68, 1.20,
               CANVAS, KHAKI_LINE, lw=0.75, radius=0.10)
    tf = shape_text(hp, mgn=0.08)
    para(tf, True, "•  Split computation", 11, space_after=4)
    para(tf, False, "•  Cache features at edge", 11)

    # I — Edge-Local panel
    rrect(slide, "panel-edge", 9.24, 4.80, 4.48, 1.92,
          LAV_PANEL, LAV_LINE)
    _, tf = textbox(slide, "panel-edge-title", 9.24, 4.86, 4.48, 0.30)
    para(tf, True, "Edge-Local", 14, bold=True, align=PP_ALIGN.CENTER)
    for i, (x, y) in enumerate(((9.75, 5.38), (10.50, 5.38),
                                (9.92, 5.80), (10.67, 5.80))):
        edge_device(slide, f"edge-dev{i+1}", x, y)
    ip = rrect(slide, "panel-edge-text", 11.48, 5.44, 2.03, 1.20,
               CANVAS, LAV_LINE, lw=0.75, radius=0.10)
    tf = shape_text(ip, mgn=0.08)
    para(tf, True, "•  Train locally on edges", 11)

    # J — bottom caption bar
    j = rrect(slide, "caption-bar", 0.35, 6.96, 13.30, 0.60,
              CAP_FILL, CAP_LINE, lw=1.0, radius=0.25)
    tf = shape_text(j)
    para(tf, True, "GNN Training on Graphs", 17, bold=True,
         align=PP_ALIGN.CENTER)

    prs.save(OUT)
    print(f"wrote {OUT}")
    print(f"shapes on slide: {len(slide.shapes)}")


if __name__ == "__main__":
    main()
