#!/usr/bin/env python3
"""Render an editable qualitative-comparison Figure 2.

The renderer reads only persisted validation outputs and the existing external
qualitative assets.  It does not load a model, run inference, train, or alter
the RGB pixels.  The PPTX is assembled directly as OOXML so it remains usable
without adding a project dependency: images are separate picture objects,
labels are text boxes, and the red ROI outlines are editable vector shapes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOUNT = Path("/mnt/wmcontent/GLX/icassp/MBRS")
VALIDATION_MANIFEST = MOUNT / "reports/content_selector/validation_manifest.pt"
MBRS_DIR = MOUNT / "reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5"
QUALITATIVE_DIR = MOUNT / "visualizations/external_qualitative"
FROZEN_MAIN = QUALITATIVE_DIR / "qualitative_5x6_main.png"
FROZEN_ZOOM = QUALITATIVE_DIR / "qualitative_5x6_zoom4x.png"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"
REPORT = PROJECT_ROOT / "reports" / "figure2_qualitative_editable_check.md"

FIXED_INDICES = (0, 10, 20, 30, 40)
OUTPUT_NAMES = {
    "Ours-base": "controlled_seed17_global_continuation_outputs.pt",
    "Ours": "controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt",
    "Ours + OKLab": "seed17_crop_hard16_stride8_top10_global_oklab_g12p5_outputs.pt",
}
METHODS_6 = ("Original", "HiDDeN-64", "MaskWM-D_64", "Ours-base", "Ours", "Ours + OKLab")
METHODS_5 = METHODS_6[:-1]
FROZEN_X_BOUNDS = ((86, 602), (741, 1257), (1395, 1911), (2050, 2566), (2704, 3220), (3359, 3875))
FROZEN_Y_BOUNDS = ((285, 800), (868, 1383), (1451, 1966), (2034, 2549), (2617, 3132))
METHOD_SOURCE_NAMES = {
    "Original": "Original",
    "HiDDeN-64": "HiDDeN-64",
    "MaskWM-D_64": "MaskWM-D_64",
    "Ours-base": "MBRS Global continuation",
    "Ours": "Hard Top10",
    "Ours + OKLab": "Hard Top10 + OKLab",
}

EMU_PER_INCH = 914400
SLIDE_W = 13.333
SLIDE_H = 7.5
RED = "E5483C"
INK = "202934"
MUTED = "66727D"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def display_tensor(tensor: torch.Tensor) -> Image.Image:
    """Match the existing qualitative display: clipped RGB, bilinear 4x view."""
    value = rgb_tensor(tensor).unsqueeze(0)
    value = F.interpolate(value, size=(512, 512), mode="bilinear", align_corners=False, antialias=True)[0]
    array = np.rint(value.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def load_encoded(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["encoded"].float()


def fixed_roi(original: torch.Tensor, hard: torch.Tensor) -> tuple[int, int, int, int]:
    """Select the registered native 32x32 block from the Hard Top10 tail."""
    left = rgb_tensor(original).permute(1, 2, 0).numpy()
    right = rgb_tensor(hard).permute(1, 2, 0).numpy()
    delta = deltaE_ciede2000(rgb2lab(left), rgb2lab(right))
    y, x = np.unravel_index(int(delta.argmax()), delta.shape)
    return int((x // 32) * 32), int((y // 32) * 32), 32, 32


def local_mse(image: torch.Tensor, reference: torch.Tensor, roi: tuple[int, int, int, int]) -> float:
    x, y, width, height = roi
    return float(((rgb_tensor(image)[:, y : y + height, x : x + width] - rgb_tensor(reference)[:, y : y + height, x : x + width]) ** 2).mean())


def frozen_panel(image: Image.Image, row: int, column: int) -> Image.Image:
    x0, x1 = FROZEN_X_BOUNDS[column]
    y0, y1 = FROZEN_Y_BOUNDS[row]
    return image.crop((x0, y0, x1, y1)).convert("RGB")


def frozen_external_mse(
    main_image: Image.Image,
    row: int,
    method_column: int,
    roi: tuple[int, int, int, int],
) -> float:
    reference = np.asarray(frozen_panel(main_image, row, 0), dtype=np.float32) / 255.0
    candidate = np.asarray(frozen_panel(main_image, row, method_column), dtype=np.float32) / 255.0
    x, y, width, height = roi
    scale_x = reference.shape[1] / 512.0
    scale_y = reference.shape[0] / 512.0
    left = int(round(x * 4 * scale_x))
    top = int(round(y * 4 * scale_y))
    right = int(round((x + width) * 4 * scale_x))
    bottom = int(round((y + height) * 4 * scale_y))
    return float(((candidate[top:bottom, left:right] - reference[top:bottom, left:right]) ** 2).mean())


def select_samples(
    original: torch.Tensor,
    global_output: torch.Tensor,
    hard_output: torch.Tensor,
    frozen_main: Image.Image,
) -> tuple[tuple[int, int], dict[int, dict[str, float | tuple[int, int, int, int]]]]:
    """Choose two fixed samples using only saved output differences.

    Positive local improvement is primary.  External-baseline local error is a
    secondary visibility proxy, so the chosen rows show both the MBRS local
    improvement and a visible contrast against the two external references.
    """
    records: dict[int, dict[str, float | tuple[int, int, int, int]]] = {}
    candidates: list[dict[str, float | int]] = []
    for index in FIXED_INDICES:
        roi = fixed_roi(original[index], hard_output[index])
        base = local_mse(global_output[index], original[index], roi)
        ours = local_mse(hard_output[index], original[index], roi)
        row = FIXED_INDICES.index(index)
        hidden_mse = frozen_external_mse(frozen_main, row, 1, roi)
        mask_mse = frozen_external_mse(frozen_main, row, 2, roi)
        improvement = base - ours
        external_contrast = (hidden_mse + mask_mse) / 2.0
        records[index] = {
            "roi": roi,
            "base_local_mse": base,
            "ours_local_mse": ours,
            "local_improvement": improvement,
            "hidden_local_mse": hidden_mse,
            "maskwm_local_mse": mask_mse,
            "external_contrast": external_contrast,
        }
        if improvement > 0:
            candidates.append({"index": index, "improvement": improvement, "external_contrast": external_contrast})

    if len(candidates) < 2:
        raise RuntimeError("Fewer than two fixed samples show positive Ours local improvement")
    max_improvement = max(float(item["improvement"]) for item in candidates)
    max_contrast = max(float(item["external_contrast"]) for item in candidates)
    for item in candidates:
        item["score"] = 0.65 * float(item["improvement"]) / max_improvement + 0.35 * float(item["external_contrast"]) / max_contrast
        records[int(item["index"])] ["selection_score"] = float(item["score"])
    chosen = tuple(int(item["index"]) for item in sorted(candidates, key=lambda item: float(item["score"]), reverse=True)[:2])
    return chosen, records


def pil_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def build_assets(
    selected: tuple[int, int],
    frozen_main: Image.Image,
    frozen_zoom: Image.Image,
) -> dict[tuple[int, str, str], bytes]:
    assets: dict[tuple[int, str, str], bytes] = {}
    for index in selected:
        row = FIXED_INDICES.index(index)
        for column, method in enumerate(METHODS_6):
            assets[(index, method, "full")] = pil_bytes(frozen_panel(frozen_main, row, column))
            assets[(index, method, "zoom")] = pil_bytes(frozen_panel(frozen_zoom, row, column))
    return assets


def layout(ncols: int) -> dict[str, float | list[float]]:
    image = 1.35
    gap = 0.22
    sample_gap = 0.32
    left = 0.90
    col_gap = (SLIDE_W - left - 0.92 - ncols * image) / max(ncols - 1, 1)
    if col_gap < 0.10:
        image = 1.25
        col_gap = (SLIDE_W - left - 0.92 - ncols * image) / max(ncols - 1, 1)
    xs = [left + index * (image + col_gap) for index in range(ncols)]
    zoom2_y = 0.70
    full2_y = zoom2_y + image + gap
    zoom1_y = full2_y + image + sample_gap
    full1_y = zoom1_y + image + gap
    return {
        "image": image,
        "gap": gap,
        "xs": xs,
        "full_y": [full1_y, full2_y],
        "zoom_y": [zoom1_y, zoom2_y],
        "header_y": 7.18,
        "footer_y": 0.16,
    }


def render_preview(
    selected: tuple[int, int],
    methods: tuple[str, ...],
    assets: dict[tuple[int, str, str], bytes],
    records: dict[int, dict[str, float | tuple[int, int, int, int]]],
    psnr_values: dict[int, dict[str, float]],
    output: Path,
) -> None:
    spec = layout(len(methods))
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor="white")
    image = float(spec["image"])
    xs = list(spec["xs"])
    full_y = list(spec["full_y"])
    zoom_y = list(spec["zoom_y"])
    for column, method in enumerate(methods):
        fig.text((xs[column] + image / 2) / SLIDE_W, float(spec["header_y"]) / SLIDE_H, method, ha="center", va="center", fontsize=10 if len(methods) == 6 else 10.5, color=f"#{INK}", fontweight="bold")
    for row, index in enumerate(selected):
        sample_center = (full_y[row] + image / 2 + zoom_y[row] + image / 2) / 2
        fig.text(0.32 / SLIDE_W, sample_center / SLIDE_H, f"Sample {row + 1}", ha="center", va="center", rotation=90, fontsize=9.5, color=f"#{INK}", fontweight="bold")
        fig.text(0.65 / SLIDE_W, (full_y[row] + image / 2) / SLIDE_H, "full", ha="center", va="center", rotation=90, fontsize=7.2, color=f"#{MUTED}")
        fig.text(0.65 / SLIDE_W, (zoom_y[row] + image / 2) / SLIDE_H, "4× zoom", ha="center", va="center", rotation=90, fontsize=7.2, color=f"#{MUTED}")
        roi = records[index]["roi"]
        assert isinstance(roi, tuple)
        rx, ry, rw, rh = roi
        for column, method in enumerate(methods):
            full = Image.open(io.BytesIO(assets[(index, method, "full")])).convert("RGB")
            zoom = Image.open(io.BytesIO(assets[(index, method, "zoom")])).convert("RGB")
            for y_pos, panel_image in ((full_y[row], full), (zoom_y[row], zoom)):
                ax = fig.add_axes([xs[column] / SLIDE_W, y_pos / SLIDE_H, image / SLIDE_W, image / SLIDE_H])
                ax.imshow(panel_image, interpolation="none", aspect="equal")
                ax.set_axis_off()
            if method != "Original":
                value = psnr_values[index][method]
                fig.text((xs[column] + image / 2) / SLIDE_W, (full_y[row] - 0.055) / SLIDE_H, f"{value:.2f} dB", ha="center", va="top", fontsize=7.0, color=f"#{MUTED}")
            ax = fig.add_axes([xs[column] / SLIDE_W, full_y[row] / SLIDE_H, image / SLIDE_W, image / SLIDE_H])
            ax.add_patch(Rectangle((rx / 128, 1.0 - (ry + rh) / 128), rw / 128, rh / 128, fill=False, edgecolor=f"#{RED}", linewidth=1.25, transform=ax.transAxes))
            ax.set_axis_off()

    fig.text(0.5, float(spec["footer_y"]) / SLIDE_H, "Red boxes indicate zoomed regions. Ours-base denotes the MBRS global continuation, and Ours denotes Hard Top10.", ha="center", va="bottom", fontsize=7.2, color=f"#{MUTED}")
    fig.savefig(output, dpi=360, facecolor="white")
    plt.close(fig)


# --- Minimal Open XML PPTX writer -------------------------------------------------

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
ET.register_namespace("a", A_NS)
ET.register_namespace("p", P_NS)
ET.register_namespace("r", R_NS)


def qn(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def emu(inches: float) -> int:
    return int(round(inches * EMU_PER_INCH))


def add_xfrm(parent: ET.Element, x: float, y: float, width: float, height: float) -> None:
    xfrm = ET.SubElement(parent, qn(A_NS, "xfrm"))
    ET.SubElement(xfrm, qn(A_NS, "off"), {"x": str(emu(x)), "y": str(emu(SLIDE_H - y - height))})
    ET.SubElement(xfrm, qn(A_NS, "ext"), {"cx": str(emu(width)), "cy": str(emu(height))})


def nv_sp_pr(sp_tree: ET.Element, shape_id: int, name: str) -> tuple[ET.Element, ET.Element]:
    shape = ET.SubElement(sp_tree, qn(P_NS, "sp"))
    nv = ET.SubElement(shape, qn(P_NS, "nvSpPr"))
    ET.SubElement(nv, qn(P_NS, "cNvPr"), {"id": str(shape_id), "name": name})
    ET.SubElement(nv, qn(P_NS, "cNvSpPr"))
    ET.SubElement(nv, qn(P_NS, "nvPr"))
    return shape, ET.SubElement(shape, qn(P_NS, "spPr"))


def add_text_box(sp_tree: ET.Element, shape_id: int, x: float, y: float, width: float, height: float, value: str, *, size: int = 12, bold: bool = False, color: str = INK, align: str = "ctr", rotate: int = 0) -> None:
    shape, sp_pr = nv_sp_pr(sp_tree, shape_id, f"Text {shape_id}")
    add_xfrm(sp_pr, x, y, width, height)
    geom = ET.SubElement(sp_pr, qn(A_NS, "prstGeom"), {"prst": "rect"})
    ET.SubElement(geom, qn(A_NS, "avLst"))
    ET.SubElement(sp_pr, qn(A_NS, "noFill"))
    tx = ET.SubElement(shape, qn(P_NS, "txBody"))
    body = ET.SubElement(tx, qn(A_NS, "bodyPr"), {"wrap": "square", "anchor": "ctr"})
    if rotate:
        body.set("vert", "vert270")
    ET.SubElement(tx, qn(A_NS, "lstStyle"))
    for line in value.split("\n"):
        para = ET.SubElement(tx, qn(A_NS, "p"))
        ET.SubElement(para, qn(A_NS, "pPr"), {"algn": align})
        run = ET.SubElement(para, qn(A_NS, "r"))
        run_pr = ET.SubElement(run, qn(A_NS, "rPr"), {"lang": "en-US", "sz": str(size * 100), "b": "1" if bold else "0"})
        solid = ET.SubElement(run_pr, qn(A_NS, "solidFill"))
        ET.SubElement(solid, qn(A_NS, "srgbClr"), {"val": color})
        ET.SubElement(run_pr, qn(A_NS, "latin"), {"typeface": "Arial"})
        ET.SubElement(run, qn(A_NS, "t")).text = line
        ET.SubElement(para, qn(A_NS, "endParaRPr"), {"lang": "en-US", "sz": str(size * 100)})


def add_rect(sp_tree: ET.Element, shape_id: int, x: float, y: float, width: float, height: float, *, line_color: str = RED, line_width: int = 15240) -> None:
    shape, sp_pr = nv_sp_pr(sp_tree, shape_id, f"ROI outline {shape_id}")
    add_xfrm(sp_pr, x, y, width, height)
    geom = ET.SubElement(sp_pr, qn(A_NS, "prstGeom"), {"prst": "rect"})
    ET.SubElement(geom, qn(A_NS, "avLst"))
    ET.SubElement(sp_pr, qn(A_NS, "noFill"))
    line = ET.SubElement(sp_pr, qn(A_NS, "ln"), {"w": str(line_width)})
    solid = ET.SubElement(line, qn(A_NS, "solidFill"))
    ET.SubElement(solid, qn(A_NS, "srgbClr"), {"val": line_color})
    ET.SubElement(line, qn(A_NS, "prstDash"), {"val": "solid"})


def add_picture(sp_tree: ET.Element, shape_id: int, x: float, y: float, width: float, height: float, rel_id: str, name: str) -> None:
    pic = ET.SubElement(sp_tree, qn(P_NS, "pic"))
    nv = ET.SubElement(pic, qn(P_NS, "nvPicPr"))
    ET.SubElement(nv, qn(P_NS, "cNvPr"), {"id": str(shape_id), "name": name})
    ET.SubElement(nv, qn(P_NS, "cNvPicPr"), {"preferRelativeResize": "0"})
    ET.SubElement(nv, qn(P_NS, "nvPr"))
    blip = ET.SubElement(pic, qn(P_NS, "blipFill"))
    ET.SubElement(blip, qn(A_NS, "blip"), {qn(R_NS, "embed"): rel_id})
    stretch = ET.SubElement(blip, qn(A_NS, "stretch"))
    ET.SubElement(stretch, qn(A_NS, "fillRect"))
    sp_pr = ET.SubElement(pic, qn(P_NS, "spPr"))
    add_xfrm(sp_pr, x, y, width, height)
    geom = ET.SubElement(sp_pr, qn(A_NS, "prstGeom"), {"prst": "rect"})
    ET.SubElement(geom, qn(A_NS, "avLst"))


def rels_xml(relationships: Iterable[tuple[str, str, str]]) -> bytes:
    root = ET.Element(qn(PKG_REL_NS, "Relationships"))
    for rel_id, rel_type, target in relationships:
        ET.SubElement(root, qn(PKG_REL_NS, "Relationship"), {"Id": rel_id, "Type": rel_type, "Target": target})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def slide_xml(sp_tree: ET.Element) -> bytes:
    root = ET.Element(qn(P_NS, "sld"), {"showMasterSp": "1", "showMasterPhAnim": "1"})
    c_sld = ET.SubElement(root, qn(P_NS, "cSld"))
    c_sld.append(sp_tree)
    clr = ET.SubElement(root, qn(P_NS, "clrMapOvr"))
    ET.SubElement(clr, qn(A_NS, "masterClrMapping"))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def group_tree() -> ET.Element:
    tree = ET.Element(qn(P_NS, "spTree"))
    nv = ET.SubElement(tree, qn(P_NS, "nvGrpSpPr"))
    ET.SubElement(nv, qn(P_NS, "cNvPr"), {"id": "1", "name": ""})
    ET.SubElement(nv, qn(P_NS, "cNvGrpSpPr"))
    ET.SubElement(nv, qn(P_NS, "nvPr"))
    ET.SubElement(tree, qn(P_NS, "grpSpPr"))
    return tree


def make_pptx(
    output: Path,
    selected: tuple[int, int],
    methods: tuple[str, ...],
    assets: dict[tuple[int, str, str], bytes],
    records: dict[int, dict[str, float | tuple[int, int, int, int]]],
    psnr_values: dict[int, dict[str, float]],
) -> None:
    spec = layout(len(methods))
    tree = group_tree()
    relationships: list[tuple[str, str, str]] = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]
    media: dict[str, bytes] = {}
    next_rel = 2
    next_shape = 2
    image = float(spec["image"])
    xs = list(spec["xs"])
    full_y = list(spec["full_y"])
    zoom_y = list(spec["zoom_y"])

    for column, method in enumerate(methods):
        add_text_box(tree, next_shape, xs[column], 6.96, image, 0.42, method, size=10, bold=True)
        next_shape += 1

    for row, index in enumerate(selected):
        sample_center = (full_y[row] + image / 2 + zoom_y[row] + image / 2) / 2
        add_text_box(tree, next_shape, 0.05, sample_center - 0.50, 0.42, 1.00, f"Sample {row + 1}", size=10, bold=True, rotate=1)
        next_shape += 1
        add_text_box(tree, next_shape, 0.50, full_y[row] + 0.30, 0.20, 0.55, "full", size=7, color=MUTED, rotate=1)
        next_shape += 1
        add_text_box(tree, next_shape, 0.42, zoom_y[row] + 0.22, 0.28, 0.85, "4× zoom", size=7, color=MUTED, rotate=1)
        next_shape += 1
        roi = records[index]["roi"]
        assert isinstance(roi, tuple)
        rx, ry, rw, rh = roi
        for column, method in enumerate(methods):
            for kind, y_pos in (("full", full_y[row]), ("zoom", zoom_y[row])):
                media_name = f"image_{len(media) + 1}.png"
                media[media_name] = assets[(index, method, kind)]
                rel_id = f"rId{next_rel}"
                next_rel += 1
                relationships.append((rel_id, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{media_name}"))
                add_picture(tree, next_shape, xs[column], y_pos, image, image, rel_id, f"{method} {kind} sample {row + 1}")
                next_shape += 1
            add_rect(tree, next_shape, xs[column] + image * rx / 128.0, full_y[row] + image * ry / 128.0, image * rw / 128.0, image * rh / 128.0)
            next_shape += 1
            if method != "Original":
                add_text_box(tree, next_shape, xs[column], full_y[row] - 0.19, image, 0.17, f"{psnr_values[index][method]:.2f} dB", size=7, color=MUTED)
                next_shape += 1

    add_text_box(tree, next_shape, 0.90, 0.12, SLIDE_W - 1.82, 0.34, "Red boxes indicate zoomed regions. Ours-base denotes the MBRS global continuation, and Ours denotes Hard Top10.", size=8, color=MUTED)

    slide_rels = rels_xml(relationships)
    slide = slide_xml(tree)
    content_types: list[tuple[str, str, str]] = []
    content_types.extend([(f"/ppt/media/{name}", "image/png", "override") for name in media])
    content_types.extend([
        ("/ppt/presentation.xml", "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml", "override"),
        ("/ppt/slides/slide1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slide+xml", "override"),
        ("/ppt/slideLayouts/slideLayout1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml", "override"),
        ("/ppt/slideMasters/slideMaster1.xml", "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml", "override"),
        ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml", "override"),
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml", "override"),
        ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml", "override"),
    ])

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(content_types))
        archive.writestr("_rels/.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))
        archive.writestr("ppt/presentation.xml", presentation_xml())
        archive.writestr("ppt/_rels/presentation.xml.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"),
            ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", "slides/slide1.xml"),
        ]))
        archive.writestr("ppt/slides/slide1.xml", slide)
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", slide_rels)
        archive.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        archive.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml"),
        ]))
        archive.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        archive.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
        ]))
        archive.writestr("ppt/theme/theme1.xml", theme_xml())
        archive.writestr("docProps/core.xml", core_props_xml())
        archive.writestr("docProps/app.xml", app_props_xml())
        for name, data in media.items():
            archive.writestr(f"ppt/media/{name}", data)


def content_types_xml(items: list[tuple[str, str, str]]) -> bytes:
    root = ET.Element(qn(CT_NS, "Types"))
    ET.SubElement(root, qn(CT_NS, "Default"), {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(root, qn(CT_NS, "Default"), {"Extension": "xml", "ContentType": "application/xml"})
    seen_defaults = {"rels", "xml"}
    for path, content_type, kind in items:
        if kind == "override":
            ET.SubElement(root, qn(CT_NS, "Override"), {"PartName": path, "ContentType": content_type})
        elif path not in seen_defaults:
            ET.SubElement(root, qn(CT_NS, "Default"), {"Extension": path, "ContentType": content_type})
            seen_defaults.add(path)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def presentation_xml() -> bytes:
    root = ET.Element(qn(P_NS, "presentation"))
    masters = ET.SubElement(root, qn(P_NS, "sldMasterIdLst"))
    ET.SubElement(masters, qn(P_NS, "sldMasterId"), {"id": "2147483648", qn(R_NS, "id"): "rId1"})
    slides = ET.SubElement(root, qn(P_NS, "sldIdLst"))
    ET.SubElement(slides, qn(P_NS, "sldId"), {"id": "256", qn(R_NS, "id"): "rId2"})
    ET.SubElement(root, qn(P_NS, "sldSz"), {"cx": str(emu(SLIDE_W)), "cy": str(emu(SLIDE_H)), "type": "screen16x9"})
    ET.SubElement(root, qn(P_NS, "notesSz"), {"cx": str(emu(10)), "cy": str(emu(7.5))})
    ET.SubElement(root, qn(P_NS, "defaultTextStyle"))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def slide_layout_xml() -> bytes:
    root = ET.Element(qn(P_NS, "sldLayout"), {"matchingName": "Blank", "type": "blank", "preserve": "1"})
    c_sld = ET.SubElement(root, qn(P_NS, "cSld"))
    c_sld.append(group_tree())
    clr = ET.SubElement(root, qn(P_NS, "clrMapOvr"))
    ET.SubElement(clr, qn(A_NS, "masterClrMapping"))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def slide_master_xml() -> bytes:
    root = ET.Element(qn(P_NS, "sldMaster"))
    c_sld = ET.SubElement(root, qn(P_NS, "cSld"))
    c_sld.append(group_tree())
    ET.SubElement(root, qn(P_NS, "clrMap"), {"bg1": "lt1", "tx1": "dk1", "bg2": "lt2", "tx2": "dk2", "accent1": "accent1", "accent2": "accent2", "accent3": "accent3", "accent4": "accent4", "accent5": "accent5", "accent6": "accent6", "hlink": "hlink", "folHlink": "folHlink"})
    ET.SubElement(root, qn(P_NS, "sldLayoutIdLst"))
    ET.SubElement(root, qn(P_NS, "txStyles"))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def theme_xml() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="MBRS Figure Theme">
  <a:themeElements>
    <a:clrScheme name="MBRS"><a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="202934"/></a:dk2><a:lt2><a:srgbClr val="F5F7F9"/></a:lt2><a:accent1><a:srgbClr val="3D78A8"/></a:accent1><a:accent2><a:srgbClr val="C66A2C"/></a:accent2><a:accent3><a:srgbClr val="4B8963"/></a:accent3><a:accent4><a:srgbClr val="806BB3"/></a:accent4><a:accent5><a:srgbClr val="E5483C"/></a:accent5><a:accent6><a:srgbClr val="66727D"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="MBRS"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="MBRS"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>
  </a:themeElements>
</a:theme>'''


def core_props_xml() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Figure 2 Qualitative Comparison</dc:title><dc:subject>Editable MBRS qualitative comparison</dc:subject><dc:creator>MBRS paper renderer</dc:creator></cp:coreProperties>'''


def app_props_xml() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>MBRS paper renderer</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>1</Slides></Properties>'''


def load_psnr() -> dict[int, dict[str, float]]:
    import csv

    rows = list(csv.DictReader((QUALITATIVE_DIR / "per_image_psnr.csv").open(newline="")))
    return {
        int(row["validation_index"]): {
            "HiDDeN-64": float(row["HiDDeN-64_psnr_db"]),
            "MaskWM-D_64": float(row["MaskWM-D_64_psnr_db"]),
            "Ours-base": float(row["MBRS Global_psnr_db"]),
            "Ours": float(row["Hard Top10_psnr_db"]),
            "Ours + OKLab": float(row["Hard Top10 + OKLab_psnr_db"]),
        }
        for row in rows
    }


def render() -> None:
    manifest = torch.load(VALIDATION_MANIFEST, map_location="cpu", weights_only=False)
    original = manifest["images"].float()
    global_output = load_encoded(MBRS_DIR / OUTPUT_NAMES["Ours-base"])
    hard_output = load_encoded(MBRS_DIR / OUTPUT_NAMES["Ours"])
    frozen_main = Image.open(FROZEN_MAIN).convert("RGB")
    frozen_zoom = Image.open(FROZEN_ZOOM).convert("RGB")
    selected, records = select_samples(original, global_output, hard_output, frozen_main)
    assets = build_assets(selected, frozen_main, frozen_zoom)
    psnr_values = load_psnr()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    preview = OUTPUT_DIR / "figure2_qualitative_comparison.png"
    preview_pdf = OUTPUT_DIR / "figure2_qualitative_comparison.pdf"
    render_preview(selected, METHODS_6, assets, records, psnr_values, preview)
    # Matplotlib chooses the format by extension; render the same editable-layout preview as PDF.
    render_preview(selected, METHODS_6, assets, records, psnr_values, preview_pdf)
    make_pptx(OUTPUT_DIR / "figure2_qualitative_comparison.pptx", selected, METHODS_6, assets, records, psnr_values)
    make_pptx(OUTPUT_DIR / "figure2_v1_5col.pptx", selected, METHODS_5, assets, records, psnr_values)
    make_pptx(OUTPUT_DIR / "figure2_v2_6col.pptx", selected, METHODS_6, assets, records, psnr_values)

    provenance = {
        "selected_validation_indices": list(selected),
        "sample_labels": {"Sample 1": selected[0], "Sample 2": selected[1]},
        "method_columns": {method: METHOD_SOURCE_NAMES[method] for method in METHODS_6},
        "selection_rule": "Among fixed indices 0,10,20,30,40, require positive ROI MSE improvement of Ours over Ours-base; rank 0.65 normalized local improvement + 0.35 normalized mean external-baseline local error.",
        "selection_records": {str(index): record for index, record in records.items()},
        "roi_rule": "Native 32x32 block containing the maximum CIEDE2000 pixel of the saved Hard Top10 output; shared by all methods for each sample.",
        "zoom": "The same 32x32 native ROI is displayed at 4x relative to the 128x128 source view using nearest-neighbor enlargement.",
        "sources": {
            "manifest": str(VALIDATION_MANIFEST),
            "frozen_main_montage": str(FROZEN_MAIN),
            "frozen_zoom_montage": str(FROZEN_ZOOM),
            "mbrs_outputs": str(MBRS_DIR),
            "qualitative_psnr": str(QUALITATIVE_DIR / "per_image_psnr.csv"),
        },
        "source_sha256": {
            str(VALIDATION_MANIFEST): sha256(VALIDATION_MANIFEST),
            str(FROZEN_MAIN): sha256(FROZEN_MAIN),
            str(FROZEN_ZOOM): sha256(FROZEN_ZOOM),
            str(MBRS_DIR / OUTPUT_NAMES["Ours-base"]): sha256(MBRS_DIR / OUTPUT_NAMES["Ours-base"]),
            str(MBRS_DIR / OUTPUT_NAMES["Ours"]): sha256(MBRS_DIR / OUTPUT_NAMES["Ours"]),
            str(MBRS_DIR / OUTPUT_NAMES["Ours + OKLab"]): sha256(MBRS_DIR / OUTPUT_NAMES["Ours + OKLab"]),
            str(QUALITATIVE_DIR / "per_image_psnr.csv"): sha256(QUALITATIVE_DIR / "per_image_psnr.csv"),
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (
                preview,
                preview_pdf,
                OUTPUT_DIR / "figure2_qualitative_comparison.pptx",
                OUTPUT_DIR / "figure2_v1_5col.pptx",
                OUTPUT_DIR / "figure2_v2_6col.pptx",
            )
        },
    }
    (OUTPUT_DIR / "figure2_qualitative_comparison_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Figure 2 Editable Qualitative Comparison Check",
        "",
        f"- Selected validation indices: `{selected[0]}`, `{selected[1]}` mapped to `Sample 1`, `Sample 2`.",
        "- Columns: `Original`, `HiDDeN-64`, `MaskWM-D_64`, `Ours-base`, `Ours`, `Ours + OKLab`.",
        "- `Ours-base` is the saved MBRS Global continuation; `Ours` is the saved Hard Top10 output.",
        "- Each sample has a full-image row and a shared-ROI 4x zoom row; the red ROI rectangles are separate editable vector objects in PPTX.",
        "- Original has no PSNR label; all watermarked columns retain the existing per-image PSNR values below the full-image row.",
        "- PPTX objects: 24 separate image objects for the six-column version, editable text boxes, and editable red ROI shapes.",
        "- No training or checkpoint inference was run; the visual columns are lossless crops from the existing frozen qualitative montages, and only saved MBRS outputs were read for sample selection.",
        "",
        "## Outputs",
        "",
        f"- `{preview}`",
        f"- `{preview_pdf}`",
        f"- `{OUTPUT_DIR / 'figure2_qualitative_comparison.pptx'}`",
        f"- `{OUTPUT_DIR / 'figure2_v1_5col.pptx'}`",
        f"- `{OUTPUT_DIR / 'figure2_v2_6col.pptx'}`",
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(preview)
    print(preview_pdf)
    print(OUTPUT_DIR / "figure2_qualitative_comparison.pptx")
    print(OUTPUT_DIR / "figure2_v1_5col.pptx")
    print(OUTPUT_DIR / "figure2_v2_6col.pptx")


if __name__ == "__main__":
    render()
