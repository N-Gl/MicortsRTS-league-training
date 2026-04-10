#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
P14_NS = "http://schemas.microsoft.com/office/powerpoint/2010/main"

ET.register_namespace("p", P_NS)
ET.register_namespace("a", A_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("p14", P14_NS)
ET.register_namespace("", CT_NS)


THESIS_DIR = Path(__file__).resolve().parent
SOURCE_TEX = THESIS_DIR / "BA_presentation.tex"
TEMP_SOURCE_TEX = THESIS_DIR / "BA_presentation_editable_source.tex"
TEMP_SOURCE_PDF = THESIS_DIR / "BA_presentation_editable_source.pdf"
OUTPUT_PPTX = THESIS_DIR / "BA_presentation_editable_embedded_videos.pptx"

EMU_PER_INCH = 914400
SLIDE_COUNT = 23

HHU_BLUE = "003D7C"
HHU_LIGHT = "EBF2F9"
ALERT_RED = "D10000"
ALERT_LIGHT = "F7E2E2"
SHADOW = "B5B5B5"
BLACK = "000000"
WHITE = "FFFFFF"
BLUE_BUTTON = "93A9C9"

VIDEO_FILES = {
    4: THESIS_DIR / "images" / "MicroRTS_example.mp4",
    17: THESIS_DIR / "images" / "MictoRTS_BaseAgent_vs_BaseAgent_almost_100.mp4",
    18: THESIS_DIR / "images" / "MictoRTS_BaseAgent_vs_BaseAgent_edited.mp4",
}


def qname(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def run(cmd: Iterable[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def inch(value: float) -> int:
    return round(value * EMU_PER_INCH)


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"expected snippet not found:\n{old}")
    return text.replace(old, new, 1)


def build_temp_source() -> None:
    text = SOURCE_TEX.read_text(encoding="utf-8")
    text = replace_once(text, r"\usepackage{movie15}", r"% \usepackage{movie15}")

    old_slide4 = """            \\movie[showcontrols]{
                \\includegraphics[width=0.94\\linewidth]{initial_pos_basesWorkers16x16A.png}
            }{images/MicroRTS_example.mp4}
            \\vspace{0.2em}

            \\scriptsize Click the play area in Foxit Reader to start the embedded MicroRTS example video."""

    new_slide4 = """            \\href{run:images/MicroRTS_example.mp4}{%
                \\includegraphics[width=0.94\\linewidth]{initial_pos_basesWorkers16x16A.png}%
            }
            \\vspace{0.2em}

            \\scriptsize Initial state of the \\texttt{basesWorkers16x16A} map."""

    text = replace_once(text, old_slide4, new_slide4)

    old_slide18 = """    \\begin{center}
        \\movie[externalviewer]{
            \\fbox{
                \\parbox[c][0.52\\textheight][c]{0.82\\linewidth}{
                    \\centering
                    \\Large Base Agent vs Base Agent\\\\[0.9em]
                    \\normalsize Video example for the Base Agent's preferred strategy\\\\[1.2em]
                    \\beamergotobutton{Open Video}
                }
            }
        }{images/MictoRTS_BaseAgent_vs_BaseAgent_edited.mp4}
    \\end{center}"""

    new_slide18 = """    \\begin{center}
        \\href{run:images/MictoRTS_BaseAgent_vs_BaseAgent_edited.mp4}{%
            \\fbox{
                \\parbox[c][0.52\\textheight][c]{0.82\\linewidth}{
                    \\centering
                    \\Large Base Agent vs Base Agent\\\\[0.9em]
                    \\normalsize Video example for the Base Agent's preferred strategy\\\\[1.2em]
                    \\beamergotobutton{Open Video}
                }
            }
        }
    \\end{center}"""

    text = replace_once(text, old_slide18, new_slide18)
    TEMP_SOURCE_TEX.write_text(text, encoding="utf-8")


def cleanup_temp_source() -> None:
    suffixes = [
        ".tex",
        ".pdf",
        ".aux",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".nav",
        ".out",
        ".snm",
        ".toc",
    ]
    for suffix in suffixes:
        path = THESIS_DIR / f"BA_presentation_editable_source{suffix}"
        if path.exists():
            path.unlink()


def compile_temp_pdf() -> None:
    run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", TEMP_SOURCE_TEX.name],
        cwd=THESIS_DIR,
    )


def parse_pdfinfo(pdf_path: Path) -> tuple[int, float, float]:
    info = run(["pdfinfo", str(pdf_path)])
    pages_match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    size_match = re.search(
        r"^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts", info, flags=re.MULTILINE
    )
    if not pages_match or not size_match:
        raise RuntimeError("unable to parse pdfinfo output")
    return int(pages_match.group(1)), float(size_match.group(1)), float(size_match.group(2))


def page_object_map(pdf_path: Path) -> dict[int, int]:
    output = run(["mutool", "show", str(pdf_path), "pages"])
    mapping: dict[int, int] = {}
    for line in output.splitlines():
        match = re.match(r"page\s+(\d+)\s+=\s+(\d+)\s+0\s+R", line.strip())
        if match:
            mapping[int(match.group(1))] = int(match.group(2))
    if not mapping:
        raise RuntimeError("unable to parse mutool page listing")
    return mapping


def largest_annotation_rect(pdf_path: Path, page_obj_num: int) -> tuple[float, float, float, float]:
    annots_text = run(["mutool", "show", str(pdf_path), f"{page_obj_num}/Annots"])
    annot_ids = [int(match) for match in re.findall(r"(\d+)\s+0\s+R", annots_text)]
    rects = []
    for annot_id in annot_ids:
        annot_text = run(["mutool", "show", str(pdf_path), str(annot_id)])
        match = re.search(
            r"/Rect\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\]",
            annot_text,
        )
        if match:
            rect = tuple(float(match.group(i)) for i in range(1, 5))
            rects.append(rect)
    if not rects:
        raise RuntimeError(f"no annotations found on page object {page_obj_num}")
    return max(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))


def pdf_rect_to_pixels(
    rect: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    width_px: int,
    height_px: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    scale_x = width_px / page_width
    scale_y = height_px / page_height
    x = round(x1 * scale_x)
    y = round((page_height - y2) * scale_y)
    w = round((x2 - x1) * scale_x)
    h = round((y2 - y1) * scale_y)
    return x, y, w, h


def render_poster_crop(
    pdf_path: Path,
    page_num: int,
    crop: tuple[int, int, int, int],
    out_prefix: Path,
    width_px: int = 1920,
    height_px: int = 1080,
) -> Path:
    x, y, w, h = crop
    run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-scale-to-x",
            str(width_px),
            "-scale-to-y",
            str(height_px),
            "-x",
            str(x),
            "-y",
            str(y),
            "-W",
            str(w),
            "-H",
            str(h),
            str(pdf_path),
            str(out_prefix),
        ]
    )
    return out_prefix.with_suffix(".png")


def render_pdf_to_png(pdf_path: Path, out_prefix: Path, dpi: int = 220) -> Path:
    run(["pdftoppm", "-png", "-singlefile", "-r", str(dpi), str(pdf_path), str(out_prefix)])
    return out_prefix.with_suffix(".png")


def build_base_pptx(slide_count: int, out_path: Path) -> None:
    md_lines = []
    for index in range(1, slide_count + 1):
        md_lines.append(f"# Slide {index}\n\n.\n")
    markdown = "\n".join(md_lines)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(markdown, encoding="utf-8")
    try:
        run(["pandoc", str(md_path), "-t", "pptx", "-o", str(out_path)])
    finally:
        if md_path.exists():
            md_path.unlink()


def relationship_element(rel_id: str, rel_type: str, target: str) -> ET.Element:
    return ET.Element(
        qname(REL_NS, "Relationship"),
        {"Id": rel_id, "Type": rel_type, "Target": target},
    )


def ensure_content_type_default(content_types_path: Path, extension: str, content_type: str) -> None:
    tree = ET.parse(content_types_path)
    root = tree.getroot()
    existing = {
        elem.attrib.get("Extension", "").lower(): elem
        for elem in root.findall(qname(CT_NS, "Default"))
    }
    if extension.lower() not in existing:
        ET.SubElement(
            root,
            qname(CT_NS, "Default"),
            {"Extension": extension, "ContentType": content_type},
        )
        tree.write(content_types_path, encoding="utf-8", xml_declaration=True)


def zip_dir(source_dir: Path, out_file: Path) -> None:
    with zipfile.ZipFile(out_file, "w") as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            rel_path = path.relative_to(source_dir).as_posix()
            compress_type = zipfile.ZIP_STORED if path.suffix.lower() == ".mp4" else zipfile.ZIP_DEFLATED
            zf.write(path, rel_path, compress_type=compress_type)


def validate_pptx(pptx_path: Path) -> None:
    with zipfile.ZipFile(pptx_path) as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"corrupt zip entry: {bad_file}")
        for name in zf.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(zf.read(name))


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        signature = fh.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"unsupported image format for {path}")
        ihdr_length = struct.unpack(">I", fh.read(4))[0]
        chunk_type = fh.read(4)
        if ihdr_length != 13 or chunk_type != b"IHDR":
            raise RuntimeError(f"invalid PNG header for {path}")
        width, height = struct.unpack(">II", fh.read(8))
        return width, height


def fit_contain(path: Path, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    img_w, img_h = png_size(path)
    scale = min(w / img_w, h / img_h)
    new_w = img_w * scale
    new_h = img_h * scale
    new_x = x + (w - new_w) / 2
    new_y = y + (h - new_h) / 2
    return new_x, new_y, new_w, new_h


@dataclass
class Paragraph:
    text: str
    size: float = 18.0
    color: str = BLACK
    bold: bool = False
    italic: bool = False
    bullet: bool = False
    level: int = 0
    align: str = "l"


def srgb(parent: ET.Element, color: str) -> None:
    ET.SubElement(parent, qname(A_NS, "srgbClr"), {"val": color})


def add_solid_fill(parent: ET.Element, color: str) -> None:
    fill = ET.SubElement(parent, qname(A_NS, "solidFill"))
    srgb(fill, color)


def add_line(parent: ET.Element, color: str | None, width: float = 1.0) -> None:
    line = ET.SubElement(parent, qname(A_NS, "ln"), {"w": str(max(1, round(width * 12700)))})
    if color is None:
        ET.SubElement(line, qname(A_NS, "noFill"))
    else:
        add_solid_fill(line, color)


def text_body(
    paragraphs: list[Paragraph],
    *,
    inset_left: float = 0.12,
    inset_right: float = 0.12,
    inset_top: float = 0.08,
    inset_bottom: float = 0.05,
    anchor: str = "t",
) -> ET.Element:
    tx_body = ET.Element(qname(P_NS, "txBody"))
    body_pr = ET.SubElement(
        tx_body,
        qname(A_NS, "bodyPr"),
        {
            "wrap": "square",
            "lIns": str(inch(inset_left)),
            "rIns": str(inch(inset_right)),
            "tIns": str(inch(inset_top)),
            "bIns": str(inch(inset_bottom)),
            "anchor": anchor,
        },
    )
    ET.SubElement(body_pr, qname(A_NS, "spAutoFit"))
    ET.SubElement(tx_body, qname(A_NS, "lstStyle"))

    for para in paragraphs:
        p = ET.SubElement(tx_body, qname(A_NS, "p"))
        p_pr_attrs: dict[str, str] = {}
        if para.align != "l":
            p_pr_attrs["algn"] = para.align
        if para.bullet:
            p_pr_attrs["lvl"] = str(para.level)
            p_pr_attrs["marL"] = str(inch(0.28 + para.level * 0.24))
            p_pr_attrs["indent"] = str(inch(-0.18))
        else:
            p_pr_attrs["marL"] = "0"
            p_pr_attrs["indent"] = "0"
        p_pr = ET.SubElement(p, qname(A_NS, "pPr"), p_pr_attrs)

        if para.bullet:
            bu_clr = ET.SubElement(p_pr, qname(A_NS, "buClr"))
            srgb(bu_clr, HHU_BLUE)
            ET.SubElement(p_pr, qname(A_NS, "buFont"), {"typeface": "Arial"})
            ET.SubElement(p_pr, qname(A_NS, "buChar"), {"char": "•"})
        else:
            ET.SubElement(p_pr, qname(A_NS, "buNone"))

        r = ET.SubElement(p, qname(A_NS, "r"))
        r_pr_attrs = {"lang": "en-US", "sz": str(round(para.size * 100))}
        if para.bold:
            r_pr_attrs["b"] = "1"
        if para.italic:
            r_pr_attrs["i"] = "1"
        r_pr = ET.SubElement(r, qname(A_NS, "rPr"), r_pr_attrs)
        fill = ET.SubElement(r_pr, qname(A_NS, "solidFill"))
        srgb(fill, para.color)

        parts = para.text.split("\n")
        for index, part in enumerate(parts):
            ET.SubElement(r, qname(A_NS, "t")).text = part
            if index != len(parts) - 1:
                ET.SubElement(p, qname(A_NS, "br"))
                r = ET.SubElement(p, qname(A_NS, "r"))
                r_pr = ET.SubElement(r, qname(A_NS, "rPr"), r_pr_attrs)
                fill = ET.SubElement(r_pr, qname(A_NS, "solidFill"))
                srgb(fill, para.color)

        end_pr_attrs = {"lang": "en-US", "sz": str(round(para.size * 100))}
        if para.bold:
            end_pr_attrs["b"] = "1"
        ET.SubElement(p, qname(A_NS, "endParaRPr"), end_pr_attrs)

    return tx_body


class SlideBuilder:
    def __init__(self, slide_num: int, total_slides: int, media_dir: Path, title: str | None) -> None:
        self.slide_num = slide_num
        self.total_slides = total_slides
        self.media_dir = media_dir
        self.shape_id = 2
        self.rel_index = 2
        self.media_index = 1
        self.relationships: list[ET.Element] = []

        self.root = ET.Element(qname(P_NS, "sld"))
        c_sld = ET.SubElement(self.root, qname(P_NS, "cSld"))
        self.sp_tree = ET.SubElement(c_sld, qname(P_NS, "spTree"))

        nv_grp_sp_pr = ET.SubElement(self.sp_tree, qname(P_NS, "nvGrpSpPr"))
        ET.SubElement(nv_grp_sp_pr, qname(P_NS, "cNvPr"), {"id": "1", "name": ""})
        ET.SubElement(nv_grp_sp_pr, qname(P_NS, "cNvGrpSpPr"))
        ET.SubElement(nv_grp_sp_pr, qname(P_NS, "nvPr"))

        grp_sp_pr = ET.SubElement(self.sp_tree, qname(P_NS, "grpSpPr"))
        xfrm = ET.SubElement(grp_sp_pr, qname(A_NS, "xfrm"))
        ET.SubElement(xfrm, qname(A_NS, "off"), {"x": "0", "y": "0"})
        ET.SubElement(xfrm, qname(A_NS, "ext"), {"cx": "0", "cy": "0"})
        ET.SubElement(xfrm, qname(A_NS, "chOff"), {"x": "0", "y": "0"})
        ET.SubElement(xfrm, qname(A_NS, "chExt"), {"cx": "0", "cy": "0"})

        if title is not None:
            self.add_shape(
                0.0,
                0.0,
                13.333,
                0.56,
                geom="rect",
                fill=HHU_LIGHT,
                line=None,
            )
            self.add_textbox(
                0.18,
                0.06,
                10.7,
                0.34,
                [Paragraph(title, size=23, color=HHU_BLUE)],
                inset_left=0.0,
                inset_top=0.0,
                inset_right=0.0,
                inset_bottom=0.0,
            )

        self.add_textbox(
            12.85,
            7.12,
            0.38,
            0.16,
            [Paragraph(f"{slide_num} / {total_slides}", size=8.5, color=BLACK, align="r")],
            inset_left=0.0,
            inset_top=0.0,
            inset_right=0.0,
            inset_bottom=0.0,
        )

    def next_shape_name(self, prefix: str) -> tuple[int, str]:
        shape_id = self.shape_id
        self.shape_id += 1
        return shape_id, f"{prefix} {shape_id}"

    def copy_media(self, src: Path, suffix: str | None = None) -> str:
        ext = suffix or src.suffix
        filename = f"slide{self.slide_num:02d}-media{self.media_index}{ext}"
        self.media_index += 1
        shutil.copy2(src, self.media_dir / filename)
        return filename

    def new_rel(self, rel_type: str, target: str) -> str:
        rel_id = f"rId{self.rel_index}"
        self.rel_index += 1
        self.relationships.append(relationship_element(rel_id, rel_type, target))
        return rel_id

    def add_shape(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        geom: str = "rect",
        fill: str | None = None,
        line: str | None = None,
        line_width: float = 1.0,
        paragraphs: list[Paragraph] | None = None,
        inset_left: float = 0.12,
        inset_right: float = 0.12,
        inset_top: float = 0.08,
        inset_bottom: float = 0.05,
        anchor: str = "t",
        name_prefix: str = "Shape",
    ) -> ET.Element:
        shape_id, shape_name = self.next_shape_name(name_prefix)
        sp = ET.SubElement(self.sp_tree, qname(P_NS, "sp"))
        nv_sp_pr = ET.SubElement(sp, qname(P_NS, "nvSpPr"))
        ET.SubElement(nv_sp_pr, qname(P_NS, "cNvPr"), {"id": str(shape_id), "name": shape_name})
        c_nv_sp_pr = ET.SubElement(nv_sp_pr, qname(P_NS, "cNvSpPr"))
        ET.SubElement(c_nv_sp_pr, qname(A_NS, "spLocks"), {"noGrp": "1"})
        ET.SubElement(nv_sp_pr, qname(P_NS, "nvPr"))

        sp_pr = ET.SubElement(sp, qname(P_NS, "spPr"))
        xfrm = ET.SubElement(sp_pr, qname(A_NS, "xfrm"))
        ET.SubElement(xfrm, qname(A_NS, "off"), {"x": str(inch(x)), "y": str(inch(y))})
        ET.SubElement(xfrm, qname(A_NS, "ext"), {"cx": str(inch(w)), "cy": str(inch(h))})

        if fill is None:
            ET.SubElement(sp_pr, qname(A_NS, "noFill"))
        else:
            add_solid_fill(sp_pr, fill)
        add_line(sp_pr, line, line_width)

        prst_geom = ET.SubElement(sp_pr, qname(A_NS, "prstGeom"), {"prst": geom})
        ET.SubElement(prst_geom, qname(A_NS, "avLst"))

        if paragraphs is not None:
            sp.append(
                text_body(
                    paragraphs,
                    inset_left=inset_left,
                    inset_right=inset_right,
                    inset_top=inset_top,
                    inset_bottom=inset_bottom,
                    anchor=anchor,
                )
            )
        return sp

    def add_textbox(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        paragraphs: list[Paragraph],
        *,
        inset_left: float = 0.05,
        inset_right: float = 0.05,
        inset_top: float = 0.02,
        inset_bottom: float = 0.02,
        anchor: str = "t",
    ) -> ET.Element:
        return self.add_shape(
            x,
            y,
            w,
            h,
            geom="rect",
            fill=None,
            line=None,
            paragraphs=paragraphs,
            inset_left=inset_left,
            inset_right=inset_right,
            inset_top=inset_top,
            inset_bottom=inset_bottom,
            anchor=anchor,
            name_prefix="Text",
        )

    def add_rule(self, x: float, y: float, w: float, h: float = 0.012, color: str = BLACK) -> None:
        self.add_shape(x, y, w, h, geom="rect", fill=color, line=None, name_prefix="Rule")

    def add_picture(self, path: Path, x: float, y: float, w: float, h: float, *, name: str = "Picture") -> None:
        x_fit, y_fit, w_fit, h_fit = fit_contain(path, x, y, w, h)
        media_name = self.copy_media(path)
        rel_id = self.new_rel(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            f"../media/{media_name}",
        )
        shape_id, shape_name = self.next_shape_name(name)

        pic = ET.SubElement(self.sp_tree, qname(P_NS, "pic"))
        nv_pic_pr = ET.SubElement(pic, qname(P_NS, "nvPicPr"))
        ET.SubElement(nv_pic_pr, qname(P_NS, "cNvPr"), {"id": str(shape_id), "name": shape_name})
        c_nv_pic_pr = ET.SubElement(nv_pic_pr, qname(P_NS, "cNvPicPr"))
        ET.SubElement(c_nv_pic_pr, qname(A_NS, "picLocks"), {"noChangeAspect": "1"})
        ET.SubElement(nv_pic_pr, qname(P_NS, "nvPr"))

        blip_fill = ET.SubElement(pic, qname(P_NS, "blipFill"))
        ET.SubElement(blip_fill, qname(A_NS, "blip"), {qname(R_NS, "embed"): rel_id})
        stretch = ET.SubElement(blip_fill, qname(A_NS, "stretch"))
        ET.SubElement(stretch, qname(A_NS, "fillRect"))

        sp_pr = ET.SubElement(pic, qname(P_NS, "spPr"))
        xfrm = ET.SubElement(sp_pr, qname(A_NS, "xfrm"))
        ET.SubElement(xfrm, qname(A_NS, "off"), {"x": str(inch(x_fit)), "y": str(inch(y_fit))})
        ET.SubElement(xfrm, qname(A_NS, "ext"), {"cx": str(inch(w_fit)), "cy": str(inch(h_fit))})
        prst_geom = ET.SubElement(sp_pr, qname(A_NS, "prstGeom"), {"prst": "rect"})
        ET.SubElement(prst_geom, qname(A_NS, "avLst"))

    def add_video_emu(self, video_path: Path, poster_path: Path, box: tuple[int, int, int, int]) -> None:
        media_name = self.copy_media(video_path)
        poster_name = self.copy_media(poster_path)

        video_rel_id = self.new_rel(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/video",
            f"../media/{media_name}",
        )
        media_rel_id = self.new_rel(
            "http://schemas.microsoft.com/office/2007/relationships/media",
            f"../media/{media_name}",
        )
        poster_rel_id = self.new_rel(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            f"../media/{poster_name}",
        )

        shape_id, shape_name = self.next_shape_name("Video")
        x, y, cx, cy = box

        pic = ET.SubElement(self.sp_tree, qname(P_NS, "pic"))
        nv_pic_pr = ET.SubElement(pic, qname(P_NS, "nvPicPr"))
        c_nv_pr = ET.SubElement(nv_pic_pr, qname(P_NS, "cNvPr"), {"id": str(shape_id), "name": shape_name})
        ET.SubElement(
            c_nv_pr,
            qname(A_NS, "hlinkClick"),
            {qname(R_NS, "id"): "", "action": "ppaction://media"},
        )
        c_nv_pic_pr = ET.SubElement(nv_pic_pr, qname(P_NS, "cNvPicPr"))
        ET.SubElement(c_nv_pic_pr, qname(A_NS, "picLocks"), {"noChangeAspect": "1"})

        nv_pr = ET.SubElement(nv_pic_pr, qname(P_NS, "nvPr"))
        ET.SubElement(nv_pr, qname(A_NS, "videoFile"), {qname(R_NS, "link"): video_rel_id})
        ext_lst = ET.SubElement(nv_pr, qname(P_NS, "extLst"))
        ext = ET.SubElement(
            ext_lst,
            qname(P_NS, "ext"),
            {"uri": "{DAA4B4D4-6D71-4841-9C94-3DE7FCFB9230}"},
        )
        ET.SubElement(ext, qname(P14_NS, "media"), {qname(R_NS, "embed"): media_rel_id})

        blip_fill = ET.SubElement(pic, qname(P_NS, "blipFill"))
        ET.SubElement(blip_fill, qname(A_NS, "blip"), {qname(R_NS, "embed"): poster_rel_id})
        stretch = ET.SubElement(blip_fill, qname(A_NS, "stretch"))
        ET.SubElement(stretch, qname(A_NS, "fillRect"))

        sp_pr = ET.SubElement(pic, qname(P_NS, "spPr"))
        xfrm = ET.SubElement(sp_pr, qname(A_NS, "xfrm"))
        ET.SubElement(xfrm, qname(A_NS, "off"), {"x": str(x), "y": str(y)})
        ET.SubElement(xfrm, qname(A_NS, "ext"), {"cx": str(cx), "cy": str(cy)})
        prst_geom = ET.SubElement(sp_pr, qname(A_NS, "prstGeom"), {"prst": "rect"})
        ET.SubElement(prst_geom, qname(A_NS, "avLst"))

    def add_block(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: list[Paragraph],
        *,
        title_fill: str = HHU_BLUE,
        body_fill: str = HHU_LIGHT,
        title_height: float = 0.42,
    ) -> None:
        self.add_shape(x + 0.07, y + 0.08, w, h, geom="roundRect", fill=SHADOW, line=None, name_prefix="Shadow")
        self.add_shape(x, y, w, h, geom="roundRect", fill=body_fill, line=None, name_prefix="Block")
        self.add_shape(
            x,
            y,
            w,
            title_height,
            geom="roundRect",
            fill=title_fill,
            line=None,
            paragraphs=[Paragraph(title, size=17, color=WHITE)],
            inset_left=0.12,
            inset_top=0.06,
            inset_right=0.1,
            inset_bottom=0.02,
            anchor="ctr",
            name_prefix="BlockTitle",
        )
        self.add_textbox(
            x + 0.12,
            y + title_height + 0.08,
            w - 0.24,
            h - title_height - 0.12,
            body,
            inset_left=0.02,
            inset_top=0.0,
            inset_right=0.02,
            inset_bottom=0.0,
        )

    def write(self, slide_path: Path, rels_path: Path) -> None:
        clr_map_ovr = ET.SubElement(self.root, qname(P_NS, "clrMapOvr"))
        ET.SubElement(clr_map_ovr, qname(A_NS, "masterClrMapping"))
        slide_path.write_bytes(ET.tostring(self.root, encoding="utf-8", xml_declaration=True))

        rel_tree = ET.parse(rels_path)
        rel_root = rel_tree.getroot()
        for rel in self.relationships:
            rel_root.append(rel)
        rel_tree.write(rels_path, encoding="utf-8", xml_declaration=True)


def win_fill(value: float) -> str:
    if value > 94.9:
        return "DBEEDA"
    if value > 84.9:
        return "E8F5E8"
    if value > 69.9:
        return "F7FAE6"
    if value > 49.9:
        return "FCF2E0"
    return "FAE6E6"


def loss_fill(value: float) -> str:
    if value < 2.1:
        return "DBEEDA"
    if value < 8.1:
        return "E8F5E8"
    if value < 18.1:
        return "FCF2E0"
    return "FAE6E6"


def build_title_slide(builder: SlideBuilder, assets: dict[str, Path], _: dict[int, tuple[int, int, int, int]], __: dict[int, Path]) -> None:
    builder.add_shape(0.24, 0.82, 12.9, 1.98, geom="roundRect", fill=SHADOW, line=None, name_prefix="Shadow")
    builder.add_shape(0.18, 0.76, 12.9, 1.98, geom="roundRect", fill=HHU_BLUE, line=None, name_prefix="TitleBar")
    builder.add_textbox(
        0.75,
        1.03,
        11.8,
        1.1,
        [
            Paragraph(
                "Implementation and evaluation of the reinforcement learning methods\nSelf-Play and League Training in the MicroRTS environment",
                size=23,
                color=WHITE,
                align="ctr",
            )
        ],
        inset_left=0.0,
        inset_top=0.0,
        inset_right=0.0,
        inset_bottom=0.0,
    )
    builder.add_textbox(
        3.55,
        2.18,
        6.2,
        0.3,
        [Paragraph("Presentation for the bachelor thesis", size=15, color=WHITE, align="ctr")],
        inset_left=0.0,
        inset_top=0.0,
        inset_right=0.0,
        inset_bottom=0.0,
    )
    builder.add_textbox(0.0, 3.45, 13.333, 0.32, [Paragraph("Niklas Glaser", size=18, align="ctr")])
    builder.add_textbox(
        0.0,
        4.15,
        13.333,
        0.3,
        [Paragraph("Heinrich-Heine-Universitaet Duesseldorf", size=14, align="ctr")],
    )
    builder.add_textbox(0.0, 4.8, 13.333, 0.36, [Paragraph("April 2026", size=20, align="ctr")])
    builder.add_picture(assets["logo"], 5.58, 5.7, 2.25, 0.75, name="Logo")


def build_overview_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    sections = ["Motivation", "Approach", "Experiments", "Results", "Discussion", "Conclusion"]
    y = 1.55
    for index, section in enumerate(sections, start=1):
        builder.add_shape(
            0.2,
            y - 0.05,
            0.3,
            0.3,
            geom="ellipse",
            fill=HHU_BLUE,
            line=None,
            paragraphs=[Paragraph(str(index), size=12, color=WHITE, align="ctr")],
            inset_left=0.0,
            inset_top=0.02,
            inset_right=0.0,
            inset_bottom=0.0,
            anchor="ctr",
            name_prefix="Circle",
        )
        builder.add_textbox(0.62, y - 0.02, 4.0, 0.32, [Paragraph(section, size=18, color=HHU_BLUE)])
        y += 0.81


def build_goals_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_block(
        0.18,
        1.0,
        13.0,
        1.4,
        "Base Agent",
        [
            Paragraph("combination of Behavioral Cloning (BC) and Proximal Policy Optimization (PPO)", size=16.5, bullet=True),
            Paragraph("strong baseline, but weaknesses against several opponents", size=16.5, bullet=True),
        ],
    )
    builder.add_block(
        0.18,
        2.72,
        13.0,
        1.82,
        "Goal",
        [
            Paragraph("improve the Base Agent with League Training (LT)", size=16.5, bullet=True),
            Paragraph("training against a more diverse set of opponents", size=16.5, bullet=True),
            Paragraph("focus on robustness and generalization", size=16.5, bullet=True),
        ],
    )
    builder.add_block(
        0.18,
        4.9,
        13.0,
        1.35,
        "Research Question",
        [
            Paragraph(
                "How much does LT with PPO improve robustness and generalization of the BC/PPO-pretrained agent?",
                size=15.8,
                bullet=True,
            ),
            Paragraph(
                "How strongly does this depend on the diversity of initial agents and BC experts?",
                size=15.8,
                bullet=True,
            ),
        ],
        title_fill=ALERT_RED,
        body_fill=ALERT_LIGHT,
    )


def build_setting_slide(builder: SlideBuilder, _: dict[str, Path], video_boxes: dict[int, tuple[int, int, int, int]], video_posters: dict[int, Path]) -> None:
    builder.add_block(
        0.18,
        1.15,
        6.9,
        2.08,
        "MicroRTS",
        [
            Paragraph("simplified RTS environment for RL research", size=16.5, bullet=True),
            Paragraph("still contains resource management, unit production\nand combat", size=16.5, bullet=True),
            Paragraph("supports controlled and repeatable experiments", size=16.5, bullet=True),
        ],
    )
    builder.add_block(
        0.18,
        3.63,
        6.9,
        2.22,
        "Main RL challenges",
        [
            Paragraph("large combinatorial action space", size=16.5, bullet=True),
            Paragraph("simultaneous decisions", size=16.5, bullet=True),
            Paragraph("delayed and sparse rewards", size=16.5, bullet=True),
            Paragraph("robustness against diverse opponents", size=16.5, bullet=True),
        ],
    )
    builder.add_video_emu(VIDEO_FILES[4], video_posters[4], video_boxes[4])
    box_x, box_y, box_w, box_h = video_boxes[4]
    x = box_x / EMU_PER_INCH
    y = box_y / EMU_PER_INCH
    w = box_w / EMU_PER_INCH
    h = box_h / EMU_PER_INCH
    builder.add_textbox(
        x,
        y + h + 0.08,
        w,
        0.6,
        [Paragraph("Click the play area to start the embedded\nMicroRTS example video.", size=11)],
        inset_left=0.0,
        inset_top=0.0,
        inset_right=0.0,
        inset_bottom=0.0,
    )


def build_pipeline_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(
        1.35,
        1.35,
        10.65,
        0.6,
        [Paragraph("BC -> BC-FT -> PPO -> LT -> PPO fine-tuning", size=23, align="ctr")],
        inset_left=0.0,
        inset_top=0.0,
        inset_right=0.0,
        inset_bottom=0.0,
    )
    builder.add_textbox(
        0.65,
        2.45,
        12.1,
        2.8,
        [
            Paragraph("BC / BC-FT: imitate strong scripted opponents to obtain a competent initial policy", size=16.5, bullet=True),
            Paragraph("PPO: refine the policy through exploration", size=16.5, bullet=True),
            Paragraph("LT: train against a population of evolving opponents instead of only bots or pure self-play", size=16.5, bullet=True),
            Paragraph("Fine-tuning: fix remaining weaknesses after LT, especially against WorkerRushAI", size=16.5, bullet=True),
        ],
    )


def build_from_sp_to_lt_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    rows = [
        ("SP", "train against the current version of the agent; simple, but prone to overfitting or cyclical strategies"),
        ("FSP", "train against a set of past policies; reduces forgetting, but opponent choice is still crude"),
        ("PFSP", "sample opponents based on current win rates; focuses on informative and non-trivial matchups"),
        ("LT", "add role-based agents: Main Agent, Historical Agents, Main Exploiters and League Exploiters"),
    ]
    y = 1.25
    for label, text in rows:
        builder.add_textbox(0.75, y, 1.05, 0.35, [Paragraph(label, size=15.5, bold=True)])
        builder.add_textbox(1.95, y, 10.95, 0.48, [Paragraph(text, size=15.3)])
        y += 0.75
    builder.add_block(
        0.78,
        4.72,
        11.8,
        1.0,
        "Why LT?",
        [Paragraph("LT does not only preserve diversity through historical snapshots, it also creates diversity through exploiters that actively search for weaknesses of the current policy.", size=14.8)],
    )


def build_lt_design_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(
        0.78,
        1.28,
        11.8,
        2.35,
        [
            Paragraph("Main Agent: primary policy", size=17, bullet=True),
            Paragraph("Historical Agents: frozen snapshots for stability", size=17, bullet=True),
            Paragraph("Main Exploiters: exploit weaknesses of historical Main Agents", size=17, bullet=True),
            Paragraph("League Exploiters: search for broader league weaknesses", size=17, bullet=True),
        ],
    )
    builder.add_block(
        0.8,
        4.35,
        11.8,
        1.0,
        "Key idea",
        [Paragraph("LT turns robustness against diverse opponents into part of the optimization target.", size=15.2)],
    )


def build_bot_envs_slide(builder: SlideBuilder, assets: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(
        0.48,
        1.32,
        6.15,
        3.8,
        [
            Paragraph("additional dynamic bot environments (CoacAI and Mayari)", size=16.2, bullet=True),
            Paragraph("scripted bots: CoacAI, Mayari, WorkerRushAI, PassiveAI, LightRushAI, RandomAI and RandomBiasedAI", size=16.2, bullet=True),
            Paragraph("MCTS based bots: Rojo, MixedBot, Izanagi, Tiamat, Droplet, GuidedRojoA3N and NaiveMCTS", size=16.2, bullet=True),
        ],
    )
    builder.add_picture(assets["num_parallel_bot_envs"], 7.25, 1.45, 5.45, 3.85, name="BotEnvs")


def build_observation_slide(builder: SlideBuilder, assets: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_picture(assets["adjust_obs"], 0.55, 0.95, 12.25, 5.7, name="AdjustObs")
    builder.add_textbox(
        1.55,
        6.5,
        10.2,
        0.3,
        [Paragraph("Player 2 observations are transformed so SP remains compatible with a Base Agent only trained as Player 1.", size=11.5, align="ctr")],
    )


def build_matchmaking_slide(builder: SlideBuilder, assets: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_picture(assets["matchmaking"], 0.92, 1.0, 11.48, 5.45, name="Matchmaking")
    builder.add_textbox(
        3.5,
        6.42,
        6.4,
        0.22,
        [Paragraph("Matchmaking and opponent sampling in the LT setup.", size=11.2, align="ctr")],
    )


def build_other_changes_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(
        4.35,
        2.2,
        4.4,
        1.45,
        [
            Paragraph("delta score", size=19, bullet=True),
            Paragraph("annealed entropy coefficient", size=19, bullet=True),
            Paragraph("new initial agents", size=19, bullet=True),
        ],
    )


def build_experiment_setup_slide(builder: SlideBuilder, assets: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(
        0.42,
        1.05,
        6.35,
        4.9,
        [
            Paragraph("map: basesWorkers16x16A", size=15.9, bullet=True),
            Paragraph("LT setup: 1 Main Agent, 4 Main Exploiters, 1 League Exploiter", size=15.9, bullet=True),
            Paragraph("25 environments per learning agent", size=15.9, bullet=True),
            Paragraph("5 to 10 additional dynamic bot environments", size=15.9, bullet=True),
            Paragraph("approximately 130 million LT steps", size=15.9, bullet=True),
            Paragraph("training time: about 100 hours", size=15.9, bullet=True),
            Paragraph("maximum setup size: 160 parallel environments", size=15.9, bullet=True),
            Paragraph("evaluations over at least 100 games against 12 opponents after LT and after fine-tuning", size=15.2, bullet=True),
        ],
    )
    builder.add_picture(assets["bot_win_rates"], 7.15, 1.46, 5.45, 3.9, name="BotRates")


def build_quantitative_slide(builder: SlideBuilder, assets: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_picture(assets["total_winrate"], 0.45, 1.4, 7.1, 4.05, name="TotalWinrate")
    builder.add_textbox(
        8.0,
        2.35,
        4.7,
        0.65,
        [Paragraph("overall loss rate drops from 7.7% to 2.2%", size=18, bullet=True)],
    )


def eval_rows() -> list[tuple[str, str | tuple[float, float, float, float]]]:
    return [
        ("category", "bots used in training"),
        ("CoacAI", (100.0, 96.0, 100.0, 100.0)),
        ("Mayari", (100.0, 67.3, 100.0, 100.0)),
        ("category", "Built-in bots"),
        ("WorkerRushAI", (31.0, 100.0, 35.0, 99.5)),
        ("PassiveAI", (100.0, 99.7, 100.0, 100.0)),
        ("LightRushAI", (100.0, 98.0, 100.0, 100.0)),
        ("RandomAI", (100.0, 99.0, 100.0, 100.0)),
        ("RandomBiasedAI", (91.0, 79.0, 97.0, 97.5)),
        ("category", "Other bots"),
        ("Rojo", (100.0, 81.0, 100.0, 100.0)),
        ("MixedBot", (45.0, 43.3, 57.0, 60.5)),
        ("Izanagi", (68.0, 71.7, 70.0, 72.5)),
        ("Tiamat", (94.0, 93.0, 99.0, 99.0)),
        ("Droplet", (45.0, 67.7, 62.0, 75.5)),
        ("GuidedRojoA3N", (64.0, 74.7, 87.0, 88.5)),
        ("NaiveMCTS", (0.0, 0.7, 2.0, 6.5)),
        ("overall", (74.1, 76.5, 79.2, 85.7)),
    ]


def loss_rows() -> list[tuple[str, str | tuple[float, float, float, float]]]:
    return [
        ("category", "bots used in training"),
        ("CoacAI", (0.0, 2.0, 0.0, 0.0)),
        ("Mayari", (0.0, 27.3, 0.0, 0.0)),
        ("category", "Built-in bots"),
        ("WorkerRushAI", (63.0, 0.0, 55.0, 0.5)),
        ("PassiveAI", (0.0, 0.0, 0.0, 0.0)),
        ("LightRushAI", (0.0, 2.0, 0.0, 0.0)),
        ("RandomAI", (0.0, 0.0, 0.0, 0.0)),
        ("RandomBiasedAI", (0.0, 0.0, 0.0, 0.0)),
        ("category", "Other bots"),
        ("Rojo", (0.0, 5.3, 0.0, 0.0)),
        ("MixedBot", (33.0, 34.3, 20.0, 13.5)),
        ("Izanagi", (15.0, 16.3, 7.0, 6.0)),
        ("Tiamat", (6.0, 7.0, 1.0, 0.5)),
        ("Droplet", (22.0, 9.0, 22.0, 10.5)),
        ("GuidedRojoA3N", (0.0, 1.7, 0.0, 0.0)),
        ("NaiveMCTS", (0.0, 2.7, 0.0, 0.0)),
        ("overall", (9.9, 7.7, 7.5, 2.2)),
    ]


def build_eval_table_slide(builder: SlideBuilder, rows: list[tuple[str, str | tuple[float, float, float, float]]], fill_fn) -> None:
    col_x = [0.3, 2.85, 5.72, 7.95, 10.8]
    col_w = [2.42, 2.65, 1.75, 2.15, 2.12]
    header_y = 1.22
    builder.add_rule(0.3, 1.16, 12.75, 0.018)
    builder.add_textbox(0.32, header_y + 0.1, col_w[0] - 0.02, 0.38, [Paragraph("Opponent", size=12.2)])
    headers = [
        "Main Agent\nwithout new initial agents",
        "Base Agent",
        "Main Agent (LT)",
        "Main Agent +\nFine-tuning",
    ]
    for idx, header in enumerate(headers, start=1):
        builder.add_textbox(
            col_x[idx],
            header_y,
            col_w[idx],
            0.55,
            [Paragraph(header, size=11.6, align="ctr")],
            anchor="ctr",
        )
    builder.add_rule(0.3, 1.82, 12.75, 0.012)

    y = 1.92
    row_h = 0.27
    cat_h = 0.28
    for label, values in rows:
        if label == "category":
            builder.add_textbox(0.38, y, 4.4, cat_h, [Paragraph(str(values), size=12.3, italic=True)])
            y += cat_h
            continue

        if label == "overall":
            builder.add_rule(0.3, y - 0.03, 12.75, 0.012)

        builder.add_textbox(0.3, y, col_w[0], row_h, [Paragraph(label, size=12.1)])
        numbers = list(values)  # type: ignore[arg-type]
        best = max(numbers) if fill_fn is win_fill else min(numbers)
        for idx, value in enumerate(numbers, start=1):
            builder.add_shape(
                col_x[idx],
                y + 0.01,
                col_w[idx],
                row_h - 0.03,
                geom="rect",
                fill=fill_fn(value),
                line=None,
                name_prefix="Cell",
            )
            builder.add_textbox(
                col_x[idx],
                y,
                col_w[idx],
                row_h,
                [Paragraph(f"{value:.1f}", size=11.8, bold=abs(value - best) < 1e-9, align="ctr")],
                anchor="ctr",
            )
        y += row_h

    builder.add_rule(0.3, y - 0.02, 12.75, 0.018)


def build_eval_win_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    build_eval_table_slide(builder, eval_rows(), win_fill)


def build_eval_loss_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    build_eval_table_slide(builder, loss_rows(), loss_fill)


def build_interpretation_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(0.55, 1.45, 2.0, 0.28, [Paragraph("hypothesis:", size=16)])
    builder.add_textbox(
        1.05,
        1.85,
        11.2,
        1.75,
        [
            Paragraph("LT turns robustness against diverse opponents into part of the optimization target", size=18, bullet=True),
            Paragraph("LT helps to find a better local optimum that is then further improved by fine-tuning, while pure PPO often gets stuck in not as generalized local optima.", size=17.5, bullet=True),
        ],
    )


def build_player1_slide(builder: SlideBuilder, _: dict[str, Path], video_boxes: dict[int, tuple[int, int, int, int]], video_posters: dict[int, Path]) -> None:
    builder.add_block(
        0.18,
        1.32,
        13.0,
        1.25,
        "Observation",
        [Paragraph("MicroRTS gives player 1 action priority, so a potential concern is that self-play style training could accidentally favor player-1-specific behavior.", size=15.6)],
    )
    builder.add_block(
        0.18,
        3.0,
        13.0,
        1.28,
        "Result from the thesis",
        [Paragraph("In one self-match evaluation of the same agent over 200 games, player 1 achieved 38% wins, 28% draws and player 2 achieved 34% wins.", size=15.6)],
    )
    builder.add_textbox(
        0.45,
        4.62,
        12.2,
        0.95,
        [
            Paragraph("minimal effect on evaluations", size=17, bullet=True),
            Paragraph("alternating the learning side to prevent the agent from exploiting player-1-specific priority", size=17, bullet=True),
        ],
    )
    builder.add_video_emu(VIDEO_FILES[17], video_posters[17], video_boxes[17])


def build_diversity_video_slide(builder: SlideBuilder, _: dict[str, Path], video_boxes: dict[int, tuple[int, int, int, int]], video_posters: dict[int, Path]) -> None:
    builder.add_video_emu(VIDEO_FILES[18], video_posters[18], video_boxes[18])


def build_diversity_text_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(
        3.2,
        1.55,
        7.4,
        4.25,
        [
            Paragraph("Exploiters initialized from similar policies tend to discover similar weaknesses.", size=18, bullet=True),
            Paragraph("Some strategically different exploits are hard to reach because they require first unlearning the Base Agent's preferred behavior.", size=18, bullet=True),
            Paragraph("New initial agents with focuses on different unit types to improve exploiter diversity and Main Agent generalization.", size=18, bullet=True),
            Paragraph("Hypothesis: LT therefore depends strongly on the diversity already present at initialization", size=18, bullet=True),
        ],
    )


def build_scope_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(
        0.58,
        1.6,
        6.4,
        2.0,
        [
            Paragraph("evaluation is limited to one map: basesWorkers16x16A", size=18, bullet=True),
            Paragraph("the setting uses full observability and no fog of war", size=18, bullet=True),
            Paragraph("high computational cost", size=18, bullet=True),
            Paragraph("exploiter diversity", size=18, bullet=True),
        ],
    )
    builder.add_block(
        0.52,
        4.35,
        12.2,
        1.05,
        "Takeaway",
        [Paragraph("The method works, but the strength of the final result depends strongly on the diversity and cost of the training setup.", size=15.4)],
    )


def build_future_work_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(
        2.45,
        1.78,
        8.5,
        2.2,
        [
            Paragraph("train on multiple maps and with fog of war", size=19, bullet=True),
            Paragraph("improve the diversity and strength of initial agents", size=19, bullet=True),
            Paragraph("reduce computational cost of MicroRTS training", size=19, bullet=True),
            Paragraph("investigate stronger diversity mechanisms for exploiters", size=19, bullet=True),
        ],
    )


def build_conclusion_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_block(
        0.18,
        1.3,
        13.0,
        1.0,
        "Conclusion",
        [Paragraph("LT with PPO improves the robustness and generalization of the MicroRTS Base Agent. Additional PPO fine-tuning improves the final agent even further.", size=15.2)],
    )
    builder.add_block(
        0.18,
        2.72,
        13.0,
        1.0,
        "Key takeaway",
        [Paragraph("The effectiveness of LT depends strongly on exploiter diversity, which in turn depends on the diversity of the initial agents and BC experts.", size=14.7)],
    )
    builder.add_textbox(
        0.76,
        4.4,
        8.7,
        1.4,
        [
            Paragraph("LT improves most matchups and the overall performance", size=18, bullet=True),
            Paragraph("fine-tuning further improves the final agent", size=18, bullet=True),
            Paragraph("diversity of initial agents is the critical practical factor", size=18, bullet=True),
        ],
    )


def build_thank_you_slide(builder: SlideBuilder, _: dict[str, Path], __: dict[int, tuple[int, int, int, int]], ___: dict[int, Path]) -> None:
    builder.add_textbox(
        4.7,
        3.35,
        4.0,
        0.55,
        [Paragraph("Questions?", size=24, align="ctr")],
        inset_left=0.0,
        inset_top=0.0,
        inset_right=0.0,
        inset_bottom=0.0,
    )


SLIDES: list[tuple[str | None, callable]] = [
    (None, build_title_slide),
    ("Overview", build_overview_slide),
    ("Goals and Research Question", build_goals_slide),
    ("Setting", build_setting_slide),
    ("Training Pipeline", build_pipeline_slide),
    ("From SP to LT", build_from_sp_to_lt_slide),
    ("League Training Design", build_lt_design_slide),
    ("Bot Environments", build_bot_envs_slide),
    ("Observation Adjustment for SP", build_observation_slide),
    ("LT Matchmaking", build_matchmaking_slide),
    ("Other Implementation Changes", build_other_changes_slide),
    ("Experiment Setup", build_experiment_setup_slide),
    ("Main Quantitative Result", build_quantitative_slide),
    ("Evaluation Results", build_eval_win_slide),
    ("Evaluation Results", build_eval_loss_slide),
    ("Interpretation of the Results", build_interpretation_slide),
    ("Player 1 Priority", build_player1_slide),
    ("Main Limitation: Diversity", build_diversity_video_slide),
    ("Main Limitation: Diversity", build_diversity_text_slide),
    ("Scope and Remaining Limitations", build_scope_slide),
    ("Future Work", build_future_work_slide),
    ("Conclusion", build_conclusion_slide),
    ("Thank you", build_thank_you_slide),
]


def prepare_assets(tmpdir: Path) -> tuple[dict[str, Path], dict[int, tuple[int, int, int, int]], dict[int, Path]]:
    assets_dir = tmpdir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    assets = {
        "logo": THESIS_DIR / "hhu_logo.png",
        "num_parallel_bot_envs": THESIS_DIR / "images" / "num_parallel_bot_envs.png",
        "bot_win_rates": THESIS_DIR / "images" / "bot_win_rates_no_draw.png",
        "adjust_obs": render_pdf_to_png(THESIS_DIR / "images" / "adjust_obs_presentation.pdf", assets_dir / "adjust_obs"),
        "matchmaking": render_pdf_to_png(THESIS_DIR / "images" / "Matchmaking.pdf", assets_dir / "matchmaking"),
        "total_winrate": render_pdf_to_png(THESIS_DIR / "images" / "abstract_total_winrate_comparison.pdf", assets_dir / "total_winrate"),
    }

    build_temp_source()
    try:
        compile_temp_pdf()
        page_count, page_width, page_height = parse_pdfinfo(TEMP_SOURCE_PDF)
        if page_count != SLIDE_COUNT:
            raise RuntimeError(f"expected {SLIDE_COUNT} slides in temp PDF, got {page_count}")
        page_objs = page_object_map(TEMP_SOURCE_PDF)

        video_boxes: dict[int, tuple[int, int, int, int]] = {}
        video_posters: dict[int, Path] = {}

        slide_cx = inch(13.333333333)
        slide_cy = inch(7.5)

        for slide_num in VIDEO_FILES:
            rect = largest_annotation_rect(TEMP_SOURCE_PDF, page_objs[slide_num])
            x1, y1, x2, y2 = rect
            x = round(x1 / page_width * slide_cx)
            y = round((page_height - y2) / page_height * slide_cy)
            cx = round((x2 - x1) / page_width * slide_cx)
            cy = round((y2 - y1) / page_height * slide_cy)
            video_boxes[slide_num] = (x, y, cx, cy)

            crop = pdf_rect_to_pixels(rect, page_width, page_height, 1920, 1080)
            video_posters[slide_num] = render_poster_crop(
                TEMP_SOURCE_PDF,
                slide_num,
                crop,
                assets_dir / f"poster-slide{slide_num}",
            )

        return assets, video_boxes, video_posters
    finally:
        cleanup_temp_source()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ba_editable_ppt_") as tmp:
        tmpdir = Path(tmp)
        base_pptx = tmpdir / "base.pptx"
        package_dir = tmpdir / "package"

        assets, video_boxes, video_posters = prepare_assets(tmpdir)
        build_base_pptx(SLIDE_COUNT, base_pptx)

        with zipfile.ZipFile(base_pptx) as zf:
            zf.extractall(package_dir)

        media_dir = package_dir / "ppt" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        ensure_content_type_default(package_dir / "[Content_Types].xml", "png", "image/png")
        ensure_content_type_default(package_dir / "[Content_Types].xml", "mp4", "video/mp4")

        for slide_num, (title, slide_builder_fn) in enumerate(SLIDES, start=1):
            builder = SlideBuilder(slide_num, SLIDE_COUNT, media_dir, title)
            slide_builder_fn(builder, assets, video_boxes, video_posters)
            slide_path = package_dir / "ppt" / "slides" / f"slide{slide_num}.xml"
            rels_path = package_dir / "ppt" / "slides" / "_rels" / f"slide{slide_num}.xml.rels"
            builder.write(slide_path, rels_path)

        zip_dir(package_dir, OUTPUT_PPTX)
        validate_pptx(OUTPUT_PPTX)
        print(f"Wrote {OUTPUT_PPTX}")


if __name__ == "__main__":
    main()
