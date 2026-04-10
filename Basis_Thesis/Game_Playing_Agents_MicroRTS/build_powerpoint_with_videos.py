#!/usr/bin/env python3

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple
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
TEMP_SOURCE_TEX = THESIS_DIR / "BA_presentation_ppt_source.tex"
TEMP_SOURCE_PDF = THESIS_DIR / "BA_presentation_ppt_source.pdf"
OUTPUT_PPTX = THESIS_DIR / "BA_presentation_embedded_videos.pptx"

FULL_RENDER_WIDTH = 1920
FULL_RENDER_HEIGHT = 1080

VIDEO_FILES = {
    4: THESIS_DIR / "images" / "MicroRTS_example.mp4",
    18: THESIS_DIR / "images" / "MictoRTS_BaseAgent_vs_BaseAgent_edited.mp4",
}


def run(cmd: Iterable[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


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
        path = THESIS_DIR / f"BA_presentation_ppt_source{suffix}"
        if path.exists():
            path.unlink()


def compile_temp_pdf() -> None:
    run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", TEMP_SOURCE_TEX.name],
        cwd=THESIS_DIR,
    )


def parse_pdfinfo(pdf_path: Path) -> Tuple[int, float, float]:
    info = run(["pdfinfo", str(pdf_path)])
    pages_match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    size_match = re.search(
        r"^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts", info, flags=re.MULTILINE
    )
    if not pages_match or not size_match:
        raise RuntimeError("unable to parse pdfinfo output")
    return int(pages_match.group(1)), float(size_match.group(1)), float(size_match.group(2))


def page_object_map(pdf_path: Path) -> Dict[int, int]:
    output = run(["mutool", "show", str(pdf_path), "pages"])
    mapping: Dict[int, int] = {}
    for line in output.splitlines():
        match = re.match(r"page\s+(\d+)\s+=\s+(\d+)\s+0\s+R", line.strip())
        if match:
            mapping[int(match.group(1))] = int(match.group(2))
    if not mapping:
        raise RuntimeError("unable to parse mutool page listing")
    return mapping


def largest_annotation_rect(pdf_path: Path, page_obj_num: int) -> Tuple[float, float, float, float]:
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


def render_full_slides(pdf_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "pdftoppm",
            "-jpeg",
            "-jpegopt",
            "quality=92",
            "-scale-to-x",
            str(FULL_RENDER_WIDTH),
            "-scale-to-y",
            str(FULL_RENDER_HEIGHT),
            "-forcenum",
            str(pdf_path),
            str(out_dir / "slide"),
        ]
    )


def pdf_rect_to_pixels(
    rect: Tuple[float, float, float, float], page_width: float, page_height: float
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    scale_x = FULL_RENDER_WIDTH / page_width
    scale_y = FULL_RENDER_HEIGHT / page_height
    x = round(x1 * scale_x)
    y = round((page_height - y2) * scale_y)
    w = round((x2 - x1) * scale_x)
    h = round((y2 - y1) * scale_y)
    return x, y, w, h


def render_poster_crop(
    pdf_path: Path, page_num: int, crop: Tuple[int, int, int, int], out_prefix: Path
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
            str(FULL_RENDER_WIDTH),
            "-scale-to-y",
            str(FULL_RENDER_HEIGHT),
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


def qname(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def relationship_element(rel_id: str, rel_type: str, target: str) -> ET.Element:
    return ET.Element(
        qname(REL_NS, "Relationship"),
        {"Id": rel_id, "Type": rel_type, "Target": target},
    )


def picture_shape(
    shape_id: int,
    name: str,
    rel_id: str,
    x: int,
    y: int,
    cx: int,
    cy: int,
) -> ET.Element:
    pic = ET.Element(qname(P_NS, "pic"))

    nv_pic_pr = ET.SubElement(pic, qname(P_NS, "nvPicPr"))
    ET.SubElement(nv_pic_pr, qname(P_NS, "cNvPr"), {"id": str(shape_id), "name": name})
    c_nv_pic_pr = ET.SubElement(nv_pic_pr, qname(P_NS, "cNvPicPr"))
    ET.SubElement(c_nv_pic_pr, qname(A_NS, "picLocks"), {"noChangeAspect": "1"})
    ET.SubElement(nv_pic_pr, qname(P_NS, "nvPr"))

    blip_fill = ET.SubElement(pic, qname(P_NS, "blipFill"))
    ET.SubElement(blip_fill, qname(A_NS, "blip"), {qname(R_NS, "embed"): rel_id})
    stretch = ET.SubElement(blip_fill, qname(A_NS, "stretch"))
    ET.SubElement(stretch, qname(A_NS, "fillRect"))

    sp_pr = ET.SubElement(pic, qname(P_NS, "spPr"))
    xfrm = ET.SubElement(sp_pr, qname(A_NS, "xfrm"))
    ET.SubElement(xfrm, qname(A_NS, "off"), {"x": str(x), "y": str(y)})
    ET.SubElement(xfrm, qname(A_NS, "ext"), {"cx": str(cx), "cy": str(cy)})
    prst_geom = ET.SubElement(sp_pr, qname(A_NS, "prstGeom"), {"prst": "rect"})
    ET.SubElement(prst_geom, qname(A_NS, "avLst"))

    return pic


def video_shape(
    shape_id: int,
    x: int,
    y: int,
    cx: int,
    cy: int,
    video_rel_id: str,
    media_rel_id: str,
    poster_rel_id: str,
) -> ET.Element:
    pic = ET.Element(qname(P_NS, "pic"))

    nv_pic_pr = ET.SubElement(pic, qname(P_NS, "nvPicPr"))
    c_nv_pr = ET.SubElement(nv_pic_pr, qname(P_NS, "cNvPr"), {"id": str(shape_id), "name": "video"})
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

    return pic


def build_slide_xml(
    slide_cx: int,
    slide_cy: int,
    video_box: Tuple[int, int, int, int] | None,
) -> bytes:
    root = ET.Element(qname(P_NS, "sld"))
    c_sld = ET.SubElement(root, qname(P_NS, "cSld"))
    sp_tree = ET.SubElement(c_sld, qname(P_NS, "spTree"))

    nv_grp_sp_pr = ET.SubElement(sp_tree, qname(P_NS, "nvGrpSpPr"))
    ET.SubElement(nv_grp_sp_pr, qname(P_NS, "cNvPr"), {"id": "1", "name": ""})
    ET.SubElement(nv_grp_sp_pr, qname(P_NS, "cNvGrpSpPr"))
    ET.SubElement(nv_grp_sp_pr, qname(P_NS, "nvPr"))

    grp_sp_pr = ET.SubElement(sp_tree, qname(P_NS, "grpSpPr"))
    xfrm = ET.SubElement(grp_sp_pr, qname(A_NS, "xfrm"))
    ET.SubElement(xfrm, qname(A_NS, "off"), {"x": "0", "y": "0"})
    ET.SubElement(xfrm, qname(A_NS, "ext"), {"cx": "0", "cy": "0"})
    ET.SubElement(xfrm, qname(A_NS, "chOff"), {"x": "0", "y": "0"})
    ET.SubElement(xfrm, qname(A_NS, "chExt"), {"cx": "0", "cy": "0"})

    sp_tree.append(picture_shape(2, "Background", "rId2", 0, 0, slide_cx, slide_cy))

    if video_box is not None:
        x, y, cx, cy = video_box
        sp_tree.append(video_shape(3, x, y, cx, cy, "rId3", "rId4", "rId5"))

    clr_map_ovr = ET.SubElement(root, qname(P_NS, "clrMapOvr"))
    ET.SubElement(clr_map_ovr, qname(A_NS, "masterClrMapping"))

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


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


def main() -> None:
    cleanup_temp_source()
    build_temp_source()
    try:
        compile_temp_pdf()
        slide_count, page_width, page_height = parse_pdfinfo(TEMP_SOURCE_PDF)

        page_objs = page_object_map(TEMP_SOURCE_PDF)
        video_rects_pts = {
            slide_num: largest_annotation_rect(TEMP_SOURCE_PDF, page_objs[slide_num])
            for slide_num in VIDEO_FILES
        }

        with tempfile.TemporaryDirectory(prefix="ba_ppt_build_") as tmp:
            tmpdir = Path(tmp)
            renders_dir = tmpdir / "renders"
            posters_dir = tmpdir / "posters"
            base_pptx = tmpdir / "base.pptx"
            package_dir = tmpdir / "package"

            render_full_slides(TEMP_SOURCE_PDF, renders_dir)
            rendered_bg_files = sorted(renders_dir.glob("slide-*.jpg"))
            if len(rendered_bg_files) != slide_count:
                raise RuntimeError(
                    f"expected {slide_count} rendered slide images, found {len(rendered_bg_files)}"
                )
            posters_dir.mkdir(parents=True, exist_ok=True)

            poster_paths: Dict[int, Path] = {}
            for slide_num, rect in video_rects_pts.items():
                crop = pdf_rect_to_pixels(rect, page_width, page_height)
                poster_paths[slide_num] = render_poster_crop(
                    TEMP_SOURCE_PDF,
                    slide_num,
                    crop,
                    posters_dir / f"poster-slide{slide_num}",
                )

            build_base_pptx(slide_count, base_pptx)

            with zipfile.ZipFile(base_pptx) as zf:
                zf.extractall(package_dir)

            presentation_tree = ET.parse(package_dir / "ppt" / "presentation.xml")
            presentation_root = presentation_tree.getroot()
            sld_sz = presentation_root.find(qname(P_NS, "sldSz"))
            if sld_sz is None:
                raise RuntimeError("unable to find slide size in presentation.xml")
            slide_cx = int(sld_sz.attrib["cx"])
            slide_cy = int(sld_sz.attrib["cy"])

            media_dir = package_dir / "ppt" / "media"
            media_dir.mkdir(parents=True, exist_ok=True)

            ensure_content_type_default(package_dir / "[Content_Types].xml", "jpg", "image/jpeg")
            ensure_content_type_default(package_dir / "[Content_Types].xml", "png", "image/png")
            ensure_content_type_default(package_dir / "[Content_Types].xml", "mp4", "video/mp4")

            for slide_num in range(1, slide_count + 1):
                bg_src = rendered_bg_files[slide_num - 1]
                bg_dst = media_dir / f"bg-slide{slide_num}.jpg"
                shutil.copy2(bg_src, bg_dst)

                rels_path = package_dir / "ppt" / "slides" / "_rels" / f"slide{slide_num}.xml.rels"
                rel_tree = ET.parse(rels_path)
                rel_root = rel_tree.getroot()

                rel_root.append(
                    relationship_element(
                        "rId2",
                        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                        f"../media/{bg_dst.name}",
                    )
                )

                video_box_emu = None

                if slide_num in VIDEO_FILES:
                    poster_src = poster_paths[slide_num]
                    poster_dst = media_dir / f"poster-slide{slide_num}.png"
                    video_dst = media_dir / f"video-slide{slide_num}.mp4"
                    shutil.copy2(poster_src, poster_dst)
                    shutil.copy2(VIDEO_FILES[slide_num], video_dst)

                    rel_root.append(
                        relationship_element(
                            "rId3",
                            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/video",
                            f"../media/{video_dst.name}",
                        )
                    )
                    rel_root.append(
                        relationship_element(
                            "rId4",
                            "http://schemas.microsoft.com/office/2007/relationships/media",
                            f"../media/{video_dst.name}",
                        )
                    )
                    rel_root.append(
                        relationship_element(
                            "rId5",
                            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                            f"../media/{poster_dst.name}",
                        )
                    )

                    x1, y1, x2, y2 = video_rects_pts[slide_num]
                    x = round(x1 / page_width * slide_cx)
                    y = round((page_height - y2) / page_height * slide_cy)
                    cx = round((x2 - x1) / page_width * slide_cx)
                    cy = round((y2 - y1) / page_height * slide_cy)
                    video_box_emu = (x, y, cx, cy)

                rel_tree.write(rels_path, encoding="utf-8", xml_declaration=True)

                slide_xml = build_slide_xml(slide_cx, slide_cy, video_box_emu)
                slide_path = package_dir / "ppt" / "slides" / f"slide{slide_num}.xml"
                slide_path.write_bytes(slide_xml)

            zip_dir(package_dir, OUTPUT_PPTX)
            validate_pptx(OUTPUT_PPTX)
            print(f"Wrote {OUTPUT_PPTX}")
    finally:
        cleanup_temp_source()


if __name__ == "__main__":
    main()
