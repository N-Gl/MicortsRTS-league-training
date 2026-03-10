#!/usr/bin/env python3
"""Generate LaTeX-ready PGFPlots snippets for selected PFSP formulas."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path


def focused_strong_boosted_with_draws(x: float) -> float:
    return (
        32 * x**2.5 * (1 - x) ** 2.5 * ((x**4) / (x**4 + 0.3**4))
        + 0.2 * 1 / (1 + math.exp(-(0.5 - x) / 0.1))
    )


def focused_strong_with_draws(x: float) -> float:
    return (
        37.5 * x**2 * (1 - x) ** 2.5 * ((x**4) / (x**4 + 0.5**4))
        + 0.2 * 1 / (1 + math.exp(-(0.5 - x) / 0.1))
    )


FORMULAS = {
    "focused_strong_boosted_with_draws": (
        focused_strong_boosted_with_draws,
        ("teal!70!black", "#0B6E6E"),
    ),
    "focused_strong_with_draws": (focused_strong_with_draws, ("orange!85!black", "#C76B00")),
}

X_AXIS_LABEL = "win rate against opponent"
Y_AXIS_LABEL = "weighting for opponent selection"


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def coordinates_to_tex(x: list[float], y: list[float]) -> str:
    return "\n".join(f"({x_i:.6f}, {y_i:.6f})" for x_i, y_i in zip(x, y))


def linspace(start: float, stop: float, num_points: int) -> list[float]:
    if num_points == 1:
        return [start]
    step = (stop - start) / (num_points - 1)
    return [start + i * step for i in range(num_points)]


def axis_header(title: str, with_legend: bool) -> list[str]:
    options = [
        r"width=12cm,",
        r"height=7cm,",
        rf"xlabel={{{X_AXIS_LABEL}}},",
        rf"ylabel={{{Y_AXIS_LABEL}}},",
        rf"title={{{title}}},",
        r"xmin=0, xmax=1,",
        r"grid=both,",
        r"minor tick num=1,",
        r"line width=0.9pt,",
    ]
    if with_legend:
        options += [
            r"legend pos=north east,",
            r"legend cell align=left,",
        ]
    return [r"\begin{tikzpicture}", r"\begin{axis}[", *[f"  {opt}" for opt in options], r"]"]


def add_plot_block(name: str, color: str, x: list[float], y: list[float], with_legend: bool) -> list[str]:
    plot_lines = [
        rf"\addplot[very thick, {color}] coordinates {{",
        coordinates_to_tex(x, y),
        r"};",
    ]
    if with_legend:
        plot_lines.append(rf"\addlegendentry{{{tex_escape(name)}}}")
    return plot_lines


def axis_footer() -> list[str]:
    return [r"\end{axis}", r"\end{tikzpicture}"]


def build_single_plot(name: str, color: str, x: list[float], y: list[float]) -> str:
    lines: list[str] = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Optional in preamble: \pgfplotsset{compat=1.18}",
    ]
    lines += axis_header(tex_escape(name), with_legend=False)
    lines += add_plot_block(name, color, x, y, with_legend=False)
    lines += axis_footer()
    return "\n".join(lines) + "\n"


def build_combined_plot(curves: dict[str, tuple[str, list[float]]], x: list[float]) -> str:
    lines: list[str] = [
        r"% Requires: \usepackage{pgfplots}",
        r"% Optional in preamble: \pgfplotsset{compat=1.18}",
    ]
    lines += axis_header("PFSP Weighting Curves", with_legend=True)
    for name, (color, y) in curves.items():
        lines += add_plot_block(name, color, x, y, with_legend=True)
    lines += axis_footer()
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_standalone_document(input_tex_name: str) -> str:
    return "\n".join(
        [
            r"\documentclass[tikz,border=3pt]{standalone}",
            r"\usepackage{pgfplots}",
            r"\pgfplotsset{compat=1.18}",
            r"\begin{document}",
            rf"\input{{{input_tex_name}}}",
            r"\end{document}",
            "",
        ]
    )


def compile_tex_to_pdf(tex_path: Path) -> Path:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex was not found. Install TeX Live or run with --no-pdf.")

    wrapper_path = tex_path.with_name(f".__{tex_path.stem}_standalone.tex")
    write_text(wrapper_path, build_standalone_document(tex_path.name))

    try:
        cmd = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-jobname={tex_path.stem}",
            wrapper_path.name,
        ]
        result = subprocess.run(
            cmd,
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stdout_tail = "\n".join(result.stdout.splitlines()[-30:])
            stderr_tail = "\n".join(result.stderr.splitlines()[-30:])
            raise RuntimeError(
                f"pdflatex failed for {tex_path.name}.\n"
                f"stdout (tail):\n{stdout_tail}\n"
                f"stderr (tail):\n{stderr_tail}"
            )
    finally:
        wrapper_path.unlink(missing_ok=True)

    for suffix in (".aux", ".log"):
        tex_path.with_suffix(suffix).unlink(missing_ok=True)

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"Expected PDF was not created: {pdf_path}")
    return pdf_path


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def nice_ticks(min_val: float, max_val: float, num_ticks: int = 6) -> list[float]:
    if num_ticks < 2:
        return [min_val, max_val]
    if math.isclose(min_val, max_val):
        return [min_val + i for i in range(num_ticks)]
    span = max_val - min_val
    raw_step = span / (num_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        nice_norm = 1
    elif normalized <= 2:
        nice_norm = 2
    elif normalized <= 5:
        nice_norm = 5
    else:
        nice_norm = 10
    step = nice_norm * magnitude
    start = math.floor(min_val / step) * step
    ticks: list[float] = []
    value = start
    max_allowed = max_val + step
    while value <= max_allowed:
        if value >= min_val - step * 0.5:
            ticks.append(value)
        value += step
    return ticks


def format_tick(v: float) -> str:
    if abs(v) >= 1000 or (0 < abs(v) < 0.01):
        return f"{v:.2e}"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def build_svg_plot(
    title: str,
    curves: list[tuple[str, str, list[float], list[float]]],
    with_legend: bool,
) -> str:
    width = 1200
    height = 760
    margin_left = 100
    margin_right = 30
    margin_top = 50
    margin_bottom = 90

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x_min, x_max = 0.0, 1.0

    all_y = [y for _, _, _, ys in curves for y in ys]
    y_min_raw = min(all_y)
    y_max_raw = max(all_y)
    if math.isclose(y_min_raw, y_max_raw):
        y_pad = max(0.1, abs(y_min_raw) * 0.1 + 0.05)
    else:
        y_pad = (y_max_raw - y_min_raw) * 0.06
    y_min = y_min_raw - y_pad
    y_max = y_max_raw + y_pad

    def x_to_px(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def y_to_px(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    x_ticks = [i / 10 for i in range(11)]
    y_ticks = nice_ticks(y_min, y_max, num_ticks=7)

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="24" font-family="Arial">{svg_escape(title)}</text>',
    ]

    # Grid lines and ticks.
    for xv in x_ticks:
        px = x_to_px(xv)
        lines.append(
            f'<line x1="{px:.2f}" y1="{margin_top}" x2="{px:.2f}" y2="{height - margin_bottom}" stroke="#e2e2e2" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{px:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="14" font-family="Arial">{format_tick(xv)}</text>'
        )

    for yv in y_ticks:
        py = y_to_px(yv)
        lines.append(
            f'<line x1="{margin_left}" y1="{py:.2f}" x2="{width - margin_right}" y2="{py:.2f}" stroke="#e2e2e2" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 12}" y="{py + 5:.2f}" text-anchor="end" font-size="14" font-family="Arial">{format_tick(yv)}</text>'
        )

    # Axis frame.
    lines.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222222" stroke-width="2"/>'
    )
    lines.append(
        f'<text x="{margin_left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="18" font-family="Arial">{svg_escape(X_AXIS_LABEL)}</text>'
    )
    lines.append(
        f'<text x="30" y="{margin_top + plot_h / 2:.2f}" transform="rotate(-90 30,{margin_top + plot_h / 2:.2f})" text-anchor="middle" font-size="18" font-family="Arial">{svg_escape(Y_AXIS_LABEL)}</text>'
    )

    for name, hex_color, x_values, y_values in curves:
        points = " ".join(f"{x_to_px(xv):.2f},{y_to_px(yv):.2f}" for xv, yv in zip(x_values, y_values))
        lines.append(
            f'<polyline fill="none" stroke="{hex_color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'
        )

    if with_legend:
        legend_x = width - margin_right - 340
        legend_y = margin_top + 16
        legend_h = 28 * len(curves) + 18
        lines.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="320" height="{legend_h}" fill="white" fill-opacity="0.9" stroke="#888888" stroke-width="1"/>'
        )
        for idx, (name, hex_color, _, _) in enumerate(curves):
            y_line = legend_y + 22 + idx * 28
            lines.append(
                f'<line x1="{legend_x + 12}" y1="{y_line}" x2="{legend_x + 52}" y2="{y_line}" stroke="{hex_color}" stroke-width="4"/>'
            )
            lines.append(
                f'<text x="{legend_x + 62}" y="{y_line + 5}" font-size="14" font-family="Arial">{svg_escape(name)}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX PGFPlots snippets for PFSP weighting functions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Basis_Thesis/plots/pfsp"),
        help="Directory where .tex plot files are written.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=401,
        help="Number of sampled x values in [0, 1].",
    )
    parser.add_argument(
        "--pdf",
        dest="generate_pdf",
        action="store_true",
        default=True,
        help="Compile standalone PDFs (one per individual graph) via pdflatex.",
    )
    parser.add_argument(
        "--no-pdf",
        dest="generate_pdf",
        action="store_false",
        help="Skip PDF generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_points < 2:
        raise ValueError("--num-points must be at least 2")

    x = linspace(0.0, 1.0, args.num_points)
    curve_data: dict[str, tuple[str, str, list[float]]] = {}
    written_files = 0
    single_plot_tex_paths: list[Path] = []

    for name, (fn, (tikz_color, svg_color)) in FORMULAS.items():
        y = [fn(x_i) for x_i in x]
        curve_data[name] = (tikz_color, svg_color, y)
        single_tex_path = args.output_dir / f"{name}.tex"
        single_plot_tex = build_single_plot(name=name, color=tikz_color, x=x, y=y)
        write_text(single_tex_path, single_plot_tex)
        single_plot_tex_paths.append(single_tex_path)
        written_files += 1
        single_plot_svg = build_svg_plot(
            title=name,
            curves=[(name, svg_color, x, y)],
            with_legend=False,
        )
        write_text(args.output_dir / f"{name}.svg", single_plot_svg)
        written_files += 1

    combined_plot_tex = build_combined_plot(
        curves={name: (tikz_color, y) for name, (tikz_color, _, y) in curve_data.items()},
        x=x,
    )
    write_text(args.output_dir / "pfsp_draw_weightings_combined.tex", combined_plot_tex)
    written_files += 1
    combined_plot_svg = build_svg_plot(
        title="PFSP Weighting Curves",
        curves=[(name, svg_color, x, y) for name, (_, svg_color, y) in curve_data.items()],
        with_legend=True,
    )
    write_text(args.output_dir / "pfsp_draw_weightings_combined.svg", combined_plot_svg)
    written_files += 1

    if args.generate_pdf:
        for single_tex_path in single_plot_tex_paths:
            compile_tex_to_pdf(single_tex_path)
            written_files += 1

    print(f"Wrote {written_files} files to {args.output_dir}")


if __name__ == "__main__":
    main()
