#!/usr/bin/env python3
"""Create a compact total win rate comparison graphic for the thesis abstract.

The script compares:
- Base Agent
- Main Agent after League Training (LT)
- Main Agent after LT and PPO finetuning

By default, the comparison is performed on the shared evaluation opponents only.
This excludes the synthetic "total" row and the additional "BaseAgent" self-play
row that is present only in the LT and PPO evaluation CSVs.

The output is a PDF that can be included directly in LaTeX.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EvaluationSource:
    label: str
    csv_name: str
    color: str


SOURCES = (
    EvaluationSource(
        label="Base Agent",
        csv_name="Base_Agent_evaluation_300.csv",
        color="8E6C88",
    ),
    EvaluationSource(
        label="Main Agent (LT)",
        csv_name="18_03_2026__finished_PPO_Basis_thesis__league_training_4__BC_focus_3_update_1674.csv",
        color="3E7CB1",
    ),
    EvaluationSource(
        label="Main Agent + Finetuning",
        csv_name="23_03_2026__PPO_finetuning_update_730_200.csv",
        color="4C9A6A",
    ),
)


@dataclass(frozen=True)
class Summary:
    label: str
    win_rate: float
    draw_rate: float
    loss_rate: float
    games: int
    wins: int
    draws: int
    losses: int
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a total win rate comparison plot for the abstract."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "abstract_total_winrate_comparison.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--mode",
        choices=("common", "reported-total"),
        default="common",
        help=(
            "Comparison mode. 'common' uses only shared opponents and excludes "
            "'BaseAgent'; 'reported-total' uses the precomputed 'total' row from each CSV."
        ),
    )
    parser.add_argument(
        "--keep-tex-source",
        action="store_true",
        help="Also save the generated standalone TeX file next to the PDF.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_common_opponents(source: EvaluationSource) -> Summary:
    rows = read_rows(SCRIPT_DIR / source.csv_name)
    relevant_rows = [
        row
        for row in rows
        if row["opponent"] not in {"total", "BaseAgent"}
    ]

    games = sum(int(row["games"]) for row in relevant_rows)
    wins = sum(int(row["wins"]) for row in relevant_rows)
    draws = sum(int(row["draws"]) for row in relevant_rows)
    losses = sum(int(row["losses"]) for row in relevant_rows)

    return Summary(
        label=source.label,
        win_rate=wins / games,
        draw_rate=draws / games,
        loss_rate=losses / games,
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        color=source.color,
    )


def summarize_reported_total(source: EvaluationSource) -> Summary:
    rows = read_rows(SCRIPT_DIR / source.csv_name)
    total_row = next(row for row in rows if row["opponent"] == "total")

    games = int(total_row["games"])
    wins = int(total_row["wins"])
    draws = int(total_row["draws"])
    losses = int(total_row["losses"])

    return Summary(
        label=source.label,
        win_rate=float(total_row["win_rate"]),
        draw_rate=float(total_row["draw_rate"]),
        loss_rate=float(total_row["loss_rate"]),
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        color=source.color,
    )


def build_summaries(mode: str) -> list[Summary]:
    if mode == "common":
        return [summarize_common_opponents(source) for source in SOURCES]
    return [summarize_reported_total(source) for source in SOURCES]


def hex_to_pgf_color_spec(hex_color: str) -> str:
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"{{rgb,255:red,{red};green,{green};blue,{blue}}}"


def build_tex(summaries: list[Summary], mode: str) -> str:
    values = "\n".join(
        (
            f"\\addplot+[ybar, bar shift=0pt, draw=none, fill={hex_to_pgf_color_spec(summary.color)}] "
            f"coordinates {{({summary.label},{summary.win_rate * 100:.2f})}};"
        )
        for summary in summaries
    )

    return rf"""\documentclass[tikz,border=6pt]{{standalone}}
\usepackage[T1]{{fontenc}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    ybar,
    width=13.2cm,
    height=7.7cm,
    ymin=0,
    ymax=100,
    bar width=22pt,
    enlarge x limits=0.28,
    symbolic x coords={{Base Agent,Main Agent (LT),Main Agent + Finetuning}},
    xtick={{Base Agent,Main Agent (LT),Main Agent + Finetuning}},
    axis x line*=bottom,
    axis y line*=left,
    ylabel={{Total win rate (\%)}},
    ymajorgrids=true,
    grid style={{draw=gray!20}},
    tick label style={{font=\small}},
    label style={{font=\small}},
    xticklabel style={{align=center, text width=3.6cm}},
    nodes near coords,
    every node near coord/.append style={{font=\bfseries\small, yshift=4pt, text=black}},
    nodes near coords={{\pgfmathprintnumber[fixed,precision=1]{{\pgfplotspointmeta}}\%}},
    title={{Total Win Rate Comparison}},
    title style={{font=\bfseries\normalsize, yshift=2pt}},
]
{values}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""


def compile_pdf(tex_source: str, output_pdf: Path, keep_tex_source: bool) -> None:
    output_pdf = output_pdf.resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="total_winrate_plot_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        tex_path = temp_dir / "plot.tex"
        tex_path.write_text(tex_source, encoding="utf-8")

        command = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ]
        completed = subprocess.run(
            command,
            cwd=temp_dir,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "pdflatex failed while generating the plot:\n" + completed.stdout
            )

        generated_pdf = temp_dir / "plot.pdf"
        shutil.copy2(generated_pdf, output_pdf)

        if keep_tex_source:
            tex_output = output_pdf.with_suffix(".tex")
            shutil.copy2(tex_path, tex_output)


def main() -> None:
    args = parse_args()
    summaries = build_summaries(args.mode)
    tex_source = build_tex(summaries, args.mode)
    compile_pdf(tex_source, args.output, args.keep_tex_source)

    print(f"Created {args.output}")
    for summary in summaries:
        print(
            f"{summary.label}: "
            f"win {summary.win_rate * 100:.2f}%, "
            f"draw {summary.draw_rate * 100:.2f}%, "
            f"loss {summary.loss_rate * 100:.2f}% "
            f"({summary.wins}/{summary.games} wins)"
        )


if __name__ == "__main__":
    main()
