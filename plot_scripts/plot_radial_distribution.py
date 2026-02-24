#!/usr/bin/env python3

import argparse
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot radial distribution functions from multiple XVG files."
	)
	parser.add_argument(
		"inputs",
		nargs="+",
		help="Input XVG files, glob patterns, and/or directories.",
	)
	parser.add_argument(
		"-o",
		"--output",
		default="rdf_overlay.png",
		help="Output figure path (default: rdf_overlay.png).",
	)
	parser.add_argument(
		"-p",
		"--pattern",
		default="rdf.xvg",
		help="Pattern used when an input is a directory (default: rdf*.xvg).",
	)
	parser.add_argument(
		"-c",
		"--column",
		type=int,
		default=1,
		help="Y-column index in XVG data (0=x, 1=first y; default: 1).",
	)
	parser.add_argument(
		"--title",
		default="Radial Distribution Function",
		help="Plot title.",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=100,
		help="Output DPI (default: 100).",
	)
	return parser.parse_args()


def collect_files(inputs: List[str], pattern: str) -> List[Path]:
	collected: List[Path] = []

	for item in inputs:
		path = Path(item)

		if path.is_dir():
			collected.extend(sorted(path.glob(pattern)))
			continue

		matches = [Path(match) for match in glob.glob(item)]
		if matches:
			collected.extend(sorted(matches))
			continue

		if path.is_file():
			collected.append(path)

	return sorted({path.resolve() for path in collected if path.is_file()})


def parse_xvg(file_path: Path, column: int) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
	x_values: List[float] = []
	y_values: List[float] = []
	legend: Optional[str] = None

	with file_path.open("r", encoding="utf-8") as handle:
		for raw_line in handle:
			line = raw_line.strip()
			if not line:
				continue

			if line.startswith("@"):
				if "legend" in line and legend is None:
					parts = line.split("legend", maxsplit=1)
					if len(parts) > 1:
						legend = parts[1].strip().strip('"')
				continue

			if line.startswith("#"):
				continue

			parts = line.split()
			if len(parts) <= column:
				continue

			try:
				x_values.append(float(parts[0]))
				y_values.append(float(parts[column]))
			except ValueError:
				continue

	if not x_values:
		raise ValueError(f"No numeric data parsed from {file_path}")

	return np.array(x_values), np.array(y_values), legend


def main() -> None:
	args = parse_args()
	files = collect_files(args.inputs, args.pattern)

	if not files:
		raise FileNotFoundError("No input XVG files found.")

	if args.column < 1:
		raise ValueError("Column must be >= 1 for RDF y-values.")

	palette = sns.color_palette("tab10")
	palette.pop(3)

	sns.set_context("talk")
	fig, ax = plt.subplots(figsize=(9, 6))

	for idx, file_path in enumerate(files):
		x_values, y_values, legend = parse_xvg(file_path, args.column)
		label = file_path.stem
		if legend:
			label = f"{file_path.stem} ({legend})"

		ax.plot(
			x_values,
			y_values,
			linewidth=2.0,
			label=label,
			color=palette[idx % len(palette)],
		)

	ax.set_title(args.title)
	ax.set_xlabel("r (nm)")
	ax.set_ylabel("g(r)")
	ax.grid(True, alpha=0.3)
	ax.legend(frameon=True)

	output_path = Path(args.output).resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=args.dpi)
	plt.close(fig)

	print(f"Plotted {len(files)} files to {output_path}")


if __name__ == "__main__":
	main()
