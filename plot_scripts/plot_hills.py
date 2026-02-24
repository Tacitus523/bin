#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


KNOWN_NON_CV_COLUMNS = {"height", "biasf", "clock"}


@dataclass
class HillsData:
	file_path: Path
	fields: List[str]
	cv_fields: List[str]
	values: np.ndarray

	@property
	def time(self) -> np.ndarray:
		return self.values[:, 0]

	def cv_values(self, cv_name: str) -> np.ndarray:
		column_index = self.fields.index(cv_name)
		return self.values[:, column_index]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot overview of CVs against time for all HILLS files."
	)
	parser.add_argument(
		"-i",
		"--input-dir",
		type=Path,
		default=Path.cwd(),
		help="Directory to search for HILLS files (default: current directory).",
	)
	parser.add_argument(
		"-p",
		"--pattern",
		type=str,
		default="HILLS*",
		help="Glob pattern to match HILLS files (default: HILLS*).",
	)
	parser.add_argument(
		"-r",
		"--recursive",
		action="store_true",
		help="Recursively search for matching files.",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=Path("hills_overview.png"),
		help="Output figure path (default: hills_overview.png).",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=250,
		help="DPI for saved figure (default: 250).",
	)
	return parser.parse_args()


def extract_fields(file_path: Path) -> List[str]:
	with file_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			if line.startswith("#! FIELDS"):
				return line.strip().split()[2:]
	raise ValueError(f"No '#! FIELDS' header found in {file_path}")


def detect_cv_fields(fields: Sequence[str]) -> List[str]:
	if not fields or fields[0] != "time":
		raise ValueError("Expected 'time' as first field in HILLS header")

	cv_fields: List[str] = []
	for field_name in fields[1:]:
		if field_name.startswith("sigma_"):
			break
		if field_name in KNOWN_NON_CV_COLUMNS:
			continue
		cv_fields.append(field_name)

	if not cv_fields:
		cv_fields = [
			field_name
			for field_name in fields[1:]
			if not field_name.startswith("sigma_")
			and field_name not in KNOWN_NON_CV_COLUMNS
		]
	return cv_fields


def load_hills_data(file_path: Path) -> HillsData:
	fields = extract_fields(file_path)
	cv_fields = detect_cv_fields(fields)

	values = np.genfromtxt(str(file_path), comments="#")
	if values.ndim == 1:
		values = values.reshape(1, -1)

	if values.shape[1] != len(fields):
		raise ValueError(
			f"Column mismatch in {file_path}: header has {len(fields)}, data has {values.shape[1]}"
		)

	return HillsData(file_path=file_path, fields=fields, cv_fields=cv_fields, values=values)


def find_hills_files(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
	iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)

	def natural_key(path: Path) -> List[object]:
		parts = re.split(r"(\d+)", path.name)
		return [int(part) if part.isdigit() else part.lower() for part in parts]

	files = sorted((path for path in iterator if path.is_file()), key=natural_key)
	return files


def plot_overview(hills_data: Sequence[HillsData], output_path: Path, dpi: int) -> None:
	rows = []
	for dataset in hills_data:
		for cv_name in dataset.cv_fields:
			cv_series = dataset.cv_values(cv_name)
			for time_value, cv_value in zip(dataset.time, cv_series):
				rows.append(
					{
						"time": time_value,
						"value": cv_value,
						"cv": cv_name,
						"hills_file": dataset.file_path.name,
					}
				)

	if not rows:
		raise ValueError("No CV columns detected in HILLS files")

	df = pd.DataFrame(rows)
	hills_order = []
	for dataset in hills_data:
		name = dataset.file_path.name
		if name not in hills_order:
			hills_order.append(name)
	sns.set_context("talk")

	g = sns.relplot(
		data=df,
		x="time",
		y="value",
		kind="line",
		col="hills_file",
		hue="cv",
		col_order=hills_order,
		col_wrap=4,
		height=3.2,
		aspect=1.4,
		facet_kws={"sharex": True, "sharey": False, "margin_titles": True},
		linewidth=0.9,
	)
	g.set_axis_labels("time", "CV value")
	g.set_titles(col_template="{col_name}")
	g.fig.suptitle("HILLS CV overview", y=1.02)
	g.figure.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	g.figure.savefig(output_path, dpi=dpi)
	plt.close(g.figure)


def main() -> None:
	args = parse_args()
	input_dir = args.input_dir.resolve()
	output_path = args.output.resolve()

	hills_files = find_hills_files(input_dir=input_dir, pattern=args.pattern, recursive=args.recursive)
	if not hills_files:
		raise FileNotFoundError(
			f"No files matched pattern '{args.pattern}' in {input_dir} (recursive={args.recursive})"
		)

	datasets = [load_hills_data(file_path) for file_path in hills_files]
	plot_overview(hills_data=datasets, output_path=output_path, dpi=args.dpi)

	print(f"Plotted {len(datasets)} files to {output_path}")


if __name__ == "__main__":
	main()
