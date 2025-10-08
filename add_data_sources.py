#!/usr/bin/env python3
"""
Add data source information to molecules in an extxyz file.
"""

import argparse
from ase import io
from ase.atoms import Atoms
from typing import List


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add data source information to molecules in an extxyz file."
    )
    parser.add_argument(
        "-g", "--geom_file",
        type=str,
        default="geoms.extxyz",
        help="Path to the geometry file in extxyz format (default: geoms.extxyz)"
    )
    parser.add_argument(
        "-d", "--data_source_file", 
        type=str,
        default="data_sources.txt",
        help="Path to the data source file with one source per line (default: data_sources.txt)"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=False, 
        help="Output file path (default: same as input)"
    )
    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = args.geom_file
    return args


def load_data_sources(data_source_file: str) -> List[str]:
    """Load data sources from file, one per line."""
    with open(data_source_file, 'r') as f:
        data_sources = [line.strip() for line in f if line.strip()]
    return data_sources


def add_data_sources_to_molecules(geom_file: str, data_sources: List[str], output_file: str) -> None:
    """Add data source information to each molecule's info dict and save."""
    # Load all molecules from extxyz file
    molecules = io.read(geom_file, index=":")
    
    if len(molecules) != len(data_sources):
        raise ValueError(
            f"Number of molecules ({len(molecules)}) doesn't match "
            f"number of data sources ({len(data_sources)})"
        )
    
    # Add data source to each molecule's info dict
    for i, (molecule, data_source) in enumerate(zip(molecules, data_sources)):
        if not "data_source" in molecule.info:
            molecule.info["data_source"] = data_source
        else:
            raise ValueError(
                f"Data source already exists for molecule {i+1}"
            )
    
    # Write molecules with data sources to output file
    io.write(output_file, molecules)
    print(f"Saved {len(molecules)} molecules with data sources to {output_file}")


def main() -> None:
    """Main function."""
    args = parse_args()
    
    print(f"Loading geometries from: {args.geom_file}")
    print(f"Loading data sources from: {args.data_source_file}")
    
    # Load data sources
    data_sources = load_data_sources(args.data_source_file)
    print(f"Loaded {len(data_sources)} data sources")
    
    # Add data sources to molecules and save
    add_data_sources_to_molecules(args.geom_file, data_sources, args.output_file)


if __name__ == "__main__":
    main()
