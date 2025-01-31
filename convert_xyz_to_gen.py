#!/usr/bin/env python
import argparse
import ase.io
import os

FILE_NAME = "geom.xyz"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converts .xyz files to .gen files")
    parser.add_argument("-i", "--input", required=False, default=FILE_NAME, help="Input file name")
    parser.add_argument("-d", "--directory_prefix", required=False, default=None, help="Prefix of the directories with input files")
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args

def convert_xyz_to_gen(input_file: str) -> None:
    name, extension = os.path.splitext(input_file)
    atoms = ase.io.read(input_file)
    ase.io.write(f"{name}.gen", atoms, format="gen")

def main():
    args = parse_args()
    if args.directory_prefix is not None:
        elegible_directories = [f.name for f in os.scandir(".") if f.is_dir() and f.name.startswith(args.directory_prefix)] # not sorted
    else:
        elegible_directories = [os.getcwd()]
    for directory_idx, directory in enumerate(elegible_directories):
        if directory_idx % 1000 == 0:
            print(f"Processing directory {args.directory_prefix}{directory_idx}")
        os.chdir(directory)
        convert_xyz_to_gen(args.input)
        os.chdir("..")

if __name__ == "__main__":
    main()
