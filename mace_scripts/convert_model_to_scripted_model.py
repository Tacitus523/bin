#!/usr/bin/env python3

# This script converts a model saved with torch.save to a scripted model saved with torch.jit.save
import argparse
import os
import sys
from copy import deepcopy
import warnings

import torch
from e3nn.util import jit

sys.path.append("/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools")
sys.path.append("/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange")

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_prefix', type=str, required=True, help='Prefix to the .model file')
    return parser.parse_args()

def main():
    args = parse_args()
    model_paths = [f for f in os.listdir(os.getcwd()) if f.endswith('.model') and f.startswith(args.model_prefix)]
    for model_path in model_paths:
        model = torch.load(model_path)
        model_compiled = jit.compile(deepcopy(model))
        torch.jit.save(model_compiled, model_path.replace('.model', '.pt'))

if __name__ == '__main__':
    main()
    