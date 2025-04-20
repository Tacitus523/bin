#!/usr/bin/env python3

# This script converts a model saved with torch.save to a scripted model saved with torch.jit.save, which can be used in C++.
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
    model_paths = sorted(model_paths)
    if len(model_paths) == 0:
        print(f'No models found with prefix {args.model_prefix}')
        return
    elif len(model_paths) == 1:
        target_model_path = model_paths[0]
        print(f'Using the model: {target_model_path}')
    elif len(model_paths) > 1:
        print(f'Multiple models found with prefix {args.model_prefix}: {model_paths}')
        swa_model_paths = [f for f in model_paths if 'swa' in f]
        if len(swa_model_paths) == 0:
            target_model_path = model_paths[-1]
            print(f'No SWA models found, using the last model: {target_model_path}')
        elif len(swa_model_paths) == 1:
            target_model_path = swa_model_paths[0]
            print(f'Using the SWA model: {target_model_path}')
        else:
            print(f'Multiple SWA models found: {swa_model_paths}')
            target_model_path = swa_model_paths[-1]
            print(f'Using the last SWA model: {target_model_path}')

    model = torch.load(target_model_path, map_location='cpu')
    model_compiled = jit.compile(deepcopy(model.eval()))
    torch.jit.save(model_compiled, target_model_path.replace('.model', '.pt'))

if __name__ == '__main__':
    main()
    