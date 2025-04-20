#!/usr/bin/env python3

# This script converts a model saved with torch.save to a scripted model saved with torch.jit.save, which can be used in C++.
import argparse
import os
import sys
from copy import deepcopy
import warnings

import torch

sys.path.append("/home/ka/ka_ipc/ka_he8978/amp_qmmm")
from wrap_amp import WrappedAMPModel # type: ignore

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to the uncompiled .pt file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = torch.load(args.model, map_location='cpu')
    model_compiled = torch.jit.optimize_for_inference(torch.jit.script(deepcopy(model.eval())))
    torch.jit.save(model_compiled, f"compiled_{args.model.replace('.model', '.pt')}")

if __name__ == '__main__':
    main()
    