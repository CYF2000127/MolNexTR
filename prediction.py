import argparse
import json
import torch
from MolNexTR import molnextr

import warnings 
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--image_path', type=str, default=None, required=True)
    args = parser.parse_args()

    device = torch.device('cuda')
    model = molnextr(args.model_path, device)
    output = model.predict_final_results(
        args.image_path, return_atoms_bonds=args.return_atoms_bonds, return_confidence=args.return_confidence)
    for key, value in output.items():
        print(f"{key}:")
        print(value + '\n' if isinstance(value, str) else json.dumps(value) + '\n')
