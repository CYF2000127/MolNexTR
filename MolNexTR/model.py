"""model for MolNexTR"""

import argparse
from typing import List

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .dataset import get_transforms
from .components import Encoder, Decoder
from .chemical import convert_graph_to_smiles
from .tokenization import get_tokenizer

def loading(module, module_states):
    """
    Loads the model's state_dict into a module, handling potential prefix mismatches.
    
    Args:
        module (torch.nn.Module): The module (model) to load the state_dict into.
        module_states (dict): The state dictionary to load.
    """
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = module.load_state_dict(remove_prefix(module_states), strict=False)
    return

BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]


class molnextr:
    """
    Main Interface for MolNexTR to get predictions
    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to run the model on, defaults to CPU if None.
    """
    def __init__(self, model_path, device=None):
        model_states = torch.load(model_path, map_location=torch.device('cpu'))
        args = self._get_args(model_states['args'])
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.tokenizer = get_tokenizer(args)
        self.encoder, self.decoder = self._get_model(args, self.tokenizer, self.device, model_states)
        self.transform = get_transforms(args.input_size, args.input_size, augment=False)

    def _get_args(self, args_states=None):
        parser = argparse.ArgumentParser()
        # Model
        parser.add_argument('--encoder', type=str, default='swin_base')
        parser.add_argument('--decoder', type=str, default='transformer')
        parser.add_argument('--trunc_encoder', action='store_true')  # use the hidden states before downsample
        parser.add_argument('--no_pretrained', action='store_true')
        parser.add_argument('--use_checkpoint', action='store_true', default=True)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--embed_dim', type=int, default=256)
        parser.add_argument('--enc_pos_emb', action='store_true')
        group = parser.add_argument_group("transformer_options")
        group.add_argument("--dec_num_layers", help="No. of layers in transformer decoder", type=int, default=6)
        group.add_argument("--dec_hidden_size", help="Decoder hidden size", type=int, default=256)
        group.add_argument("--dec_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
        group.add_argument("--dec_num_queries", type=int, default=128)
        group.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
        group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
        group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
        parser.add_argument('--continuous_coords', action='store_true')
        parser.add_argument('--compute_confidence', action='store_true')
        # Data
        parser.add_argument('--input_size', type=int, default=384)
        parser.add_argument('--vocab_file', type=str, default=None)
        parser.add_argument('--coord_bins', type=int, default=64)
        parser.add_argument('--sep_xy', action='store_true', default=True)

        args = parser.parse_args([])
        if args_states:
            for key, value in args_states.items():
                args.__dict__[key] = value
        return args

    def _get_model(self, args, tokenizer, device, states):
        encoder = Encoder(args, pretrained=False)
        args.encoder_dim = encoder.n_features
        decoder = Decoder(args, tokenizer)

        loading(encoder, states['encoder'])
        loading(decoder, states['decoder'])

        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        return encoder, decoder

    def predict_images(self, input_images: List, return_atoms_bonds=False, return_confidence=False, batch_size=16):
        device = self.device
        predictions = []
        self.decoder.compute_confidence = return_confidence

        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]
            images = [self.transform(image=image, keypoints=[])['image'] for image in batch_images]
            images = torch.stack(images, dim=0).to(device)
            with torch.no_grad():
                features, hiddens = self.encoder(images)
                batch_predictions = self.decoder.decode(features, hiddens)
            predictions += batch_predictions

        node_coords = [pred['chartok_coords']['coords'] for pred in predictions]
        node_symbols = [pred['chartok_coords']['symbols'] for pred in predictions]
        edges = [pred['edges'] for pred in predictions]

        smiles_list, molblock_list, r_success = convert_graph_to_smiles(
            node_coords, node_symbols, edges, images=input_images)

        outputs = []
        for smiles, molfile, pred in zip(smiles_list, molblock_list, predictions):
            pred_dict = {"predicted_smiles": smiles, "predicted_molfile": molfile}
            if return_atoms_bonds:
                coords = pred['chartok_coords']['coords']
                symbols = pred['chartok_coords']['symbols']
                # get atoms info
                atom_list = []
                for i, (symbol, coord) in enumerate(zip(symbols, coords)):
                    atom_dict = {"atom_number":f"{i}", "atom_symbol": symbol, "coords":(round(coord[0],3),round(coord[1],3))}
                    if return_confidence:
                        atom_dict["confidence"] = pred['chartok_coords']['atom_scores'][i]
                    atom_list.append(atom_dict)
                pred_dict["atom_sets"] = atom_list
                # get bonds info
                bond_list = []
                num_atoms = len(symbols)
                for i in range(num_atoms-1):
                    for j in range(i+1, num_atoms):
                        bond_type_int = pred['edges'][i][j]
                        if bond_type_int != 0:
                            bond_type_str = BOND_TYPES[bond_type_int]
                            bond_dict = {"atom_number":f"{i}","bond_type": bond_type_str, "endpoints": (i, j)}
                            if return_confidence:
                                bond_dict["confidence"] = pred["edge_scores"][i][j]
                            bond_list.append(bond_dict)
                pred_dict["bond_sets"] = bond_list
            outputs.append(pred_dict)
        return outputs

    def predict_image(self, image, return_atoms_bonds=False, return_confidence=False):
        """
        Predicts SMILES and molecular structure from a single input image.
        
        Args:
            image (ndarray): Input image.
            return_atoms_bonds (bool): Whether to return atom and bond information.
            return_confidence (bool): Whether to return confidence scores.

        Returns:
            dict: Prediction result for the image.
        """
        return self.predict_images([
            image], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]

    def predict_image_files(self, image_files: List, return_atoms_bonds=False, return_confidence=False):
        """
        Predicts SMILES and molecular structure from a list of image file paths.
        
        Args:
            image_files (List): List of image file paths.
            return_atoms_bonds (bool): Whether to return atom and bond information.
            return_confidence (bool): Whether to return confidence scores.

        Returns:
            List: List of prediction results for each image file.
        """
        input_images = []
        for path in image_files:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_images.append(image)
        return self.predict_images(
            input_images, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)

    def predict_final_results(self, image_file: str, return_atoms_bonds=False, return_confidence=False):
        """
        Predicts SMILES and molecular structure from a single image file path.
        
        Args:
            image_file (str): Path to the input image file.
            return_atoms_bonds (bool): Whether to return atom and bond information.
            return_confidence (bool): Whether to return confidence scores.

        Returns:
            dict: Prediction result for the image file.
        """
        return self.predict_image_files(
            [image_file], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]
