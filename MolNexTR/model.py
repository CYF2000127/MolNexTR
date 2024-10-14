""" Main model construction for MolNexTR """

import argparse
from typing import List
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Custom imports for dataset processing, model components, chemical conversion, and tokenization
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
        return {k.replace('module.', ''): v for k, v in state_dict.items()}  # Remove 'module.' prefix if present
    missing_keys, unexpected_keys = module.load_state_dict(remove_prefix(module_states), strict=False)
    return

BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]

class molnextr:
    """
    Main interface for MolNexTR to make predictions from input images.
    
    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to run the model on, defaults to CPU if None.
    """
    def __init__(self, model_path, device=None):
        # Load model states from file
        model_states = torch.load(model_path, map_location=torch.device('cpu'))
        # Get model arguments from saved states
        args = self._get_args(model_states['args'])
        # Set device to CPU or specified device
        if device is None:
            device = torch.device('cpu')
        self.device = device
        # Get tokenizer and build the encoder/decoder models
        self.tokenizer = get_tokenizer(args)
        self.encoder, self.decoder = self._get_model(args, self.tokenizer, self.device, model_states)
        # Load image transforms
        self.transform = get_transforms(args.input_size, augment=False)

    def _get_args(self, args_states=None):
        """
        Retrieves model arguments from state dictionary or initializes with default values.
        
        Args:
            args_states (dict): Saved model arguments, if available.

        Returns:
            args (argparse.Namespace): Model arguments.
        """
        parser = argparse.ArgumentParser()
        # Encoder settings
        parser.add_argument('--encoder', type=str, default='swin_base')
        parser.add_argument('--decoder', type=str, default='transformer')
        parser.add_argument('--trunc_encoder', action='store_true')  # Use hidden states before downsampling
        parser.add_argument('--no_pretrained', action='store_true')
        parser.add_argument('--use_checkpoint', action='store_true', default=True)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--embed_dim', type=int, default=256)
        parser.add_argument('--enc_pos_emb', action='store_true')  # Use positional embedding in encoder
        
        # Transformer decoder settings
        group = parser.add_argument_group("transformer_options")
        group.add_argument("--dec_num_layers", type=int, default=6, help="Number of layers in transformer decoder")
        group.add_argument("--dec_hidden_size", type=int, default=256, help="Decoder hidden size")
        group.add_argument("--dec_attn_heads", type=int, default=8, help="Number of attention heads in decoder")
        group.add_argument("--dec_num_queries", type=int, default=128)
        group.add_argument("--hidden_dropout", type=float, default=0.1, help="Hidden layer dropout rate")
        group.add_argument("--attn_dropout", type=float, default=0.1, help="Attention dropout rate")
        group.add_argument("--max_relative_positions", type=int, default=0)

        # Data processing settings
        parser.add_argument('--input_size', type=int, default=384)
        parser.add_argument('--vocab_file', type=str, default=None)
        parser.add_argument('--coord_bins', type=int, default=64)
        parser.add_argument('--sep_xy', action='store_true', default=True)

        args = parser.parse_args([])  # Create an empty argument set
        if args_states:
            for key, value in args_states.items():
                args.__dict__[key] = value  # Load saved arguments into current args
        return args

    def _get_model(self, args, tokenizer, device, states):
        """
        Initializes the encoder and decoder models from saved states.
        
        Args:
            args (argparse.Namespace): Model arguments.
            tokenizer (Tokenizer): Tokenizer instance.
            device (torch.device): Device to load the model onto.
            states (dict): Saved model states.

        Returns:
            tuple: Encoder and decoder models.
        """
        # Initialize the encoder and decoder models
        encoder = Encoder(args, pretrained=False)
        args.encoder_dim = encoder.n_features
        decoder = Decoder(args, tokenizer)

        # Load model weights from saved states
        loading(encoder, states['encoder'])
        loading(decoder, states['decoder'])

        # Move models to the specified device and set to evaluation mode
        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        return encoder, decoder

    def predict_images(self, input_images: List, return_atoms_bonds=False, return_confidence=False, batch_size=16):
        """
        Predicts SMILES and molecular structure from input images.
        
        Args:
            input_images (List): List of input images.
            return_atoms_bonds (bool): Whether to return atom and bond information.
            return_confidence (bool): Whether to return confidence scores.
            batch_size (int): Batch size for processing.

        Returns:
            List: List of prediction results, including SMILES, molecular structure, and optional atom/bond info.
        """
        device = self.device
        predictions = []
        self.decoder.compute_confidence = return_confidence

        # Process images in batches
        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]
            images = [self.transform(image=image, keypoints=[])['image'] for image in batch_images]
            images = torch.stack(images, dim=0).to(device)

            with torch.no_grad():
                features, hiddens = self.encoder(images)
                batch_predictions = self.decoder.decode(features, hiddens)
            predictions += batch_predictions

        # Extract coordinates, symbols, and edges from predictions
        node_coords = [pred['chartok_coords']['coords'] for pred in predictions]
        node_symbols = [pred['chartok_coords']['symbols'] for pred in predictions]
        edges = [pred['edges'] for pred in predictions]

        # Convert predicted graph structures to SMILES strings and MOL files
        smiles_list, molblock_list, r_success = convert_graph_to_smiles(
            node_coords, node_symbols, edges, images=input_images)

        # Compile results, including optional atom/bond details and confidence scores
        outputs = []
        for smiles, molfile, pred in zip(smiles_list, molblock_list, predictions):
            pred_dict = {"predicted_smiles": smiles, "predicted_molfile": molfile}
            if return_atoms_bonds:
                coords = pred['chartok_coords']['coords']
                symbols = pred['chartok_coords']['symbols']
                atom_list = []
                for i, (symbol, coord) in enumerate(zip(symbols, coords)):
                    atom_dict = {"atom_number": f"{i}", "atom_symbol": symbol, "coords": (round(coord[0], 3), round(coord[1], 3))}
                    if return_confidence:
                        atom_dict["confidence"] = pred['chartok_coords']['atom_scores'][i]
                    atom_list.append(atom_dict)
                pred_dict["atom_sets"] = atom_list

                bond_list = []
                num_atoms = len(symbols)
                for i in range(num_atoms - 1):
                    for j in range(i + 1, num_atoms):
                        bond_type_int = pred['edges'][i][j]
                        if bond_type_int != 0:
                            bond_type_str = BOND_TYPES[bond_type_int]
                            bond_dict = {"atom_number": f"{i}", "bond_type": bond_type_str, "endpoints": (i, j)}
                            if return_confidence:
                                bond_dict["confidence"] = pred["edge_scores"][i][j]
                            bond_list.append(bond_dict)
                pred_dict["bond_sets"] = bond_list
            outputs.append(pred_dict)
        return outputs

    def predict_image(self, image, return_atoms_bonds=True, return_confidence=False):
        """
        Predicts SMILES and molecular structure from a single input image.
        
        Args:
            image (ndarray): Input image.
            return_atoms_bonds (bool): Whether to return atom and bond information.
            return_confidence (bool): Whether to return confidence scores.

        Returns:
            dict: Prediction result for the image.
        """
        return self.predict_images([image], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]

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
        return self.predict_images(input_images, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)

    def predict_final_results(self, image_file: str, return_atoms_bonds=True, return_confidence=False):
        """
        Predicts SMILES and molecular structure from a single image file path.
        
        Args:
            image_file (str): Path to the input image file.
            return_atoms_bonds (bool): Whether to return atom and bond information.
            return_confidence (bool): Whether to return confidence scores.

        Returns:
            dict: Prediction result for the image file.
        """
        return self.predict_image_files([image_file], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]
