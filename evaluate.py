import json
import argparse
import numpy as np
import multiprocessing
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs

rdkit.RDLogger.DisableLog('rdApp.*')
from SmilesPE.pretokenizer import atomwise_tokenizer


def get_args():
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', type=str, required=True, help="Path to the file containing gold SMILES strings.")
    parser.add_argument('--pred_file', type=str, required=True, help="Path to the file containing predicted SMILES strings.")
    parser.add_argument('--pred_field', type=str, default='SMILES', help="Column name for SMILES in the prediction file.")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of parallel workers for multiprocessing.")
    parser.add_argument('--tanimoto', action='store_true', help="Calculate Tanimoto similarity if set.")
    parser.add_argument('--keep_main', action='store_true', help="Keep only the main molecule fragment.")
    return parser.parse_args()


def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True):
    """
    Canonicalize a SMILES string with options to ignore chirality, cis/trans stereochemistry, and replace R groups.

    Args:
        smiles (str): SMILES string to canonicalize.
        ignore_chiral (bool): If True, ignore chirality during canonicalization.
        ignore_cistrans (bool): If True, ignore cis/trans stereochemistry.
        replace_rgroup (bool): If True, replace R groups with wildcard characters.

    Returns:
        str: Canonicalized SMILES string.
        bool: True if successful, False otherwise.
    """
    if not isinstance(smiles, str) or smiles == '':
        return '', False
    # Modify SMILES based on cis/trans and R group options
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token.startswith('[') and token.endswith(']'):
                symbol = token[1:-1]
                if symbol.startswith('R') and symbol[1:].isdigit():
                    tokens[j] = f'[{symbol[1:]}*]'
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = '*'
        smiles = ''.join(tokens)
    # Attempt to canonicalize SMILES
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=not ignore_chiral)
        success = True
    except:
        canon_smiles = smiles
        success = False
    return canon_smiles, success


def convert_smiles_to_canonsmiles(smiles_list, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True, num_workers=16):
    """
    Convert a list of SMILES strings to canonical SMILES using parallel processing.

    Args:
        smiles_list (list): List of SMILES strings.
        ignore_chiral (bool): If True, ignore chirality during canonicalization.
        ignore_cistrans (bool): If True, ignore cis/trans stereochemistry.
        replace_rgroup (bool): If True, replace R groups with wildcard characters.
        num_workers (int): Number of parallel workers.

    Returns:
        list: List of canonical SMILES strings.
        float: Average success rate of the conversion.
    """
    with multiprocessing.Pool(num_workers) as p:
        results = p.starmap(canonicalize_smiles,
                            [(smiles, ignore_chiral, ignore_cistrans, replace_rgroup) for smiles in smiles_list],
                            chunksize=128)
    canon_smiles, success = zip(*results)
    return list(canon_smiles), np.mean(success)


def _keep_main_molecule(smiles):
    """
    Retain only the main (largest) molecule fragment from a SMILES string.

    Args:
        smiles (str): SMILES string.

    Returns:
        str: SMILES string for the main fragment.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            main_mol = max(frags, key=lambda m: m.GetNumAtoms())
            return Chem.MolToSmiles(main_mol)
    except Exception:
        pass
    return smiles


def keep_main_molecule(smiles_list, num_workers=16):
    """
    Process a list of SMILES strings to retain only the main molecule fragments using parallel processing.

    Args:
        smiles_list (list): List of SMILES strings.
        num_workers (int): Number of parallel workers.

    Returns:
        list: List of SMILES strings with only the main fragment retained.
    """
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(_keep_main_molecule, smiles_list, chunksize=128)
    return results


def tanimoto_similarity(smiles1, smiles2):
    """
    Compute the Tanimoto similarity between two SMILES strings.

    Args:
        smiles1 (str): First SMILES string.
        smiles2 (str): Second SMILES string.

    Returns:
        float: Tanimoto similarity score.
    """
    try:
        mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
        fp1, fp2 = Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2)
        return DataStructs.FingerprintSimilarity(fp1, fp2)
    except:
        return 0


def compute_tanimoto_similarities(gold_smiles, pred_smiles, num_workers=16):
    """
    Calculate Tanimoto similarities for lists of gold and predicted SMILES strings using parallel processing.

    Args:
        gold_smiles (list): List of gold SMILES strings.
        pred_smiles (list): List of predicted SMILES strings.
        num_workers (int): Number of parallel workers.

    Returns:
        list: List of Tanimoto similarity scores.
    """
    with multiprocessing.Pool(num_workers) as p:
        similarities = p.starmap(tanimoto_similarity, zip(gold_smiles, pred_smiles))
    return similarities


class SmilesEvaluator:
    """
    Evaluator for comparing gold and predicted SMILES strings, with options for Tanimoto similarity and chirality evaluation.
    """
    def __init__(self, gold_smiles, num_workers=16, tanimoto=False):
        self.gold_smiles = gold_smiles
        self.num_workers = num_workers
        self.tanimoto = tanimoto
        self.gold_smiles_cistrans, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                     ignore_cistrans=True,
                                                                     num_workers=num_workers)
        self.gold_smiles_chiral, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                   ignore_chiral=True, ignore_cistrans=True,
                                                                   num_workers=num_workers)
        self.gold_smiles_cistrans = self._replace_empty(self.gold_smiles_cistrans)
        self.gold_smiles_chiral = self._replace_empty(self.gold_smiles_chiral)

    def _replace_empty(self, smiles_list):
        """Replace empty SMILES in the gold, otherwise it will be considered correct if both pred and gold is empty."""
        return [smiles if smiles is not None and type(smiles) is str and smiles != "" else "<empty>"
                for smiles in smiles_list]

    def evaluate(self, pred_smiles, include_details=False):
        """
        Evaluate the predicted SMILES strings against the gold standard.

        Args:
            pred_smiles (list): List of predicted SMILES strings.
            include_details (bool): If True, include detailed match information.

        Returns:
            dict: Evaluation metrics including Tanimoto similarity and canonical SMILES accuracy.
        """
        results = {}
        if self.tanimoto:
            results['tanimoto'] = np.mean(compute_tanimoto_similarities(self.gold_smiles, pred_smiles))
        # Ignore double bond cis/trans
        pred_smiles_cistrans, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                                ignore_cistrans=True,
                                                                num_workers=self.num_workers)
        results['canon_smiles'] = np.mean(np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        if include_details:
            results['canon_smiles_details'] = (np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        # Ignore chirality (Graph exact match)
        pred_smiles_chiral, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                              ignore_chiral=True, ignore_cistrans=True,
                                                              num_workers=self.num_workers)
        results['graph'] = np.mean(np.array(self.gold_smiles_chiral) == np.array(pred_smiles_chiral))
        # Evaluate on molecules with chiral centers
        chiral = np.array([[g, p] for g, p in zip(self.gold_smiles_cistrans, pred_smiles_cistrans) if '@' in g])
        results['chiral'] = np.mean(chiral[:, 0] == chiral[:, 1]) if len(chiral) > 0 else -1
        return results


if __name__ == "__main__":
    args = get_args()
    gold_df = pd.read_csv(args.gold_file)
    pred_df = pd.read_csv(args.pred_file)

    if len(pred_df) != len(gold_df):
        print(f"Pred ({len(pred_df)}) and Gold ({len(gold_df)}) have different lengths!")

    # Re-order pred_df to have the same order with gold_df
    image2goldidx = {image_id: idx for idx, image_id in enumerate(gold_df['image_id'])}
    image2predidx = {image_id: idx for idx, image_id in enumerate(pred_df['image_id'])}
    for image_id in gold_df['image_id']:
        # If image_id doesn't exist in pred_df, add an empty prediction.
        if image_id not in image2predidx:
            pred_df = pred_df.append({'image_id': image_id, args.pred_field: ""}, ignore_index=True)
    image2predidx = {image_id: idx for idx, image_id in enumerate(pred_df['image_id'])}
    pred_df = pred_df.reindex([image2predidx[image_id] for image_id in gold_df['image_id']])

    evaluator = SmilesEvaluator(gold_df['SMILES'], args.num_workers, args.tanimoto)
    scores = evaluator.evaluate(pred_df[args.pred_field])
    print(json.dumps(scores, indent=4))
