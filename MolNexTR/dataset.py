""" Preprocessing and data augmentation for training dataset """

import os
import cv2
import time
import random
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Custom imports for Indigo toolkit and data augmentations
from .indigo import Indigo
from .indigo.renderer import IndigoRenderer
from .data_aug import (SafeRotate, CropWhite, PadWhite, SaltAndPepperNoise, 
                       PadToSquare, ConditionalPadToSquare, AddIncompleteStructuralNoise, 
                       AddBondNoise, AddLineNoise, AddEdgeElementSymbolNoise, DrawBorder)
from .utils import FORMAT_INFO
from .tokenization import PAD_ID
from .chemical import get_num_atoms, normalize_nodes
from .abbrs import RGROUP_SYMBOLS, SUBSTITUTIONS, ELEMENTS, COLORS

# Limit OpenCV to use a single thread to prevent CPU overuse
cv2.setNumThreads(1)

# Constants controlling probabilities for various transformations and augmentations
INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2

def add_functional_group(indigo, mol, debug=False):
    """Randomly adds a functional group to a molecule."""
    if random.random() > INDIGO_FUNCTIONAL_GROUP_PROB:
        return mol

    substitutions = [sub for sub in SUBSTITUTIONS]
    random.shuffle(substitutions)
    for sub in substitutions:
        query = indigo.loadSmarts(sub.smarts)
        matcher = indigo.substructureMatcher(mol)
        matched_atoms_ids = set()
        for match in matcher.iterateMatches(query):
            if random.random() < sub.probability or debug:
                atoms = []
                atoms_ids = set()
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item)
                    atoms.append(atom)
                    atoms_ids.add(atom.index())
                if len(matched_atoms_ids.intersection(atoms_ids)) > 0:
                    continue
                abbrv = random.choice(sub.abbrvs)
                superatom = mol.addAtom(abbrv)
                for atom in atoms:
                    for nei in atom.iterateNeighbors():
                        if nei.index() not in atoms_ids:
                            if nei.symbol() == 'H':
                                atoms_ids.add(nei.index())
                            else:
                                superatom.addBond(nei, nei.bond().bondOrder())
                for id in atoms_ids:
                    mol.getAtom(id).remove()
                matched_atoms_ids = matched_atoms_ids.union(atoms_ids)
    return mol

def add_explicit_hydrogen(indigo, mol):
    """Adds explicit hydrogen atoms to the molecule."""
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append((atom, hs))
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_HYGROGEN_PROB:
        atom, hs = random.choice(atoms)
        for i in range(hs):
            h = mol.addAtom('H')
            h.addBond(atom, 1)
    return mol

def add_rgroup(indigo, mol, smiles):
    """Randomly adds an R-group to the molecule."""
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and '*' not in smiles:
        if random.random() < INDIGO_RGROUP_PROB:
            atom_idx = random.choice(range(len(atoms)))
            atom = atoms[atom_idx]
            atoms.pop(atom_idx)
            symbol = random.choice(RGROUP_SYMBOLS)
            r = mol.addAtom(symbol)
            r.addBond(atom, 1)
    return mol

def get_rand_symb():
    """Generates a random element symbol, possibly with modifiers like lowercase/uppercase letters."""
    symb = random.choice(ELEMENTS)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_lowercase)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_uppercase)
    if random.random() < 0.1:
        symb = f'({gen_rand_condensed()})'
    return symb

def get_rand_num():
    """Generates a random number for chemical formula representation, biased towards common cases."""
    if random.random() < 0.9:
        if random.random() < 0.8:
            return ''
        else:
            return str(random.randint(2, 9))
    else:
        return '1' + str(random.randint(2, 9))

def gen_rand_condensed():
    """Generates a random condensed formula, which could represent a fragment or group in a molecule."""
    tokens = []
    for i in range(5):
        if i >= 1 and random.random() < 0.8:
            break
        tokens.append(get_rand_symb())
        tokens.append(get_rand_num())
    return ''.join(tokens)

def add_rand_condensed(indigo, mol):
    """Adds a condensed formula as a pseudo-atom to a randomly chosen atom in the molecule."""
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_CONDENSED_PROB:
        atom = random.choice(atoms)
        symbol = gen_rand_condensed()
        r = mol.addAtom(symbol)
        r.addBond(atom, 1)
    return mol

def get_transforms(input_size, test_file, augment=True, rotate=True, debug=False):
    """Composes a sequence of image transformations, including augmentations, for training and testing."""
    trans_list = []
    if augment and rotate:
        trans_list.append(SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    trans_list.append(CropWhite(pad=50))
    if test_file == 'real/acs.csv' or test_file == 'real/UOB.csv':
        trans_list.append(PadToSquare(p=1))
    if augment:
        trans_list += [
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            PadWhite(pad_ratio=0.4, p=0.2),
            A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
            A.Blur(),
            A.GaussNoise(),
            SaltAndPepperNoise(num_dots=20, p=0.5),
            AddIncompleteStructuralNoise(), 
            AddBondNoise(), 
            AddLineNoise(), 
            AddEdgeElementSymbolNoise(), 
            DrawBorder()
        ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def generate_output_smiles(indigo, mol):
    """Generates SMILES output from an Indigo molecule, handling special cases for R-groups and other symbols."""
    smiles = mol.smiles()
    mol = indigo.loadMolecule(smiles)
    if '*' in smiles:
        part_a, part_b = smiles.split(' ', maxsplit=1)
        part_b = re.search(r'\$.*\$', part_b).group(0)[1:-1]
        symbols = [t for t in part_b.split(';') if len(t) > 0]
        output = ''
        cnt = 0
        for i, c in enumerate(part_a):
            if c != '*':
                output += c
            else:
                output += f'[{symbols[cnt]}]'
                cnt += 1
        return mol, output
    else:
        if ' ' in smiles:
            smiles = smiles.split(' ')[0]
        return mol, smiles

def add_comment(indigo):
    """Adds a random comment to the Indigo rendering options, for augmentation purposes."""
    if random.random() < INDIGO_COMMENT_PROB:
        indigo.setOption('render-comment', str(random.randint(1, 20)) + random.choice(string.ascii_letters))
        indigo.setOption('render-comment-font-size', random.randint(40, 60))
        indigo.setOption('render-comment-alignment', random.choice([0, 0.5, 1]))
        indigo.setOption('render-comment-position', random.choice(['top', 'bottom']))
        indigo.setOption('render-comment-offset', random.randint(2, 30))

def add_color(indigo, mol):
    """Adds color to specific atoms in the Indigo molecule rendering."""
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-coloring', True)
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption('render-base-color', random.choice(list(COLORS.values())))
    if random.random() < INDIGO_COLOR_PROB:
        if random.random() < 0.5:
            indigo.setOption('render-highlight-color-enabled', True)
            indigo.setOption('render-highlight-color', random.choice(list(COLORS.values())))
        if random.random() < 0.5:
            indigo.setOption('render-highlight-thickness-enabled', True)
        for atom in mol.iterateAtoms():
            if random.random() < 0.1:
                atom.highlight()
    return mol

def get_graph(mol, image, shuffle_nodes=False, pseudo_coords=False):
    """Generates a graph representation of the molecule, including atom coordinates and bond types."""
    mol.layout()
    coords, symbols = [], []
    index_map = {}
    atoms = [atom for atom in mol.iterateAtoms()]
    if shuffle_nodes:
        random.shuffle(atoms)
    for i, atom in enumerate(atoms):
        if pseudo_coords:
            x, y, z = atom.xyz()
        else:
            x, y = atom.coords()
        coords.append([x, y])
        symbols.append(atom.symbol())
        index_map[atom.index()] = i
    if pseudo_coords:
        coords = normalize_nodes(np.array(coords))
        h, w, _ = image.shape
        coords[:, 0] = coords[:, 0] * w
        coords[:, 1] = coords[:, 1] * h
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.iterateBonds():
        s = index_map[bond.source().index()]
        t = index_map[bond.destination().index()]
        edges[s, t] = bond.bondOrder()
        edges[t, s] = bond.bondOrder()
        if bond.bondStereo() in [5, 6]:
            edges[s, t] = bond.bondStereo()
            edges[t, s] = 11 - bond.bondStereo()
    graph = {
        'coords': coords,
        'symbols': symbols,
        'edges': edges,
        'num_atoms': len(symbols)
    }
    return graph

def generate_indigo_image(smiles, mol_augment=True, default_option=False, shuffle_nodes=False, pseudo_coords=False,
                          include_condensed=True, debug=False):
    """Generates a molecule image using Indigo, with optional augmentations and adjustments."""
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    indigo.setOption('render-output-format', 'png')
    indigo.setOption('render-background-color', '1,1,1')
    indigo.setOption('render-stereo-style', 'none')
    indigo.setOption('render-label-mode', 'hetero')
    indigo.setOption('render-font-family', 'Arial')
    if not default_option:
        thickness = random.uniform(0.5, 2)
        indigo.setOption('render-relative-thickness', thickness)
        indigo.setOption('render-bond-line-width', random.uniform(1, 4 - thickness))
        if random.random() < 0.5:
            indigo.setOption('render-font-family', random.choice(['Arial', 'Times', 'Courier', 'Helvetica']))
        indigo.setOption('render-label-mode', random.choice(['hetero', 'terminal-hetero']))
        indigo.setOption('render-implicit-hydrogens-visible', random.choice([True, False]))
        if random.random() < 0.1:
            indigo.setOption('render-stereo-style', 'old')
        if random.random() < 0.2:
            indigo.setOption('render-atom-ids-visible', True)

    try:
        mol = indigo.loadMolecule(smiles)
        if mol_augment:
            if random.random() < INDIGO_DEARMOTIZE_PROB:
                mol.dearomatize()
            else:
                mol.aromatize()
            smiles = mol.canonicalSmiles()
            add_comment(indigo)
            mol = add_explicit_hydrogen(indigo, mol)
            mol = add_rgroup(indigo, mol, smiles)
            if include_condensed:
                mol = add_rand_condensed(indigo, mol)
            mol = add_functional_group(indigo, mol, debug)
            mol = add_color(indigo, mol)
            mol, smiles = generate_output_smiles(indigo, mol)

        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
        graph = get_graph(mol, img, shuffle_nodes, pseudo_coords)
        success = True
    except Exception:
        if debug:
            raise Exception
        img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
        graph = {}
        success = False
    return img, smiles, graph, success

class TrainDataset(Dataset):
    """Dataset class for training, includes SMILES and associated image transformations."""
    def __init__(self, args, df, tokenizer, split='train', dynamic_indigo=False):
        super().__init__()
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        if 'file_path' in df.columns:
            self.file_paths = df['file_path'].values
            if not self.file_paths[0].startswith(args.data_path):
                self.file_paths = [os.path.join(args.data_path, path) for path in df['file_path']]
        self.smiles = df['SMILES'].values if 'SMILES' in df.columns else None
        self.formats = args.formats
        self.labelled = (split == 'train')
        if self.labelled:
            self.labels = {}
            for format_ in self.formats:
                if format_ in ['atomtok', 'inchi']:
                    field = FORMAT_INFO[format_]['name']
                    if field in df.columns:
                        self.labels[format_] = df[field].values
        self.transform = get_transforms(args.input_size, args.test_file,
                                        augment=(self.labelled and args.augment))
        self.dynamic_indigo = (dynamic_indigo and split == 'train')
        if self.labelled and not dynamic_indigo and args.coords_file is not None:
            if args.coords_file == 'aux_file':
                self.coords_df = df
                self.pseudo_coords = True
            else:
                self.coords_df = pd.read_csv(args.coords_file)
                self.pseudo_coords = False
        else:
            self.coords_df = None
            self.pseudo_coords = args.pseudo_coords

    def __len__(self):
        return len(self.df)

    def image_transform(self, image, coords=[], renormalize=False):
        """Applies image transformations and augments coordinates if provided."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image, keypoints=coords)
        image = augmented['image']
        if len(coords) > 0:
            coords = np.array(augmented['keypoints'])
            if renormalize:
                coords = normalize_nodes(coords, flip_y=False)
            else:
                _, height, width = image.shape
                coords[:, 0] = coords[:, 0] / width
                coords[:, 1] = coords[:, 1] / height
            coords = np.array(coords).clip(0, 1)
            return image, coords
        return image

    def __getitem__(self, idx):
        """Retrieves and processes an item based on index."""
        try:
            return self.getitem(idx)
        except Exception as e:
            with open(os.path.join(self.args.save_path, f'error_dataset_{int(time.time())}.log'), 'w') as f:
                f.write(str(e))
            raise e

    def getitem(self, idx):
        """Processes a single item from the dataset, applying transformations and tokenizations."""
        ref = {}
        if self.dynamic_indigo:
            begin = time.time()
            image, smiles, graph, success = generate_indigo_image(
                self.smiles[idx], mol_augment=self.args.mol_augment, default_option=self.args.default_option,
                shuffle_nodes=self.args.shuffle_nodes, pseudo_coords=self.args.pseudo_coords,
                include_condensed=self.args.include_condensed)
            end = time.time()
            if not success:
                return idx, None, {}
            image, coords = self.image_transform(image, graph['coords'], renormalize=self.args.pseudo_coords)
            graph['coords'] = coords
            ref['time'] = end - begin
            if 'atomtok' in self.formats:
                max_len = FORMAT_INFO['atomtok']['max_len']
                label = self.tokenizer['atomtok'].text_to_sequence(smiles, tokenized=False)
                ref['atomtok'] = torch.LongTensor(label[:max_len])
            if 'edges' in self.formats:
                ref['edges'] = torch.tensor(graph['edges'])
            return idx, image, ref
        else:
            file_path = self.file_paths[idx]
            image = cv2.imread(file_path)
            if image is None:
                image = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
                print(file_path, 'not found!')
            if self.coords_df is not None:
                h, w, _ = image.shape
                coords = np.array(eval(self.coords_df.loc[idx, 'node_coords']))
                if self.pseudo_coords:
                    coords = normalize_nodes(coords)
                coords[:, 0] = coords[:, 0] * w
                coords[:, 1] = coords[:, 1] * h
                image, coords = self.image_transform(image, coords, renormalize=self.pseudo_coords)
            else:
                image = self.image_transform(image)
                coords = None
            if self.labelled:
                smiles = self.smiles[idx]
                if 'atomtok' in self.formats:
                    max_len = FORMAT_INFO['atomtok']['max_len']
                    label = self.tokenizer['atomtok'].text_to_sequence(smiles, False)
                    ref['atomtok'] = torch.LongTensor(label[:max_len])
            return idx, image, ref

    def _process_atomtok_coords(self, idx, ref, smiles, coords=None, edges=None, mask_ratio=0):
        """Processes atom coordinates and edges for atom-level tokenization."""
        max_len = FORMAT_INFO['atomtok_coords']['max_len']
        tokenizer = self.tokenizer['atomtok_coords']
        if smiles is None or type(smiles) is not str:
            smiles = ""
        label, indices = tokenizer.smiles_to_sequence(smiles, coords, mask_ratio=mask_ratio)
        ref['atomtok_coords'] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref['atom_indices'] = torch.LongTensor(indices)
        if tokenizer.continuous_coords:
            if coords is not None:
                ref['coords'] = torch.tensor(coords)
            else:
                ref['coords'] = torch.ones(len(indices), 2) * -1.
        if edges is not None:
            ref['edges'] = torch.tensor(edges)[:len(indices), :len(indices)]

    def _process_chartok_coords(self, idx, ref, smiles, coords=None, edges=None, mask_ratio=0):
        """Processes coordinates and edges for character-level tokenization with coordinates."""
        max_len = FORMAT_INFO['chartok_coords']['max_len']
        tokenizer = self.tokenizer['chartok_coords']
        if smiles is None or type(smiles) is not str:
            smiles = ""
        label, indices = tokenizer.smiles_to_sequence(smiles, coords, mask_ratio=mask_ratio)
        ref['chartok_coords'] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref['atom_indices'] = torch.LongTensor(indices)
        if tokenizer.continuous_coords:
            if coords is not None:
                ref['coords'] = torch.tensor(coords)
            else:
                ref['coords'] = torch.ones(len(indices), 2) * -1.
        if edges is not None:
            ref['edges'] = torch.tensor(edges)[:len(indices), :len(indices)]

class AuxTrainDataset(Dataset):
    """Auxiliary dataset that combines training and auxiliary datasets for comprehensive training."""
    def __init__(self, args, train_df, aux_df, tokenizer):
        super().__init__()
        self.train_dataset = TrainDataset(args, train_df, tokenizer, dynamic_indigo=args.dynamic_indigo)
        self.aux_dataset = TrainDataset(args, aux_df, tokenizer, dynamic_indigo=False)

    def __len__(self):
        return len(self.train_dataset) + len(self.aux_dataset)

    def __getitem__(self, idx):
        if idx < len(self.train_dataset):
            return self.train_dataset[idx]
        else:
            return self.aux_dataset[idx - len(self.train_dataset)]

def pad_images(imgs):
    """Pads a batch of images to ensure they are all of the same size."""
    max_shape = [0, 0]
    for img in imgs:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[-1 - i])
    stack = []
    for img in imgs:
        pad = []
        for i in range(len(max_shape)):
            pad = pad + [0, max_shape[i] - img.shape[-1 - i]]
        stack.append(F.pad(img, pad, value=0))
    return torch.stack(stack)

def bms_collate(batch):
    """Custom collate function for batch data processing, padding sequences and stacking images."""
    ids = []
    imgs = []
    batch = [ex for ex in batch if ex[1] is not None]
    formats = list(batch[0][2].keys())
    seq_formats = [k for k in formats if
                   k in ['atomtok', 'inchi', 'nodes', 'atomtok_coords', 'chartok_coords', 'atom_indices']]
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        ref = ex[2]
        for key in seq_formats:
            refs[key][0].append(ref[key])
            refs[key][1].append(torch.LongTensor([len(ref[key])]))
    for key in seq_formats:
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    if 'coords' in formats:
        refs['coords'] = pad_sequence([ex[2]['coords'] for ex in batch], batch_first=True, padding_value=-1.)
    if 'edges' in formats:
        edges_list = [ex[2]['edges'] for ex in batch]
        max_len = max([len(edges) for edges in edges_list])
        refs['edges'] = torch.stack(
            [F.pad(edges, (0, max_len - len(edges), 0, max_len - len(edges)), value=-100) for edges in edges_list],
            dim=0)
    return ids, pad_images(imgs), refs
