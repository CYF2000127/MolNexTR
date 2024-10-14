""" Tokenization for MolNexTR's output sequences """

import os
import json
import random
import numpy as np
from SmilesPE.pretokenizer import atomwise_tokenizer

# Define special token IDs and corresponding strings
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4
PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
MASK = '<mask>'

class Tokenizer(object):
    """
    Tokenizer for basic tokenization, supporting saving and loading vocabularies.
    Converts texts to sequences of IDs and vice versa.
    """
    def __init__(self, path=None):
        self.stoi = {}  # String-to-Index dictionary
        self.itos = {}  # Index-to-String dictionary
        if path:
            self.load(path)

    def __len__(self):
        return len(self.stoi)

    @property
    def output_constraint(self):
        return False

    def save(self, path):
        """ Save vocabulary to a JSON file. """
        with open(path, 'w') as f:
            json.dump(self.stoi, f)

    def load(self, path):
        """ Load vocabulary from a JSON file. """
        with open(path) as f:
            self.stoi = json.load(f)
        self.itos = {item[1]: item[0] for item in self.stoi.items()}  # Invert dictionary for ID-to-String lookup

    def fit_on_texts(self, texts):
        """ Build vocabulary based on provided texts. """
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))  # Assume space-separated tokens
        vocab = [PAD, SOS, EOS, UNK] + list(vocab)  # Add special tokens to the vocabulary
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        # Ensure special tokens have fixed IDs
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID

    def text_to_sequence(self, text, tokenized=True):
        """ Convert text to sequence of token IDs. """
        sequence = [self.stoi[SOS]]
        tokens = text.split(' ') if tokenized else atomwise_tokenizer(text)
        for s in tokens:
            sequence.append(self.stoi.get(s, UNK_ID))  # Use UNK_ID for unknown tokens
        sequence.append(self.stoi[EOS])
        return sequence

    def sequence_to_text(self, sequence):
        """ Convert sequence of token IDs to text. """
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def predict_caption(self, sequence):
        """ Convert sequence of token IDs to caption, stopping at EOS or PAD tokens. """
        caption = ''
        for i in sequence:
            if i in [self.stoi[EOS], self.stoi[PAD]]:
                break
            caption += self.itos[i]
        return caption

class NodeTokenizer(Tokenizer):
    """
    Tokenizer for handling node-based inputs with optional separate X/Y coordinates.
    Supports continuous and discrete coordinate mappings.
    """
    def __init__(self, input_size=100, path=None, sep_xy=False, continuous_coords=False, debug=False):
        super().__init__(path)
        self.maxx = input_size  # Max X dimension
        self.maxy = input_size  # Max Y dimension
        self.sep_xy = sep_xy  # Separate X and Y coordinate IDs
        self.continuous_coords = continuous_coords
        self.debug = debug

    @property
    def offset(self):
        """ Return offset for coordinate token IDs. """
        return len(self.stoi)

    @property
    def output_constraint(self):
        """ Define whether output constraints apply (e.g., for continuous coordinates). """
        return not self.continuous_coords

    def fit_atom_symbols(self, atoms):
        """ Define vocabulary based on unique atom symbols. """
        vocab = self.special_tokens + list(set(atoms))
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        # Ensure special tokens have fixed IDs
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID
        assert self.stoi[MASK] == MASK_ID

    def x_to_id(self, x):
        """ Map X coordinate to token ID. """
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y):
        """ Map Y coordinate to token ID, considering separation if applicable. """
        if self.sep_xy:
            return self.offset + self.maxx + round(y * (self.maxy - 1))
        return self.offset + round(y * (self.maxy - 1))

    def id_to_x(self, id):
        """ Convert token ID to X coordinate in normalized space. """
        return (id - self.offset) / (self.maxx - 1)

    def id_to_y(self, id):
        """ Convert token ID to Y coordinate in normalized space, considering separation. """
        if self.sep_xy:
            return (id - self.offset - self.maxx) / (self.maxy - 1)
        return (id - self.offset) / (self.maxy - 1)

class CharTokenizer(NodeTokenizer):
    """
    Tokenizer for handling character-based tokenization of atom symbols.
    Extends NodeTokenizer with character-specific methods.
    """
    def fit_on_texts(self, texts):
        """ Fit tokenizer based on character-level vocabulary. """
        vocab = set()
        for text in texts:
            vocab.update(list(text))
        vocab.discard(' ')
        vocab = [PAD, SOS, EOS, UNK] + list(vocab)
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID

    def text_to_sequence(self, text, tokenized=True):
        """ Convert character-based text to sequence of token IDs. """
        sequence = [self.stoi[SOS]]
        tokens = list(text) if not tokenized else text.split(' ')
        for s in tokens:
            sequence.append(self.stoi.get(s, UNK_ID))
        sequence.append(self.stoi[EOS])
        return sequence

    def nodes_to_sequence(self, nodes):
        """ Convert nodes with coordinates and symbols to token ID sequence. """
        coords, symbols = nodes['coords'], nodes['symbols']
        labels = [SOS_ID]
        for (x, y), symbol in zip(coords, symbols):
            labels.append(self.x_to_id(x))
            labels.append(self.y_to_id(y))
            for char in symbol:
                labels.append(self.symbol_to_id(char))
        labels.append(EOS_ID)
        return labels

    def sequence_to_nodes(self, sequence):
        """ Convert token ID sequence back to nodes with coordinates and symbols. """
        coords, symbols = [], []
        i = 0
        if sequence[0] == SOS_ID:
            i += 1
        while i < len(sequence):
            if sequence[i] == EOS_ID:
                break
            if i+2 < len(sequence) and self.is_x(sequence[i]) and self.is_y(sequence[i+1]) and self.is_symbol(sequence[i+2]):
                x = self.id_to_x(sequence[i])
                y = self.id_to_y(sequence[i+1])
                symbol = ''.join([self.itos[sequence[j]] for j in range(i+2, len(sequence)) if self.is_symbol(sequence[j])])
                coords.append([x, y])
                symbols.append(symbol)
                i += len(symbol) + 2
            else:
                i += 1
        return {'coords': coords, 'symbols': symbols}

def get_tokenizer(args):
    """
    Factory function to initialize the appropriate tokenizer(s) based on configuration.
    
    Args:
        args (Namespace): Arguments defining tokenizer options.
    
    Returns:
        dict: Tokenizer(s) corresponding to specified formats.
    """
    tokenizer = {}
    for format_ in args.formats:
        if format_ == 'atomtok':
            tokenizer['atomtok'] = Tokenizer(args.vocab_file)
        elif format_ == "atomtok_coords":
            tokenizer["atomtok_coords"] = NodeTokenizer(args.coord_bins, args.vocab_file, args.sep_xy,
                                                        continuous_coords=args.continuous_coords)
        elif format_ == "chartok_coords":
            tokenizer["chartok_coords"] = CharTokenizer(args.coord_bins, args.vocab_file, args.sep_xy,
                                                        continuous_coords=args.continuous_coords)
    return tokenizer
