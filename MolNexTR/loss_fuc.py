""" Loss functions for training """

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .tokenization import PAD_ID, MASK, MASK_ID

class GraphLoss(nn.Module):
    """
    Loss function for graph-based predictions, including coordinates and edge classification.

    - Coordinates loss: L1 loss masked by valid coordinates.
    - Edge classification loss: CrossEntropy loss with a custom weighting scheme.
    """
    def __init__(self):
        super(GraphLoss, self).__init__()
        # Weighting the CrossEntropy loss for edge classification, higher weight for classes > 0
        weight = torch.ones(7) * 10
        weight[0] = 1  # Class 0 (no bond) has lower weight
        self.criterion = nn.CrossEntropyLoss(weight, ignore_index=-100)

    def forward(self, outputs, targets):
        results = {}
        # Calculate coordinate regression loss if present in outputs
        if 'coords' in outputs:
            pred = outputs['coords']
            max_len = pred.size(1)
            target = targets['coords'][:, :max_len]
            mask = target.ge(0)
            loss = F.l1_loss(pred, target, reduction='none')
            results['coords'] = (loss * mask).sum() / mask.sum()
        
        # Calculate edge classification loss if present in outputs
        if 'edges' in outputs:
            pred = outputs['edges']
            max_len = pred.size(-1)
            target = targets['edges'][:, :max_len, :max_len]
            results['edges'] = self.criterion(pred, target)
        
        return results


class LabelSmoothingLoss(nn.Module):
    """
    Applies label smoothing to CrossEntropy loss, which helps prevent overconfidence in predictions.

    Args:
        label_smoothing (float): Amount of smoothing (between 0 and 1).
        tgt_vocab_size (int): Size of the target vocabulary.
        ignore_index (int): Index to ignore during loss computation (e.g., padding token).
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()
        self.ignore_index = ignore_index

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        Computes the smoothed CrossEntropy loss.

        Args:
            output (torch.FloatTensor): Model predictions, size (batch_size, n_classes).
            target (torch.LongTensor): True labels, size (batch_size).

        Returns:
            torch.FloatTensor: Calculated KL-divergence loss.
        """
        # Convert output logits to log probabilities
        log_probs = F.log_softmax(output, dim=-1)

        # Prepare smoothed label distribution
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        # Compute KL-divergence loss
        return F.kl_div(log_probs, model_prob, reduction='batchmean')


class SequenceLoss(nn.Module):
    """
    Loss function for sequence modeling tasks, with optional label smoothing.

    Args:
        label_smoothing (float): Amount of label smoothing to apply.
        vocab_size (int): Size of the vocabulary.
        ignore_index (int): Index to ignore during loss computation.
        ignore_indices (list): List of additional indices to ignore (e.g., MASK_ID).
    """
    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, ignore_indices=[]):
        super(SequenceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ignore_indices = ignore_indices

        # Choose appropriate loss criterion based on label smoothing parameter
        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        else:
            self.criterion = LabelSmoothingLoss(label_smoothing, vocab_size, ignore_index)

    def forward(self, output, target):
        """
        Computes the loss for sequence modeling tasks.

        Args:
            output (torch.FloatTensor): Model predictions, size (batch, len, vocab).
            target (torch.LongTensor): True labels, size (batch, len).

        Returns:
            torch.FloatTensor: Calculated loss value.
        """
        batch_size, max_len, vocab_size = output.size()
        output = output.reshape(-1, vocab_size)
        target = target.reshape(-1)
        
        # Apply ignore indices masking, if specified
        for idx in self.ignore_indices:
            if idx != self.ignore_index:
                target.masked_fill_((target == idx), self.ignore_index)
        
        return self.criterion(output, target)


class Criterion(nn.Module):
    """
    Wrapper class for managing different loss functions for various output formats.

    Args:
        args (Namespace): Arguments containing model configuration.
        tokenizer (Tokenizer): Tokenizer containing vocabulary and special tokens.
    """
    def __init__(self, args, tokenizer):
        super(Criterion, self).__init__()
        criterion = {}

        # Define loss criterion based on output format
        for format_ in args.formats:
            if format_ == 'edges':
                criterion['edges'] = GraphLoss()
            else:
                # Configure SequenceLoss with appropriate ignore indices
                if MASK in tokenizer[format_].stoi:
                    ignore_indices = [PAD_ID, MASK_ID]
                else:
                    ignore_indices = []
                criterion[format_] = SequenceLoss(args.label_smoothing, len(tokenizer[format_]),
                                                  ignore_index=PAD_ID, ignore_indices=ignore_indices)
        
        self.criterion = nn.ModuleDict(criterion)

    def forward(self, results, refs):
        """
        Calculates the loss for each output format using the respective criterion.

        Args:
            results (dict): Model predictions for each output format.
            refs (dict): Ground truth references for each output format.

        Returns:
            dict: Losses for each format.
        """
        losses = {}
        for format_ in results:
            predictions, targets, *_ = results[format_]
            loss_ = self.criterion[format_](predictions, targets)
            
            # Handle multi-component loss and aggregation
            if type(loss_) is dict:
                losses.update(loss_)
            else:
                losses[format_] = loss_.mean() if loss_.numel() > 1 else loss_

        return losses
