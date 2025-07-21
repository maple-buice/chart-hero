import torch
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length.
    """
    spectrograms, labels = zip(*batch)

    # Pad spectrograms
    spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=0.0)

    # Pad labels
    labels = pad_sequence(labels, batch_first=True, padding_value=0.0)

    return spectrograms, labels
