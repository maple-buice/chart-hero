from typing import List, Tuple

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """
    Custom collate function to pad sequences to the same length.
    Returns only (spectrograms, labels) for backward compatibility.
    """
    spectrograms_tup, labels_tup = zip(*batch)

    # Squeeze the channel dimension and transpose (Freq, Time) -> (Time, Freq)
    spectrograms = [s.squeeze(0).transpose(0, 1) for s in spectrograms_tup]
    spectrograms_padded = pad_sequence(
        spectrograms, batch_first=True, padding_value=0.0
    )
    # Transpose back (Time, Freq) -> (Freq, Time) and unsqueeze channel dimension
    spectrograms_padded = spectrograms_padded.transpose(1, 2).unsqueeze(1)

    # Pad labels
    labels = list(labels_tup)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)

    return spectrograms_padded, labels_padded


def collate_with_lengths(
    batch: List[Tuple[Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Collate variant that additionally returns original unpadded lengths [batch].
    """
    spectrograms_tup, labels_tup = zip(*batch)

    lengths = [s.shape[-1] for s in spectrograms_tup]

    spectrograms = [s.squeeze(0).transpose(0, 1) for s in spectrograms_tup]
    spectrograms_padded = pad_sequence(
        spectrograms, batch_first=True, padding_value=0.0
    )
    # Transpose back (Time, Freq) -> (Freq, Time) and unsqueeze channel dimension
    spectrograms_padded = spectrograms_padded.transpose(1, 2).unsqueeze(1)

    # Pad labels
    labels = list(labels_tup)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)

    # Return lengths as a 1D tensor [batch]
    lengths_tensor = Tensor(lengths).to(dtype=spectrograms_padded.dtype)

    return spectrograms_padded, labels_padded, lengths_tensor
