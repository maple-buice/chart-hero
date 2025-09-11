from typing import List, Tuple

from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """Collate function that supports sequential spectrogram windows."""

    spectrograms_tup, labels_tup = zip(*batch)

    first = spectrograms_tup[0]
    # Sequential samples: [L, 1, F, T]
    if first.dim() == 4:
        spectrograms = torch.stack(list(spectrograms_tup), dim=0)
        labels = torch.stack(list(labels_tup), dim=0)
        return spectrograms, labels

    # Single-window samples: [1, F, T]
    spectrograms = [s.squeeze(0).transpose(0, 1) for s in spectrograms_tup]
    spectrograms_padded = pad_sequence(
        spectrograms, batch_first=True, padding_value=0.0
    )
    spectrograms_padded = spectrograms_padded.transpose(1, 2).unsqueeze(1)

    labels = list(labels_tup)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)

    return spectrograms_padded, labels_padded


def collate_with_lengths(
    batch: List[Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Collate variant that additionally returns sequence lengths.

    Works with datasets that optionally include per-sample length tensors.
    """

    if len(batch[0]) == 3:
        spectrograms_tup, labels_tup, length_tup = zip(*batch)  # type: ignore[assignment]
    else:
        spectrograms_tup, labels_tup = zip(*batch)  # type: ignore[misc]
        length_tup = None

    first = spectrograms_tup[0]
    if first.dim() == 4:
        spectrograms = torch.stack(list(spectrograms_tup), dim=0)
        labels = torch.stack(list(labels_tup), dim=0)
        if length_tup is not None:
            lengths_tensor = torch.stack(
                [torch.as_tensor(l, dtype=torch.long) for l in length_tup], dim=0
            )
        else:
            lengths = [[s.shape[-1] for _ in range(s.shape[0])] for s in spectrograms_tup]
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        return spectrograms, labels, lengths_tensor

    if length_tup is not None:
        lengths = [int(l) for l in length_tup]
    else:
        lengths = [s.shape[-1] for s in spectrograms_tup]
    spectrograms = [s.squeeze(0).transpose(0, 1) for s in spectrograms_tup]
    spectrograms_padded = pad_sequence(
        spectrograms, batch_first=True, padding_value=0.0
    )
    spectrograms_padded = spectrograms_padded.transpose(1, 2).unsqueeze(1)

    labels = list(labels_tup)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return spectrograms_padded, labels_padded, lengths_tensor
