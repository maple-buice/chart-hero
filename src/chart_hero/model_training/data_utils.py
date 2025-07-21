from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length.
    """
    spectrograms, labels = zip(*batch)

    # Squeeze the channel dimension and transpose (Freq, Time) -> (Time, Freq)
    spectrograms = [s.squeeze(0).transpose(0, 1) for s in spectrograms]
    spectrograms_padded = pad_sequence(spectrograms, batch_first=True, padding_value=0.0)
    # Transpose back (Time, Freq) -> (Freq, Time) and unsqueeze channel dimension
    spectrograms_padded = spectrograms_padded.transpose(1, 2).unsqueeze(1)

    # Pad labels
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0.0)

    return spectrograms_padded, labels_padded
