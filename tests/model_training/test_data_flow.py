import unittest

import torch.nn.functional as F

from chart_hero.model_training.transformer_config import auto_detect_config
from chart_hero.model_training.transformer_data import create_data_loaders
from chart_hero.model_training.transformer_model import create_model


class TestDataFlow(unittest.TestCase):
    def test_data_flow(self):
        config = auto_detect_config()
        config.data_dir = "datasets/processed_segmented"

        train_loader, _, _ = create_data_loaders(config, config.data_dir)

        model = create_model(config)

        for batch in train_loader:
            spectrograms, labels = batch

            print(f"spectrograms shape: {spectrograms.shape}")
            print(f"labels shape: {labels.shape}")

            outputs = model(spectrograms)
            logits = outputs["logits"]

            print(f"logits shape: {logits.shape}")

            pool_size = config.patch_size[0]
            labels = F.max_pool1d(
                labels.transpose(1, 2), kernel_size=pool_size, stride=pool_size
            ).transpose(1, 2)

            print(f"labels shape after pooling: {labels.shape}")

            break


if __name__ == "__main__":
    unittest.main()
