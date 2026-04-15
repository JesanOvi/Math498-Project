from transformers import BertForSequenceClassification
import torch.nn as nn

class BERTClassifier(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=self.cfg.num_labels
        ).to(device)

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        self.model = BertForSequenceClassification.from_pretrained(path).to(self.device)