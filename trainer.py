import torch
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, device, use_amp=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(device_type='cuda') if self.use_amp else None

    def train_epoch(self, dataloader, batch_size):
        self.model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(dataloader, desc='train', disable=False):
            b_ids = batch['token_ids'].to(self.device)
            b_mask = batch['attention_mask'].to(self.device)
            b_labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    logits = self.model(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size
                loss.backward()
                self.optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        return train_loss / num_batches
