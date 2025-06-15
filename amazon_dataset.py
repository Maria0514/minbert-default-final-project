import csv
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
import torch

class AmazonFineFoodDataset(Dataset):
    """
    加载Amazon Fine Food Reviews数据，格式：
    Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,Score,Time,Summary,Text
    只用Text和Score，Score>=4为正面，<=2为负面，3为中性（可选是否过滤）
    """
    def __init__(self, csv_path, args=None, filter_neutral=True, max_samples=None):
        self.data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row['Text']
                score = int(float(row['Score']))
                if filter_neutral and score == 3:
                    continue
                # 二分类：1=正面(4,5)，0=负面(1,2)
                if score >= 4:
                    label = 1
                elif score <= 2:
                    label = 0
                else:
                    continue
                self.data.append((text, label))
                if max_samples and len(self.data) >= max_samples:
                    break
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/')
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = torch.LongTensor(labels)
        return token_ids, attention_mask, labels, sents

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents = self.pad_data(all_data)
        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents': sents
        }
        return batched_data 