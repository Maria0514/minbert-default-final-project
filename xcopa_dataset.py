import json
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
import torch

class XcopaDataset(Dataset):
    """
    加载XCOPA数据（以中文为例），每条数据格式：
    {
        "premise": "...",
        "choice1": "...",
        "choice2": "...",
        "question": "cause" 或 "effect",
        "label": 0/1
    }
    """
    def __init__(self, jsonl_path, args=None):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/')
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 任务格式：输入(premise, question, choice1/2)，输出label
        # 这里拼接为：premise + [SEP] + question + [SEP] + choice
        premise = item['premise']
        question = item['question']  # 'cause' or 'effect'
        choice1 = item['choice1']
        choice2 = item['choice2']
        label = item.get('label', -1)  # 测试集无label
        # 返回两个选项
        return {
            'premise': premise,
            'question': question,
            'choice1': choice1,
            'choice2': choice2,
            'label': label
        }

    def collate_fn(self, batch):
        # batch: list of dict
        input_texts = []
        labels = []
        for item in batch:
            # 拼接输入：premise [SEP] question [SEP] choice1/2
            for i, choice in enumerate([item['choice1'], item['choice2']]):
                text = f"{item['premise']} [SEP] {item['question']} [SEP] {choice}"
                input_texts.append(text)
                # label: 1表示正确选项，0表示错误
                if item['label'] != -1:
                    labels.append(1 if i == item['label'] else 0)
        encoding = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
        if labels:
            labels = torch.LongTensor(labels)
        else:
            labels = None
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels,
            'batch_size': len(batch)
        } 