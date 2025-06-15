import torch
import torch.nn as nn
from types import SimpleNamespace
from multitask_classifier import MultitaskBERT
from xcopa_dataset import XcopaDataset
from amazon_dataset import AmazonFineFoodDataset
from torch.utils.data import DataLoader
from collections import Counter

# ===== 配置参数 =====
class Args:
    xcopa_path = 'data/xcopa/zh/val.zh.jsonl'  # 可换为test.zh.jsonl
    amazon_path = 'data/amazon/Reviews.csv'
    batch_size = 32
    hidden_dropout_prob = 0.1
    option = 'finetune'
    num_labels = 2
    hidden_size = 768
    max_samples_amazon = None  # None表示全量

args = Args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 加载模型 =====
config = SimpleNamespace(hidden_dropout_prob=args.hidden_dropout_prob, num_labels=args.num_labels, hidden_size=args.hidden_size, data_dir='.', option=args.option)
model = MultitaskBERT(config).to(device)
model.eval()

# ===== 定义XCOPA简单线性头（未训练，仅演示推理流程） =====
xcopa_head = nn.Linear(model.bert.config.hidden_size, 1).to(device)
xcopa_head.eval()

# ===== XCOPA跨语言零样本迁移评测 =====
print("\n===== XCOPA 中文零样本迁移评测（零样本）=====")
xcopa_dataset = XcopaDataset(args.xcopa_path)
xcopa_loader = DataLoader(xcopa_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=xcopa_dataset.collate_fn)
correct, total = 0, 0
with torch.no_grad():
    for batch in xcopa_loader:
        input_ids = batch['input_ids'].to(device)         # [2*B, L]
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']                          # [2*B] or None
        batch_size = batch['batch_size']
        outputs = model.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs['pooler_output']             # [2*B, H]
        logits = xcopa_head(cls_output).squeeze(-1)       # [2*B]
        logits = logits.view(batch_size, 2)               # [B, 2]
        pred = logits.argmax(dim=1).cpu()                 # [B]
        if labels is not None:
            labels = labels.view(batch_size, 2)
            true_idx = labels.argmax(dim=1)
            correct += (pred == true_idx).sum().item()
            total += batch_size
if total > 0:
    print(f"XCOPA准确率: {correct/total:.4f}")
else:
    print("XCOPA无标签，未评估准确率")

# ===== Amazon Fine Food 微调训练（仅情感分类头+部分BERT解冻+类别加权） =====
print("\n===== Amazon Fine Food 微调训练（仅情感分类头+部分BERT解冻+类别加权）=====")
amazon_train_dataset = AmazonFineFoodDataset(args.amazon_path, max_samples=10000)  # 用更多数据
amazon_train_loader = DataLoader(amazon_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=amazon_train_dataset.collate_fn)
# 打印类别分布
labels_list = [x[1] for x in amazon_train_dataset.data]
counter = Counter(labels_list)
print('标签分布:', counter)
# 损失加权
weight = torch.tensor([1.0/counter[0], 1.0/counter[1]], device=device)
# 解冻BERT最后两层
for name, param in model.bert.named_parameters():
    if 'bert_layers.10' in name or 'bert_layers.11' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
for param in model.sentiment_classifier.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in amazon_train_loader:
        input_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits = model.predict_sentiment(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1} done, avg loss: {total_loss/len(amazon_train_loader):.4f}')
model.eval()
print('Amazon Fine Food情感分类头微调完成！')

# ===== Amazon Fine Food领域适配评测 =====
print("\n===== Amazon Fine Food领域适配评测 =====")
amazon_dataset = AmazonFineFoodDataset(args.amazon_path, max_samples=args.max_samples_amazon)
amazon_loader = DataLoader(amazon_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=amazon_dataset.collate_fn)
correct, total = 0, 0
with torch.no_grad():
    for batch in amazon_loader:
        input_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits = model.predict_sentiment(input_ids, attention_mask)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Amazon Fine Food情感分类准确率: {correct/total:.4f}") 