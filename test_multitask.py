import torch
from types import SimpleNamespace
from multitask_classifier import MultitaskBERT
from datasets import load_multitask_data, SentenceClassificationDataset, SentencePairDataset
import torch.nn.functional as F
from optimizer import AdamW

# 配置参数（可根据实际情况调整）
class Args:
    sst_train = 'data/ids-sst-train.csv'
    para_train = 'data/quora-train.csv'
    sts_train = 'data/sts-train.csv'
    hidden_dropout_prob = 0.1
    option = 'finetune'
    batch_size = 2
    lr = 1e-4
    epochs = 1

if __name__ == '__main__':
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_train_dataset = SentenceClassificationDataset(sst_train_data, args)
    para_train_dataset = SentencePairDataset(para_train_data, args)
    sts_train_dataset = SentencePairDataset(sts_train_data, args, isRegression=True)

    sst_loader = torch.utils.data.DataLoader(sst_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=sst_train_dataset.collate_fn)
    para_loader = torch.utils.data.DataLoader(para_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=para_train_dataset.collate_fn)
    sts_loader = torch.utils.data.DataLoader(sts_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=sts_train_dataset.collate_fn)

    config = SimpleNamespace(hidden_dropout_prob=args.hidden_dropout_prob, num_labels=num_labels, hidden_size=768, data_dir='.', option=args.option)
    model = MultitaskBERT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    model.train()

    # 只训练1步，三任务各1个batch
    # 情感分析
    for batch in sst_loader:
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model.predict_sentiment(b_ids, b_mask)
        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
        loss.backward()
        optimizer.step()
        print('Sentiment train step loss:', loss.item())
        break
    # 同义句
    for batch in para_loader:
        b1 = batch['token_ids_1'].to(device)
        m1 = batch['attention_mask_1'].to(device)
        b2 = batch['token_ids_2'].to(device)
        m2 = batch['attention_mask_2'].to(device)
        labels = batch['labels'].float().to(device)
        optimizer.zero_grad()
        logits = model.predict_paraphrase(b1, m1, b2, m2)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        print('Paraphrase train step loss:', loss.item())
        break
    # 语义相似度
    for batch in sts_loader:
        b1 = batch['token_ids_1'].to(device)
        m1 = batch['attention_mask_1'].to(device)
        b2 = batch['token_ids_2'].to(device)
        m2 = batch['attention_mask_2'].to(device)
        labels = batch['labels'].float().to(device)
        optimizer.zero_grad()
        logits = model.predict_similarity(b1, m1, b2, m2)
        loss = F.mse_loss(logits, labels)
        loss.backward()
        optimizer.step()
        print('Similarity train step loss:', loss.item())
        break
    print('多任务BERT模型本地小batch训练验证通过！')
