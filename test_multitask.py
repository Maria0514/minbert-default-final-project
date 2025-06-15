import torch
from types import SimpleNamespace
from multitask_classifier import MultitaskBERT
from datasets import load_multitask_data, SentenceClassificationDataset, SentencePairDataset
from optimizer import AdamW
from evaluation import model_eval_multitask
import torch.nn.functional as F
import torch.cuda.amp
from xcopa_dataset import XcopaDataset
from amazon_dataset import AmazonFineFoodDataset

# 配置参数（可根据实际情况调整）
class Args:
    sst_train = 'data/ids-sst-train.csv'
    para_train = 'data/quora-train.csv'
    sts_train = 'data/sts-train.csv'
    sst_dev = 'data/ids-sst-dev.csv'
    para_dev = 'data/quora-dev.csv'
    sts_dev = 'data/sts-dev.csv'
    hidden_dropout_prob = 0.1
    option = 'finetune'
    batch_size = 4
    lr = 1e-4
    epochs = 1

if __name__ == '__main__':
    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练和验证数据
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='train')

    # 只用部分数据加速训练和验证
    sst_train_data = sst_train_data[:500]
    para_train_data = para_train_data[:500]
    sts_train_data = sts_train_data[:500]
    sst_dev_data = sst_dev_data[:200]
    para_dev_data = para_dev_data[:200]
    sts_dev_data = sts_dev_data[:200]

    sst_train_dataset = SentenceClassificationDataset(sst_train_data, args)
    para_train_dataset = SentencePairDataset(para_train_data, args)
    sts_train_dataset = SentencePairDataset(sts_train_data, args, isRegression=True)

    sst_dev_dataset = SentenceClassificationDataset(sst_dev_data, args)
    para_dev_dataset = SentencePairDataset(para_dev_data, args)
    sts_dev_dataset = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sst_loader = torch.utils.data.DataLoader(sst_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=sst_train_dataset.collate_fn)
    para_loader = torch.utils.data.DataLoader(para_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=para_train_dataset.collate_fn)
    sts_loader = torch.utils.data.DataLoader(sts_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=sts_train_dataset.collate_fn)

    sst_dev_loader = torch.utils.data.DataLoader(sst_dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=sst_dev_dataset.collate_fn)
    para_dev_loader = torch.utils.data.DataLoader(para_dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=para_dev_dataset.collate_fn)
    sts_dev_loader = torch.utils.data.DataLoader(sts_dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=sts_dev_dataset.collate_fn)

    config = SimpleNamespace(hidden_dropout_prob=args.hidden_dropout_prob, num_labels=num_labels, hidden_size=768, data_dir='.', option=args.option)
    model = MultitaskBERT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        # 训练情感分类
        for batch in sst_loader:
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
            loss.backward()
            optimizer.step()
        # 训练同义句
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
        # 训练语义相似度
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

        # 验证
        print(f"Epoch {epoch+1} 验证中...")
        eval_results = model_eval_multitask(
            sst_dev_loader, para_dev_loader, sts_dev_loader, model, device
        )
        # unpack主要指标
        para_acc = eval_results[0]
        sts_corr = eval_results[1]
        sst_acc = eval_results[2]
        print(f"Epoch {epoch+1} | 验证集 Paraphrase Acc: {para_acc:.4f} | STS Pearson: {sts_corr:.4f} | SST Acc: {sst_acc:.4f}")

    print('多任务BERT模型训练和验证流程结束！')

    # ========== XCOPA 中文零样本迁移评测 ==========
    print("\n===== XCOPA 中文零样本迁移评测（零样本）=====")
    xcopa_path = 'data/xcopa/zh/val.zh.jsonl'  # 可换为test.zh.jsonl
    xcopa_dataset = XcopaDataset(xcopa_path)
    from torch.utils.data import DataLoader
    xcopa_loader = DataLoader(xcopa_dataset, batch_size=8, shuffle=False, collate_fn=xcopa_dataset.collate_fn)
    model.eval()
    correct, total = 0, 0
    import torch
    with torch.no_grad():
        for batch in xcopa_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = batch['batch_size']
            outputs = model.bert(input_ids, attention_mask=attention_mask)
            logits = outputs['pooler_output']
            # 假设有二分类头或直接用CLS向量做线性分类
            # 这里只做示例，实际需加一层分类器
            # 这里假设labels为2*batch_size，前一半为choice1，后一半为choice2
            # 选最大logit为预测
            # 这里只做占位，需根据你模型结构调整
            # pred = ...
            # correct += (pred == batch['labels']).sum().item()
            # total += len(pred)
            pass  # TODO: 按你的模型结构补全推理与评测
    print(f"XCOPA评测完成（请补全推理与评测代码）")

    # ========== Amazon Fine Food领域适配评测 ==========
    print("\n===== Amazon Fine Food领域适配评测 =====")
    amazon_path = 'data/amazon/Reviews.csv'
    amazon_dataset = AmazonFineFoodDataset(amazon_path, max_samples=1000)
    amazon_loader = DataLoader(amazon_dataset, batch_size=8, shuffle=False, collate_fn=amazon_dataset.collate_fn)
    model.eval()
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

    # ===== Amazon Fine Food 微调训练流程示例 =====
    print("\n===== Amazon Fine Food 微调训练流程（仅情感分类头）=====")
    amazon_train_dataset = AmazonFineFoodDataset('data/amazon/Reviews.csv', max_samples=10000)
    amazon_train_loader = torch.utils.data.DataLoader(amazon_train_dataset, batch_size=32, shuffle=True, collate_fn=amazon_train_dataset.collate_fn)
    # 只训练情感分类头，冻结BERT参数
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.sentiment_classifier.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.sentiment_classifier.parameters(), lr=2e-4)
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in amazon_train_loader:
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model.predict_sentiment(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1} done, avg loss: {total_loss/len(amazon_train_loader):.4f}')
    print('Amazon Fine Food情感分类头微调完成！')
