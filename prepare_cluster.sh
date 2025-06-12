#!/bin/bash
mkdir -p predictions
mkdir -p bert-base-uncased
# scp -r /path/to/local/bert-base-uncased/* username@cluster:/path/to/project/bert-base-uncased/
# python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased', cache_dir='./bert-base-uncased')"
chmod +x setup.sh
chmod +x submit_job.sh
./setup.sh
