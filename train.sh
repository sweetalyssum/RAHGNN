#!/usr/bin/env bash
python run_summarization.py --mode=train --data_path="data/tmp_train.jsonl" --eval_path="data/tmp_train.jsonl" --test_path="data/train.jsonl" --vocab_path=data/vocab.txt --dataset_size=500000 --exp_name=base --max_enc_steps=30 --max_dec_steps=30 --min_dec_steps=5 --eval_every_step=10000
