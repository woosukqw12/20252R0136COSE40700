import argparse
import torch
from transformers import (
    AutoTokenizer,
    Trainer, 
    TrainingArguments, 
    DataCollatorForTokenClassification, 
    PreTrainedTokenizerFast, 
    AutoModelForTokenClassification,
    AutoConfig
)
import custom
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import Subset
from seqeval.metrics import precision_score, recall_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report
from sklearn.metrics import f1_score as sk_f1_score, cohen_kappa_score
from sklearn.metrics import classification_report as sk_classification_report
import numpy as np
import re
import pandas as pd 
import glob
import os

# --- 모델 및 토크나이저 경로 ---
model_name = "thunder-research-group/SNU_Thunder-DeID-1.5B"
tokenizer_path = "/mnt/data3/Korean_abstraction/python/SNU_Thunder-DeID-main/tokenizer/default_tokenizers/mecab_bpe_deid_128k"

# --- 데이터셋 경로 ---
train_path = './datasets/pii_ner_3dataset_for_thunder/train'
eval_path = './datasets/pii_ner_3dataset_for_thunder/validation'
test_path = './datasets/pii_ner_3dataset_for_thunder/test'


LABEL2ID = {
    "B-이름": 0, "I-이름": 1,
    "B-학교": 2, "I-학교": 3,
    "B-회사": 4, "I-회사": 5,
    "B-주소": 6, "I-주소": 7,
    "B-번호": 8, "I-번호": 9,
    "B-URL": 10, "I-URL": 11,
    "B-계좌번호": 12, "I-계좌번호": 13,
    "B-은행명": 14, "I-은행명": 15,
    "B-보안코드": 16, "I-보안코드": 17,
    "B-이메일": 18, "I-이메일": 19,
    "B-아이디": 20, "I-아이디": 21,
    "O": 22
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
num_labels = len(LABEL2ID) # 23


def compute_metric(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    # seqeval 계산을 위한 리스트 (엔티티 단위)
    true_seqs, pred_seqs = [], []
    true_seqs_binary, pred_seqs_binary = [], [] 
    
    # scikit-learn 계산을 위한 1차원 리스트 (토큰 단위)
    y_true_flat, y_pred_flat = [], []
    y_true_merged, y_pred_merged = [], []
    y_true_binary_flat, y_pred_binary_flat = [], []
    
    for pred_row, label_row in zip(preds, labels):
        true_seq_current, pred_seq_current = [], []
        true_seq_binary_current, pred_seq_binary_current = [], []
        
        for p_id, l_id in zip(pred_row, label_row):
            if l_id == -100:  # padding 무시
                continue

            # 👈 (수정) 전역 변수 ID2LABEL을 사용하도록 변경
            true_tag = ID2LABEL.get(int(l_id), "O")
            pred_tag = ID2LABEL.get(int(p_id), "O")
            
            # 1. seqeval용 데이터 추가 (엔티티 단위)
            true_seq_current.append(true_tag)
            pred_seq_current.append(pred_tag)

            # 2. scikit-learn용 데이터 추가 (토큰 단위)
            y_true_flat.append(true_tag)
            y_pred_flat.append(pred_tag)
            y_true_merged.append(true_tag[2:] if true_tag != 'O' else 'O')
            y_pred_merged.append(pred_tag[2:] if pred_tag != 'O' else 'O')
            
            y_true_binary_flat.append("O" if true_tag == "O" else "PII")
            y_pred_binary_flat.append("O" if pred_tag == "O" else "PII")
            true_seq_binary_current.append("O" if true_tag == "O" else "PII")
            pred_seq_binary_current.append("O" if pred_tag == "O" else "PII")

        true_seqs.append(true_seq_current)
        pred_seqs.append(pred_seq_current)
        true_seqs_binary.append(true_seq_binary_current)
        pred_seqs_binary.append(pred_seq_binary_current)

    # 다중 클래스 F1 (엔티티 단위, seqeval)
    multiclass_micro_f1 = seq_f1_score(true_seqs, pred_seqs, average="micro", zero_division=0)
    multiclass_weighted_f1 = seq_f1_score(true_seqs, pred_seqs, average="weighted", zero_division=0)
    entity_level_binary_f1 = seq_f1_score(true_seqs_binary, pred_seqs_binary, average="micro", zero_division=0)
    report_str = seq_classification_report(true_seqs, pred_seqs, digits=4, zero_division=0)
    
    # 토큰 단위 메트릭 계산 (sklearn)
    binary_f1 = sk_f1_score(y_true_binary_flat, y_pred_binary_flat, average="weighted", zero_division=0)
    all_bio_labels = list(LABEL2ID.keys())
    all_merged_labels = ['O'] # 'O'를 명시적으로 추가
    for label in all_bio_labels:
        if label != 'O' and label[2:] not in all_merged_labels:
            all_merged_labels.append(label[2:])
            
    token_merged_report_str = sk_classification_report(y_true_merged, y_pred_merged, labels=all_merged_labels, digits=4, zero_division=0)

    tokenlevel_micro_f1 = sk_f1_score(y_true_flat, y_pred_flat, average="micro", zero_division=0)
    kappa = cohen_kappa_score(y_true_flat, y_pred_flat)

    # 4. MUC-style 분석 (seqeval 리포트 재가공)
    report_dict = seq_classification_report(true_seqs, pred_seqs, output_dict=True, zero_division=0)
    compact_muc_report = {
        "label": [], 
        "Correct": [], 
        "Spurious": [], 
        "Missing": []
    }
    total_correct, total_spurious, total_missing = 0, 0, 0

    for label, metrics in report_dict.items():
        if label not in ["micro avg", "macro avg", "weighted avg"]:
            support = metrics.get('support', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            
            correct = int(round(recall * support))
            # Spurious (FP) 계산 수정: (Correct / Precision) - Correct
            spurious = int(round(correct / precision - correct)) if precision > 0 else (0 if correct == 0 else support)
            missing = support - correct
            
            compact_muc_report["label"].append(label)
            compact_muc_report["Correct"].append(correct)
            compact_muc_report["Spurious"].append(spurious)
            compact_muc_report["Missing"].append(missing)
            
            total_correct += correct
            total_spurious += spurious
            total_missing += missing
            
    # 전체 결과(OVERALL) 추가
    compact_muc_report["label"].append("OVERALL")
    compact_muc_report["Correct"].append(total_correct)
    compact_muc_report["Spurious"].append(total_spurious)
    compact_muc_report["Missing"].append(total_missing)
    
    return {
        "entity_level_micro_f1": multiclass_micro_f1,
        "entity_level_weighted_f1": multiclass_weighted_f1,
        "entity_level_binary_f1": entity_level_binary_f1,
        "token_level_micro_f1": tokenlevel_micro_f1,
        "token_level_binary_f1": binary_f1,
        "cohen_kappa": kappa,
        "muc_report": compact_muc_report,
        "entity_level_report": report_str,
        "token_level_report": token_merged_report_str
    }

# -----------------------------------------------------------------
# 3. 메인 실행 로직
# -----------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train/test/infer")
    parser.add_argument("--save_dir", type=str, default="./results/finetuned_thunder_deid")
    parser.add_argument("--dataset_path", type=str, default="./datasets")
    
    args = parser.parse_args()
    
    # 데이터셋 경로 설정
    if args.dataset_path == "./datasets":
        train_path = './datasets/pii_ner_3dataset_for_thunder/train'
        eval_path = './datasets/pii_ner_3dataset_for_thunder/validation'
        test_path = './datasets/pii_ner_3dataset_for_thunder/test'
    else:
        train_path = args.dataset_path + '/train'
        eval_path = args.dataset_path + '/validation'
        test_path = args.dataset_path + '/test'

    # 데이터셋 로드
    print("Loading datasets...")
    train_dataset = load_from_disk(train_path)
    eval_dataset = load_from_disk(eval_path)
    test_dataset = load_from_disk(test_path)
    print("Datasets loaded.")
    
    # 토크나이저 로드
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    # tokenizer = custom.switch_dummy(tokenizer)
    print("Tokenizer loaded.")
    
    max_len = 512
    data_collator = DataCollatorForTokenClassification(
        tokenizer, 
        padding='max_length', 
        max_length=max_len, 
        label_pad_token_id=-100
    )

    # --------------------------------------------------
    # 학습 모드
    # --------------------------------------------------
    if args.mode == "train":
        print("Starting [TRAIN] mode...")
        
        print("Dataset already uses 23 labels. Skipping relabeling.")

        # 2. (핵심) 모델 Config 수정 및 모델 로드
        print(f"Loading base model '{model_name}' and replacing head for {num_labels} labels.")
        
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            trust_remote_code=True,
        )

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
        print("Model loaded with new classification head.")

        # 3. TrainingArguments 설정
        training_args = TrainingArguments(
            output_dir=args.save_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_entity_level_micro_f1", 
            bf16=True, 
            report_to="tensorboard",
            logging_steps=100,
        )
        # train_dataset = Subset(train_dataset, list(range(100)))
        # eval_dataset = Subset(eval_dataset, list(range(100)))
        # 4. Trainer 생성
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metric, 
        )

        # 5. 학습 시작
        print("Starting training...")
        trainer.train()
        print("Training finished.")

        # 6. 최종 모델 저장
        print(f"Saving best model to {args.save_dir}")
        trainer.save_model(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print("Model saved.")

    # --------------------------------------------------
    # 테스트 모드
    # --------------------------------------------------
    elif args.mode == "test":
        print("Starting [TEST] mode...")
        torch.cuda.empty_cache()
        ckpt_path = '/mnt/data3/Korean_abstraction/python/coreference/results/finetuned_thunder_deid_1.5B/checkpoint-573300'

        print(f"Loading fine-tuned model from: {ckpt_path}")
        config = AutoConfig.from_pretrained(
            ckpt_path,
            trust_remote_code=True  
        )
        model = AutoModelForTokenClassification.from_pretrained(
            ckpt_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")
        print("Model loaded.")
        
        print("Test dataset already uses 23 labels. Skipping relabeling.")

        # 3. Trainer 생성 후 평가
        test_args = TrainingArguments(
            output_dir=args.save_dir + '/test_results',
            per_device_eval_batch_size=8, 
            bf16=True,
            dataloader_drop_last=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=test_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metric, 
        )
        
        print("Evaluating test dataset...")
        metrics = trainer.evaluate(test_dataset)
        
        # 👈 (수정) 상세한 메트릭 출력
        print("\n--- Test Results ---")
        print(f"Entity Level Micro F1      : {metrics.get('eval_entity_level_micro_f1', 0.0):.4f}")
        print(f"Entity Level Weighted F1   : {metrics.get('eval_entity_level_weighted_f1', 0.0):.4f}")
        print(f"Entity Level Binary F1 (PII/O) : {metrics.get('eval_entity_level_binary_f1', 0.0):.4f}")
        print(f"Token Level Micro F1       : {metrics.get('eval_token_level_micro_f1', 0.0):.4f}")
        print(f"Token Level Binary F1 (PII/O)  : {metrics.get('eval_token_level_binary_f1', 0.0):.4f}")
        print(f"Token Level Cohen's Kappa  : {metrics.get('eval_cohen_kappa', 0.0):.4f}")

        print("\n--- MUC-style Report (Correct, Spurious, Missing) ---")
        muc_report_data = metrics.get('eval_muc_report')
        if muc_report_data:
            df = pd.DataFrame(muc_report_data)
            print(df.to_string(index=False))
        else:
            print("MUC Report not found.")

        print("\n--- Entity Level Classification Report (seqeval) ---")
        print(metrics.get('eval_entity_level_report', 'Report not found.'))
        
        print("\n--- Token Level (Merged) Classification Report (sklearn) ---")
        print(metrics.get('eval_token_level_report', 'Report not found.'))