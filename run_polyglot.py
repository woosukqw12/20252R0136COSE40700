import json
import os
import re
from datasets import load_from_disk, DatasetDict
import torch
import argparse
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    Trainer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import gc
from tqdm import tqdm
from functools import partial
from sklearn.metrics import f1_score as sk_f1_score

# --- (0) 설정 및 맵핑 ---
NORMALIZATION_MAP = {
    'name': '이름', 'school': '학교', 'company': '회사', 'organization': '회사',
    'address': '주소', 'phone': '번호', 'number': '번호', 'url': 'URL',
    'account_number': '계좌번호', 'account': '계좌번호', 'bank': '은행명',
    'security_code': '보안코드', 'email': '이메일', 'id': '아이디',
    'user_id': '아이디', 'username': '아이디',
    '이름': '이름', '학교': '학교', '회사': '회사', '주소': '주소', '번호': '번호',
    'url': 'URL', '계좌번호': '계좌번호', '은행명': '은행명', '보안코드': '보안코드',
    '이메일': '이메일', '아이디': '아이디',
}

# --- (1) 헬퍼 함수 정의 ---

def create_entity_label_list(spans_data):
    """ 학습용: 인덱스를 제거한 [{"entity": "...", "label": "..."}] 리스트 반환 """
    try:
        spans = json.loads(spans_data) if isinstance(spans_data, str) else spans_data
        if not isinstance(spans, list): return []
    except json.JSONDecodeError: return []

    output_list = []
    for item in spans:
        if not isinstance(item, dict): continue
        entity = item.get("entity") or item.get("text")
        label = item.get("label") or item.get("type")
        if not entity or not label: continue
        
        normalized_label = NORMALIZATION_MAP.get(str(label).lower(), label)
        output_list.append({"entity": str(entity), "label": normalized_label})
    return output_list

def normalize_ground_truth(spans_data):
    """ 테스트용: 인덱스(start, end) 포함 정규화 """
    try:
        spans = json.loads(spans_data) if isinstance(spans_data, str) else spans_data
        if not isinstance(spans, list): return []
    except (json.JSONDecodeError, TypeError): return []

    output_list = []
    for item in spans:
        if not isinstance(item, dict): continue
        entity = item.get("entity") or item.get("text")
        label = item.get("label") or item.get("type")
        start = item.get("start")
        end = item.get("end")
        
        if not entity or not label or start is None or end is None: continue
        normalized_label = NORMALIZATION_MAP.get(str(label).lower(), label)
        output_list.append({"entity": str(entity), "label": normalized_label, "start": int(start), "end": int(end)})
    return output_list

def post_process_find_indices(sentence, pred_entities_list):
    """ 문자열 검색을 통해 엔티티의 start, end 인덱스를 찾습니다. """
    final_entities = []
    found_indices = set()

    for item in pred_entities_list:
        entity_text = item.get('entity')
        label = item.get('label')
        if not entity_text or not label: continue

        start_index = 0
        while True:
            found_at = sentence.find(entity_text, start_index)
            if found_at == -1: break
            
            # 이미 찾은 영역과 겹치는지 확인
            is_overlap = False
            for i in range(found_at, found_at + len(entity_text)):
                if i in found_indices:
                    is_overlap = True; break
            
            if not is_overlap:
                end_index = found_at + len(entity_text)
                final_entities.append({"entity": entity_text, "label": label, "start": found_at, "end": end_index})
                for i in range(found_at, end_index): found_indices.add(i)
                break # Greedy matching
            else:
                start_index = found_at + 1
    return final_entities

def convert_to_bio_tags(tokenizer, input_text, entities_with_idx):
    """
    문장과 인덱스가 포함된 엔티티 리스트를 받아 Token-level BIO 태그 리스트를 반환합니다.
    """
    # 1. 토크나이즈 (Offset Mapping 포함)
    tokenized = tokenizer(input_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = tokenized['offset_mapping']
    
    # 2. 문자 단위 라벨링 (초기화)
    char_labels = ['O'] * len(input_text)
    
    for ent in entities_with_idx:
        start, end = ent['start'], ent['end']
        label = ent['label']
        if start < len(input_text) and end <= len(input_text):
            char_labels[start] = f"B-{label}"
            for i in range(start + 1, end):
                char_labels[i] = f"I-{label}"
    
    # 3. 토큰 단위 라벨링 (각 토큰의 시작 문자 태그를 따름)
    token_labels = []
    for start_offset, end_offset in offsets:
        # 공백만 있는 토큰 등 예외 처리
        if start_offset >= len(char_labels):
            token_labels.append('O')
            continue
        token_labels.append(char_labels[start_offset])
        
    return token_labels

# --- (2) 데이터 전처리 함수 ---
def formatting_prompts_func(example, tokenizer):
    instruction = "주어진 문장에서 모든 개인 식별 정보(PII)를 찾아서, 각 PII의 'entity'(텍스트)와 'label'(종류)을 JSON 리스트 형식으로 추출하세요."
    input_text = example["sentence"]
    target_entities = create_entity_label_list(example["spans"])
    response_text = json.dumps(target_entities, ensure_ascii=False)
    prompt = f"""### 지시: {instruction}\n\n### 입력:\n{input_text}\n\n### 답변:\n{response_text}{tokenizer.eos_token}"""
    return {"text": prompt}

def load_and_preprocess_dataset(args, tokenizer):
    if os.path.exists(args.processed_dataset_path):
        print(f"전처리된 데이터셋 로드: {args.processed_dataset_path}")
        return load_from_disk(args.processed_dataset_path)

    print("데이터셋 전처리 시작...")
    raw_train = load_from_disk(os.path.join(args.dataset_path, 'train'))
    raw_eval = load_from_disk(os.path.join(args.dataset_path, 'validation'))
    try: raw_test = load_from_disk(os.path.join(args.dataset_path, 'test'))
    except: raw_test = raw_eval

    map_func = partial(formatting_prompts_func, tokenizer=tokenizer)
    proc_kwargs = {"num_proc": os.cpu_count(), "remove_columns": raw_train.column_names}
    
    processed_dataset = DatasetDict({
        'train': raw_train.map(map_func, **proc_kwargs),
        'validation': raw_eval.map(map_func, **proc_kwargs),
        'test': raw_test.map(map_func, **proc_kwargs)
    })
    processed_dataset.save_to_disk(args.processed_dataset_path)
    return processed_dataset

# --- (3) 핵심: Compute Metrics (Entity & Token Level) ---

def compute_metrics(eval_tuple, tokenizer):
    json_pattern = re.compile(r"(\[.*?\])", re.DOTALL)
    predictions, labels = eval_tuple
    
    pred_ids = np.argmax(predictions[0], axis=-1)
    del predictions; gc.collect()
    
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Entity-level 통계 변수
    ent_tp, ent_fp, ent_fn = 0, 0, 0          # Micro (Class 구분)
    ent_bin_tp, ent_bin_fp, ent_bin_fn = 0, 0, 0 # Binary (PII vs Non-PII)
    
    # Token-level 데이터 리스트
    all_true_tokens = []
    all_pred_tokens = []
    
    for pred_text, true_text in zip(decoded_preds, decoded_labels):
        # 1. 원본 문장 복원
        try:
            input_part = true_text.split('### 입력:\n')[1]
            input_text = input_part.split('\n\n### 답변:')[0].strip()
        except IndexError:
            input_text = ""
            
        # 2. 정답/예측 JSON 파싱
        try: true_json = json.loads(true_text.split('### 답변:\n')[-1].strip())
        except: true_json = []
        
        try:
            match = json_pattern.search(pred_text)
            pred_json_str = match.group(0) if match else "[]"
            pred_json = json.loads(pred_json_str)
        except: pred_json = []
        
        # 리스트 생성
        true_list = create_entity_label_list(true_json)
        pred_list = create_entity_label_list(pred_json)

        # --- [Entity-level Micro F1] (클래스 구분 O) ---
        true_set = set((e['entity'], e['label']) for e in true_list)
        pred_set = set((e['entity'], e['label']) for e in pred_list)
        
        ent_tp += len(true_set.intersection(pred_set))
        ent_fp += len(pred_set - true_set)
        ent_fn += len(true_set - pred_set)

        # --- [Entity-level Binary F1] (클래스 구분 X, 모두 'PII'로 취급) ---
        # 라벨을 모두 'PII'로 변경하여 Set 생성
        true_bin_set = set((e['entity'], 'PII') for e in true_list)
        pred_bin_set = set((e['entity'], 'PII') for e in pred_list)

        ent_bin_tp += len(true_bin_set.intersection(pred_bin_set))
        ent_bin_fp += len(pred_bin_set - true_bin_set)
        ent_bin_fn += len(true_bin_set - pred_bin_set)
        
        # --- [Token-level] ---
        if input_text:
            true_ents_idx = post_process_find_indices(input_text, true_list)
            pred_ents_idx = post_process_find_indices(input_text, pred_list)
            
            true_bio = convert_to_bio_tags(tokenizer, input_text, true_ents_idx)
            pred_bio = convert_to_bio_tags(tokenizer, input_text, pred_ents_idx)
            
            min_len = min(len(true_bio), len(pred_bio))
            all_true_tokens.extend(true_bio[:min_len])
            all_pred_tokens.extend(pred_bio[:min_len])

    # --- Metric Calculation ---
    
    # 1. Entity-level Micro F1
    prec = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) > 0 else 0
    rec = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) > 0 else 0
    ent_f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    # 2. Entity-level Binary F1
    bin_prec = ent_bin_tp / (ent_bin_tp + ent_bin_fp) if (ent_bin_tp + ent_bin_fp) > 0 else 0
    bin_rec = ent_bin_tp / (ent_bin_tp + ent_bin_fn) if (ent_bin_tp + ent_bin_fn) > 0 else 0
    ent_bin_f1 = 2 * (bin_prec * bin_rec) / (bin_prec + bin_rec) if (bin_prec + bin_rec) > 0 else 0

    # 3. Token-level Micro F1 (Class 구분)
    labels_except_o = [l for l in sorted(list(set(all_true_tokens + all_pred_tokens))) if l != 'O']
    token_micro_f1 = sk_f1_score(all_true_tokens, all_pred_tokens, average='micro', labels=labels_except_o, zero_division=0)
    
    # 4. Token-level Binary F1 (PII vs Non-PII)
    # 'O'는 0, 나머지는 1로 변환
    bin_true_tokens = [0 if t == 'O' else 1 for t in all_true_tokens]
    bin_pred_tokens = [0 if t == 'O' else 1 for t in all_pred_tokens]
    
    # pos_label=1 (PII)에 대한 Binary F1 계산
    token_bin_f1 = sk_f1_score(bin_true_tokens, bin_pred_tokens, average='binary', pos_label=1, zero_division=0)
    
    return {
        "entity_micro_f1": ent_f1,
        "entity_binary_f1": ent_bin_f1,   
        "token_micro_f1": token_micro_f1,
        "token_binary_f1": token_bin_f1,  
        "entity_precision": prec,
        "entity_recall": rec
    }

# --- (4) 메인 실행 ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default="./models/polygolt-ko-1.3b")
    parser.add_argument('--dataset_path', type=str, default='./datasets/pii_ner_3dataset')
    parser.add_argument('--processed_dataset_path', type=str, default='./processed_pii_dataset_polyglot')
    parser.add_argument('--output_dir', type=str, default='./polyglot_pii_finetuned')
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    if args.mode == 'train':
        processed_ds = load_and_preprocess_dataset(args, tokenizer)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map={"":0}, trust_remote_code=True
        )
        if hasattr(model, 'lm_head'): model.lm_head = model.lm_head.to(torch.float32)
        model = prepare_model_for_kbit_training(model)
        
        # Polyglot(GPT-NeoX)용 모듈
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        training_args = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,
            bf16=True,
            logging_steps=500,
            eval_strategy="steps",
            eval_steps=5000, # 자주 확인하기 위해 500으로 설정
            save_strategy="steps",
            save_steps=5000,
            metric_for_best_model="entity_micro_f1", # Best Model 기준
            load_best_model_at_end=True,
            max_length=512,
            dataset_text_field="text"
        )
        
        # partial로 tokenizer 전달
        compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer)

        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=processed_ds['train'],
            eval_dataset=processed_ds['validation'],
            processing_class=tokenizer,
            compute_metrics=compute_metrics_func
        )
        
        print("학습 시작...")
        trainer.train()
        trainer.save_model(args.output_dir)

    # --- 테스트 모드 ---
    elif args.mode == 'test':
        print("--- 테스트 모드 (Strict Index Match) ---")
        
        # 1. 모델 로드
        print(f"모델 로딩 중... ({args.model_path})")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
        model = model.merge_and_unload()
        model.eval()
        print("모델 로딩 완료.")

        # 2. 테스트 데이터셋 로드 (Raw Data)
        test_dataset_raw = load_from_disk(os.path.join(args.dataset_path, 'test'))
        # test_dataset_raw = Subset(test_dataset_raw, range(0, 50)) # 디버깅용
        print(f"테스트 데이터셋 로드 완료. 샘플 수: {len(test_dataset_raw)}")

        # 3. 추론 및 후처리
        all_true_entities_indexed = [] 
        all_pred_entities_indexed = []
        
        instruction = "주어진 문장에서 모든 개인 식별 정보(PII)를 찾아서, 각 PII의 'entity'(텍스트)와 'label'(종류)을 JSON 리스트 형식으로 추출하세요."

        for example in tqdm(test_dataset_raw, desc="Test 추론 중"):
            input_text = example["sentence"]
            
            # (A) 정답 준비 (인덱스 포함 정규화)
            true_entities_indexed = normalize_ground_truth(example["spans"])
            all_true_entities_indexed.append(true_entities_indexed)
            
            # (B) 모델 추론
            prompt = f"""### 지시: {instruction}\n\n### 입력:\n{input_text}\n\n### 답변:\n"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, 
                    eos_token_id=tokenizer.eos_token_id, 
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            
            # (C) 예측 파싱 (인덱스 없는 상태)
            pred_entities_no_index = []
            try:
                # JSON 부분만 추출 시도
                match = re.search(r"(\[.*?\])", response_text, re.DOTALL)
                if match:
                    pred_json_str = match.group(0)
                    pred_entities_no_index = create_entity_label_list(json.loads(pred_json_str))
                else:
                    # 직접 파싱 시도
                    pred_json_str = response_text.split('### 답변:')[-1].strip()
                    if pred_json_str.startswith("```json"):
                        pred_json_str = pred_json_str[7:].replace("```", "").strip()
                    pred_entities_no_index = create_entity_label_list(json.loads(pred_json_str))
            except Exception:
                pass # 파싱 실패 시 빈 리스트

            # (D) 후처리: 인덱스 복원
            pred_entities_with_index = post_process_find_indices(input_text, pred_entities_no_index)
            all_pred_entities_indexed.append(pred_entities_with_index)

        # 4. 최종 평가 (Strict Match)
        total_tp, total_fp, total_fn = 0, 0, 0
        for true_list, pred_list in zip(all_true_entities_indexed, all_pred_entities_indexed):
            
            # (entity, label, start, end) 튜플 비교
            true_set = set(
                (e['entity'], e['label'], e['start'], e['end']) 
                for e in true_list
            )
            pred_set = set(
                (e['entity'], e['label'], e['start'], e['end']) 
                for e in pred_list
            )
            
            total_tp += len(true_set.intersection(pred_set))
            total_fp += len(pred_set - true_set)
            total_fn += len(true_set - pred_set)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n" + "="*50)
        print("최종 성능 평가 (Entity-level, Strict Index Match)")
        print("="*50)
        print(f"Micro F1-Score  : {f1:.4f}")
        print(f"Micro Precision : {precision:.4f}")
        print(f"Micro Recall    : {recall:.4f}")
        print(f"Counts          : TP={total_tp}, FP={total_fp}, FN={total_fn}")
        
        # 결과 저장
        with open("polyglot_strict_results.jsonl", "w", encoding="utf-8") as f:
            for i in range(len(all_true_entities_indexed)):
                f.write(json.dumps({
                    "sentence": test_dataset_raw[i]['sentence'],
                    "ground_truth": all_true_entities_indexed[i],
                    "prediction": all_pred_entities_indexed[i]
                }, ensure_ascii=False) + "\n")
        print("상세 결과가 'polyglot_strict_results.jsonl'에 저장되었습니다.")