import json
import os
from datasets import load_from_disk, DatasetDict
import torch
import argparse
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, TrainerCallback, Trainer
from transformers.trainer_utils import EvalLoopOutput
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import gc
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report
from functools import partial

# --- (1) 정답 JSON을 "인덱스 없이" 변환하는 헬퍼 함수 ---
# (기존 코드의 NORMALIZATION_MAP을 여기에 붙여넣기)
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

def create_entity_label_list(spans_data):
    """
    원본 spans (JSON 문자열 또는 리스트)를 받아,
    인덱스를 제거한 [{"entity": "...", "label": "..."}] 리스트를 반환합니다.
    """
    try:
        spans = json.loads(spans_data) if isinstance(spans_data, str) else spans_data
        if not isinstance(spans, list):
            return []
    except json.JSONDecodeError:
        return []

    output_list = []
    for item in spans:
        if not isinstance(item, dict):
            continue
            
        entity = item.get("entity") or item.get("text") # 'entity' 또는 'text' 키 사용
        label = item.get("label") or item.get("type") # 'label' 또는 'type' 키 사용

        if not entity or not label:
            continue
            
        # 레이블 정규화 (예: 'name' -> '이름')
        normalized_label = NORMALIZATION_MAP.get(str(label).lower(), label)
        
        output_list.append({
            "entity": str(entity),
            "label": normalized_label
        })
    return output_list

# --- (2) SFTTrainer를 위한 프롬프트 포맷팅 함수 ---
def formatting_prompts_func(example, tokenizer):
    """
    SFTTrainer에 맞게 데이터를 변환합니다.
    '인덱스'를 요구하지 않고 '엔티티'와 '레이블'만 요구합니다.
    """
    # (수정) 지시문에서 "인덱스" 관련 내용 제거
    instruction = "주어진 문장에서 모든 개인 식별 정보(PII)를 찾아서, 각 PII의 'entity'(텍스트)와 'label'(종류)을 JSON 리스트 형식으로 추출하세요."
    input_text = example["sentence"]
    
    # (수정) 정답 데이터를 '인덱스 없는' JSON 리스트로 변환
    target_entities = create_entity_label_list(example["spans"])
    response_text = json.dumps(target_entities, ensure_ascii=False)
    
    # 모델이 학습할 최종 텍스트
    prompt = f"""### 지시: {instruction}\n\n### 입력:\n{input_text}\n\n### 답변:\n{response_text}{tokenizer.eos_token}"""
    return {"text": prompt}

def load_and_preprocess_dataset(args, tokenizer):
    """
    데이터셋을 로드하고 프롬프트 형식으로 전처리합니다.
    """
    if os.path.exists(args.processed_dataset_path):
        print(f"전처리된 데이터셋을 '{args.processed_dataset_path}'에서 로드합니다...")
        return load_from_disk(args.processed_dataset_path)

    print("전처리된 데이터셋이 없습니다. 새로 생성합니다...")
    
    # 원본 데이터 로드
    raw_train = load_from_disk(os.path.join(args.dataset_path, 'train'))
    raw_eval = load_from_disk(os.path.join(args.dataset_path, 'validation'))
    raw_test = load_from_disk(os.path.join(args.dataset_path, 'test'))

    # partial을 사용해 tokenizer를 map 함수에 전달
    
    map_func = partial(formatting_prompts_func, tokenizer=tokenizer)

    # .map() 적용
    processed_train = raw_train.map(map_func, remove_columns=raw_train.column_names, num_proc=os.cpu_count())
    processed_eval = raw_eval.map(map_func, remove_columns=raw_eval.column_names, num_proc=os.cpu_count())
    processed_test = raw_test.map(map_func, remove_columns=raw_test.column_names, num_proc=os.cpu_count())
    
    processed_dataset = DatasetDict({
        'train': processed_train,
        'validation': processed_eval,
        'test': processed_test
    })
    
    # 다음 실행을 위해 저장
    print(f"전처리된 데이터셋을 '{args.processed_dataset_path}' 경로에 저장합니다...")
    processed_dataset.save_to_disk(args.processed_dataset_path)
    
    return processed_dataset


# ------------------------------------------------------------------------------------------------------------------------------
# compute metrics
import numpy as np
import gc

def compute_metrics(eval_tuple, tokenizer):
    """
    Trainer 검증 단계에서 호출될 함수입니다.
    모델 예측(logits)을 디코딩하고 '인덱스 없는' JSON으로 파싱하여 F1을 계산합니다.
    """
    json_pattern = re.compile(r"(\[.*?\])", re.DOTALL)
    predictions, labels = eval_tuple
    
    pred_ids = np.argmax(predictions[0], axis=-1)
    del predictions
    gc.collect()

    labels[labels == -100] = tokenizer.pad_token_id
    
    # 1. 토큰 ID를 텍스트로 디코딩
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for pred_text, true_text in zip(decoded_preds, decoded_labels):
        
        # 2. 정답(label) JSON 파싱
        try:
            # '### 답변:\n' 뒷부분의 텍스트 추출
            true_json_str = true_text.split('### 답변:\n')[-1].strip()
            true_entities = json.loads(true_json_str)
        except (json.JSONDecodeError, IndexError):
            true_entities = []
            
        # 3. 예측(prediction) JSON 파싱
        try:
            # # '### 답변:' 뒷부분의 텍스트 추출
            # pred_json_str = pred_text.split('### 답변:')[-1].strip()
            # # 모델이 생성할 수 있는 ```json 마크다운 제거
            # if pred_json_str.startswith("```json"):
            #     pred_json_str = pred_json_str[7:].replace("```", "").strip()
            # pred_entities = json.loads(pred_json_str)
            match = json_pattern.search(pred_text)
            if match:
                pred_json_str = match.group(0)
                pred_entities = json.loads(pred_json_str)
        except (json.JSONDecodeError, IndexError):
            pred_entities = []

        # 4. (entity, label) 쌍을 비교하기 위해 set으로 변환
        # (정규화 로직 포함)
        true_set = set()
        for e in create_entity_label_list(true_entities): # 정규화 및 형식 통일
             true_set.add((e['entity'], e['label']))
             
        pred_set = set()
        for e in create_entity_label_list(pred_entities): # 정규화 및 형식 통일
            pred_set.add((e['entity'], e['label']))

        # 5. TP, FP, FN 누적
        total_tp += len(true_set.intersection(pred_set))
        total_fp += len(pred_set - true_set)
        total_fn += len(true_set - pred_set)

    # 6. 최종 Micro F1 스코어 계산
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"eval_f1_score": f1, "eval_precision": precision, "eval_recall": recall}


# ------------------------------------------------------------------------------------------------------------------------------
# post-process & train & test
def post_process_find_indices(sentence, pred_entities_list):
    """
    모델이 생성한 엔티티 리스트를 받아 원본 문장에서 인덱스를 찾습니다.
    """
    final_entities_with_indices = []
    
    # 중복 찾기를 방지하기 위해 이미 찾은 인덱스를 기록
    found_indices = set()

    for item in pred_entities_list:
        entity_text = item.get('entity')
        label = item.get('label')
        
        if not entity_text or not label:
            continue

        start_index = 0
        while True:
            # 이전에 찾지 않은 위치에서 'entity_text' 검색
            found_at = sentence.find(entity_text, start_index)
            
            if found_at == -1:
                break # 문장에서 더 이상 찾을 수 없음
                
            # 해당 위치가 이미 다른 엔티티의 일부로 사용되었는지 확인
            is_overlap = False
            for i in range(found_at, found_at + len(entity_text)):
                if i in found_indices:
                    is_overlap = True
                    break
            
            if not is_overlap:
                # 겹치지 않으면, 이 인덱스를 사용
                end_index = found_at + len(entity_text)
                final_entities_with_indices.append({
                    "entity": entity_text,
                    "label": label,
                    "start": found_at,
                    "end": end_index
                })
                # 사용된 인덱스 기록
                for i in range(found_at, end_index):
                    found_indices.add(i)
                break # 이 엔티티에 대한 검색 완료
            else:
                # 겹친다면, 이 위치 다음부터 다시 검색
                start_index = found_at + 1
                
    return final_entities_with_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kanana-8B PII NER Fine-tuning Script with Caching")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_path', type=str, default="/mnt/data3/Korean_abstraction/python/coreference/models/kanana-1.5-2.1b-instruct-2505")
    parser.add_argument('--dataset_path', type=str, default='./datasets/pii_ner_3dataset', help='Path to the raw dataset directory.')
    parser.add_argument('--processed_dataset_path', type=str, default='./processed_pii_dataset_kanana_new', help='Path to save/load the processed and split dataset.')
    parser.add_argument('--output_dir', type=str, default='./kanana_pii_finetuned_new')
    parser.add_argument('--lora_adapter_path', type=str, default='./kanana_pii_finetuned_new', help='학습된 LoRA 어댑터 가중치가 저장된 경로 (예: ./kanana_pii_finetuned)')
    
    args = parser.parse_args()
    
    # --- 공통 설정: 토크나이저 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 학습 모드 ---
    if args.mode == 'train':
        print("--- 파인튜닝 모드를 시작합니다 (인덱스 미포함) ---")
        
        processed_dataset = load_and_preprocess_dataset(args, tokenizer)
        train_dataset = processed_dataset['train']
        eval_dataset = processed_dataset['validation']

        # 2. 모델 로드 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # 4비트 양자화 타입 (NF4가 성능 저하가 가장 적음)
            bnb_4bit_compute_dtype=torch.bfloat16, # 계산 시 사용할 데이터 타입
            bnb_4bit_use_double_quant=True,    # 2차 양자화로 메모리 추가 절약
        )
        
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # accelerate 쓸때 사용
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            # device_map="auto",
            trust_remote_code=True
        )
        if hasattr(model, 'lm_head'):
            model.lm_head = model.lm_head.to(torch.float32)
        # 3. LoRA 설정 (Parameter-Efficient Fine-Tuning)
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters() # 학습 가능한 파라미터 수 출력

        training_args = SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            num_train_epochs=3,                     # 전체 데이터셋에 대한 학습 횟수
            per_device_train_batch_size=1,          # 장치당 학습 배치 크기
            per_device_eval_batch_size=1,           # 장치당 검증 배치 크기
            gradient_accumulation_steps=4,          # 그래디언트 축적 단계 (메모리 부족 시 유용)
            learning_rate=2e-5,                     # 학습률
            bf16=True,                              # bfloat16 사용 (A100 이상 GPU에서 효율적)
            logging_strategy="steps",
            logging_steps=100,       
            eval_strategy="steps",
            eval_steps=6000,
            eval_accumulation_steps=3, 
            save_strategy="steps",
            save_steps=6000,                          # 6000 스텝마다 체크포인트 저장
            load_best_model_at_end=True,            # 학습 종료 후 최적 모델 로드
            metric_for_best_model="eval_f1_score",  # 최적 모델 선택 기준
            save_total_limit=4,                     # 최대 4개의 체크포인트만 저장
            report_to="none",                       # WandB 등 로깅 비활성화
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=512,
            dataloader_num_workers=0,
        )
        
        compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer)

        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer, 
            compute_metrics=compute_metrics_with_tokenizer,
        )
        # trainer = CustomStreamTrainer(
        #     model=peft_model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=eval_dataset,
        #     processing_class=tokenizer,
        #     compute_metrics=compute_metrics_with_tokenizer,
        # )
        
        # (참고: CustomStreamTrainer를 사용하려면 compute_metrics 로직을 _calculate_batch_metrics로 옮겨야 함)

        # 4. 학습 시작
        print("파인튜닝을 시작합니다...")
        trainer.train()
        trainer.save_model(args.output_dir)

    elif args.mode == 'test':
        # 1. 모델 로딩 (기존과 동일)
        print("모델 로딩 중...")
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

        # 2. (수정) 테스트 데이터셋 로드 (전처리가 *안 된* 원본)
        test_dataset_raw = load_from_disk(os.path.join(args.dataset_path, 'test'))
        # (필요시 Subset 사용)
        # test_dataset_raw = Subset(test_dataset_raw, range(0, 100)) 
        print(f"테스트 데이터셋 로드 완료. 샘플 수: {len(test_dataset_raw)}")

        # 3. 추론 및 후처리
        all_true_entities_indexed = [] # 정답 (인덱스 포함)
        all_pred_entities_indexed = [] # 예측 (후처리 인덱스 포함)
        
        instruction = "주어진 문장에서 모든 개인 식별 정보(PII)를 찾아서, 각 PII의 'entity'(텍스트)와 'label'(종류)을 JSON 리스트 형식으로 추출하세요."

        for example in tqdm(test_dataset_raw, desc="Test 데이터셋 추론 및 후처리 중"):
            input_text = example["sentence"]
            
            # (A) 정답 준비: (수정!)
            # normalize_ground_truth 함수를 사용하여 인덱스가 포함된 정답을 파싱합니다.
            true_entities_indexed = normalize_ground_truth(example["spans"])
            all_true_entities_indexed.append(true_entities_indexed)
            
            # (B) 모델 추론: 프롬프트 생성 및 모델 호출 (기존과 동일)
            prompt = f"""### 지시: {instruction}\n\n### 입력:\n{input_text}\n\n### 답변:\n"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, # 토큰 길이는 데이터에 맞게 조절
                    eos_token_id=tokenizer.eos_token_id, 
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            
            # (C) 예측 파싱 (인덱스 없음, 기존과 동일)
            try:
                pred_json_str = response_text.split('### 답변:')[-1].strip()
                if pred_json_str.startswith("```json"):
                    pred_json_str = pred_json_str[7:].replace("```", "").strip()
                # 'create_entity_label_list'를 사용해 모델의 인덱스 없는 출력을 파싱
                pred_entities_no_index = create_entity_label_list(json.loads(pred_json_str))
            except Exception:
                pred_entities_no_index = []

            # (D) 후처리로 인덱스 찾기 (기존과 동일)
            pred_entities_with_index = post_process_find_indices(input_text, pred_entities_no_index)
            all_pred_entities_indexed.append(pred_entities_with_index)

        # 4. 최종 평가 (Micro F1) - (수정!)
        # (entity, label, start, end) 튜플을 비교하여 엄격한(Strict) F1을 계산합니다.
        total_tp, total_fp, total_fn = 0, 0, 0
        for true_list, pred_list in zip(all_true_entities_indexed, all_pred_entities_indexed):
            
            # (수정) (entity, label, start, end) 튜플로 set 생성
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
        
        # (수정)
        print("\n--- 최종 성능 평가 (Entity-level, Strict Index Match) ---")
        print(f"Micro F1-Score: {f1:.4f}")
        print(f"Micro Precision: {precision:.4f}")
        print(f"Micro Recall: {recall:.4f}")
        print(f"(Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn})")

        # (참고: 후처리된 인덱스 포함 결과 저장)
        with open("test_results_with_strict_indices.jsonl", "w", encoding="utf-8") as f:
            for i in range(len(all_true_entities_indexed)):
                f.write(json.dumps({
                    "sentence": test_dataset_raw[i]['sentence'],
                    "ground_truth": all_true_entities_indexed[i],
                    "prediction_with_index": all_pred_entities_indexed[i]
                }, ensure_ascii=False) + "\n")
        print("후처리된 인덱스 포함 결과가 'test_results_with_strict_indices.jsonl'에 저장되었습니다.")