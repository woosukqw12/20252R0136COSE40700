import os
import json
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
from datasets import load_from_disk
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report
# --- 1. 평가 함수 정의 ---

class CustomStreamTrainer(SFTTrainer):
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        메모리 효율적인 평가 루프로 재정의합니다.
        logits를 누적하는 대신, 배치마다 메트릭을 계산하고 중간값만 누적합니다.
        """
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        total_tp, total_fp, total_fn = 0, 0, 0
        total_eval_loss = 0.0
        num_eval_samples = 0

        for step, inputs in tqdm(enumerate(dataloader)):
            # --- 👇 핵심 수정 부분 ---
            # prediction_step의 반환값에 의존하지 않고, inputs에서 직접 labels를 가져옵니다.
            # SFTTrainer의 데이터 콜레이터는 'labels' 키를 만들어주므로 이 키를 사용합니다.
            labels = inputs.get("labels")
            if labels is None:
                # 만약을 위한 대비책: labels 키가 없다면 input_ids를 사용합니다.
                labels = inputs.get("input_ids")
            # --- 👆 ---

            with torch.no_grad():
                # 이제 prediction_step에서 반환되는 labels는 사용하지 않으므로 _로 받습니다.
                loss, logits, _ = self.prediction_step(
                    model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
                )
            
            # 직접 가져온 'labels' 변수를 사용하므로 len() 오류가 발생하지 않습니다.
            total_eval_loss += loss.item() * len(labels)
            num_eval_samples += len(labels)

            # --- (이후 로직은 동일) ---
            pred_ids = torch.argmax(logits, axis=-1)
            
            labels = labels.cpu().numpy()
            pred_ids = pred_ids.cpu().numpy()
            
            batch_tp, batch_fp, batch_fn = self._calculate_batch_metrics(labels, pred_ids)
            total_tp += batch_tp
            total_fp += batch_fp
            total_fn += batch_fn
            
            del loss, logits, labels, pred_ids
            gc.collect()

        # 최종 메트릭 계산
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            f"{metric_key_prefix}_loss": total_eval_loss / num_eval_samples if num_eval_samples > 0 else 0,
            f"{metric_key_prefix}_f1_score": f1,
        }

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_eval_samples)

    def _calculate_batch_metrics(self, labels, pred_ids):
        """
        단일 배치에 대해 TP, FP, FN을 계산하는 헬퍼 함수입니다.
        """
        labels[labels == -100] = self.processing_class.pad_token_id
        batch_tp, batch_fp, batch_fn = 0, 0, 0

        for i in range(pred_ids.shape[0]):
            true_text = self.processing_class.decode(labels[i], skip_special_tokens=True)
            pred_text = self.processing_class.decode(pred_ids[i], skip_special_tokens=True)

            true_entities, pred_entities = [], []
            try: # 정답 파싱
                true_json_str = true_text.split('### 답변:\n')[-1]
                true_entities = json.loads(true_json_str)
            except (json.JSONDecodeError, IndexError): pass
            
            try: # 예측 파싱
                pred_json_str = pred_text.split('### 답변:')[-1].strip()
                if pred_json_str.startswith("```json"): pred_json_str = pred_json_str[7:-3].strip()
                pred_entities = json.loads(pred_json_str)
            except (json.JSONDecodeError, IndexError): pass

            true_set = {json.dumps(e, sort_keys=True) for e in true_entities}
            pred_set = {json.dumps(e, sort_keys=True) for e in pred_entities}
            
            batch_tp += len(true_set.intersection(pred_set))
            batch_fp += len(pred_set - true_set)
            batch_fn += len(true_set - pred_set)
            
        return batch_tp, batch_fp, batch_fn

NORMALIZATION_MAP = {
    'name': '이름',
    'school': '학교',
    'company': '회사',
    'organization': '회사', # 'organization'도 '회사'로 처리
    'address': '주소',
    'phone': '번호',
    'number': '번호',
    'url': 'URL',
    'account_number': '계좌번호',
    'account': '계좌번호',
    'bank': '은행명',
    'security_code': '보안코드',
    'email': '이메일',
    'id': '아이디',
    'user_id': '아이디',
    'username': '아이디',
}

def normalize_pii_types(entity_list):
    """
    예측된 엔티티 리스트를 받아 PII 타입의 값을 정규화하고,
    'type' 키를 'label' 키로 변경합니다.
    """
    if not isinstance(entity_list, list):
        return []

    normalized_list = []
    for entity in entity_list:
        # 엔티티가 딕셔너리가 아닌 경우를 대비
        if not isinstance(entity, dict): continue
            
        new_entity = entity.copy()
        
        # 'type' 키가 있는지 먼저 확인
        if "type" in new_entity:
            # 1. 'type' 키의 값을 가져와 소문자로 변환합니다 (예: 'Name' -> 'name')
            original_type_value = new_entity.get("type", "").lower()
            
            # 2. NORMALIZATION_MAP을 사용해 값을 정규화합니다 (예: 'name' -> '이름')
            # 맵에 없는 값이면 원래 값을 유지합니다.
            normalized_type_value = NORMALIZATION_MAP.get(original_type_value, new_entity.get("type"))
            
            # 3. 'label'이라는 새로운 키에 정규화된 값을 할당합니다.
            new_entity["label"] = normalized_type_value
            
            # 4. 기존의 'type' 키는 삭제합니다.
            del new_entity["type"]
        elif "label" in new_entity:
            # 'label' 키가 이미 있는 경우에도 값을 정규화합니다.
            original_label_value = new_entity.get("label", "").lower()
            normalized_label_value = NORMALIZATION_MAP.get(original_label_value, new_entity.get("label"))
            new_entity["label"] = normalized_label_value

        normalized_list.append(new_entity)
        
    return normalized_list

def evaluate_predictions(true_entities_list, pred_entities_list):
    """
    예측된 PII(JSON)와 실제 PII(JSON)를 비교하여
    개체(Entity) 단위의 Precision, Recall, F1-score를 계산합니다.
    """
    total_tp, total_fp, total_fn = 0, 0, 0

    for true_entities, pred_entities in zip(true_entities_list, pred_entities_list):
        # JSON 객체를 정렬된 문자열로 변환하여 비교의 일관성을 보장합니다.
        true_set = {json.dumps(e, sort_keys=True) for e in true_entities}
        pred_set = {json.dumps(e, sort_keys=True) for e in pred_entities}

        tp = len(true_set.intersection(pred_set)) # 정답과 예측이 모두 일치
        fp = len(pred_set - true_set)             # 예측은 했지만 정답에는 없음
        fn = len(true_set - pred_set)             # 정답이지만 예측하지 못함
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # F1 스코어 계산
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1-score": f1}


def compute_metrics(eval_tuple):
    """
    Trainer의 검증 단계에서 호출될 함수입니다.
    모델의 예측(logits)을 디코딩하고 JSON으로 파싱하여 F1-score를 계산합니다.
    """
    predictions, labels = eval_tuple
    
    # 예측 결과에서 가장 확률이 높은 토큰 ID를 선택합니다.
    pred_ids = np.argmax(predictions[0], axis=-1)
    del predictions
    gc.collect()
    # 레이블에서 패딩 토큰(-100)을 tokenizer의 pad_token_id로 변경합니다.
    labels[labels == -100] = tokenizer.pad_token_id
    
    # 2. 중간 리스트 생성을 최소화하고 스트리밍 방식으로 F1 스코어의 구성 요소(tp, fp, fn)를 직접 누적합니다.
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # 데이터를 하나씩 순회
    for i in range(pred_ids.shape[0]):
        # 개별 샘플에 대해 디코딩
        true_text = tokenizer.decode(labels[i], skip_special_tokens=True)
        pred_text = tokenizer.decode(pred_ids[i], skip_special_tokens=True)
        
        true_entities = []
        pred_entities = []

        # 정답 JSON 파싱
        try:
            true_json_str = true_text.split('### 답변:\n')[-1]
            true_entities = json.loads(true_json_str)
        except (json.JSONDecodeError, IndexError):
            pass # 실패 시 빈 리스트 유지
        
        # 예측 JSON 파싱
        try:
            pred_json_str = pred_text.split('### 답변:')[-1].strip()
            if pred_json_str.startswith("```json"):
                pred_json_str = pred_json_str[7:-3].strip()
            pred_entities = json.loads(pred_json_str)
        except (json.JSONDecodeError, IndexError):
            pass # 실패 시 빈 리스트 유지

        # 개별 샘플에 대한 tp, fp, fn 계산
        true_set = {json.dumps(e, sort_keys=True) for e in true_entities}
        pred_set = {json.dumps(e, sort_keys=True) for e in pred_entities}
        
        total_tp += len(true_set.intersection(pred_set))
        total_fp += len(pred_set - true_set)
        total_fn += len(true_set - pred_set)

    # 누적된 값으로 최종 F1 스코어 계산
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"eval_f1_score": f1}

def extract_json_from_text(text):
    # JSON 객체는 '{'로, JSON 배열은 '['로 시작합니다.
    first_brace = text.find('{')
    first_bracket = text.find('[')

    # 중괄호나 대괄호가 전혀 없으면 JSON이 없는 것으로 간주
    if first_brace == -1 and first_bracket == -1:
        return None
    
    # 더 먼저 나오는 것으로 시작점을 정함
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_index = first_brace
    else:
        start_index = first_bracket

    # 시작점부터 문자열을 잘라내어 JSONDecoder로 파싱 시도
    text_to_decode = text[start_index:]
    try:
        # raw_decode는 첫 번째 유효한 JSON 객체만 파싱하고, 나머지 텍스트는 무시합니다.
        decoded_json, _ = json.JSONDecoder().raw_decode(text_to_decode)
        return decoded_json
    except json.JSONDecodeError:
        return None

def convert_json_to_bio_tags(text, entities, tokenizer):
    """
    원본 텍스트와 PII 엔티티(JSON)를 받아 토큰 단위 BIO 태그 시퀀스를 생성합니다.
    """
    # 1. 원본 텍스트를 기준으로 문자 단위 태그 배열 생성
    
    char_tags = ['O'] * len(text)
    for entity in entities:
        if not isinstance(entity, dict):
            # print("Not dict")
            continue

        label = entity.get("label") or entity.get("type")
        # start = int(entity.get("start"))
        # end = int(entity.get("end"))
        start_val = entity.get("start")
        end_val = entity.get("end")

        # 3. 필수 값들이 모두 존재하는지, label이 문자열인지 확인
        if not (label and isinstance(label, str) and start_val is not None and end_val is not None):
            continue

        # 4. start, end를 정수형으로 변환 (타입이 int 또는 str.isdigit()인지 확인)
        start, end = None, None
        
        if isinstance(start_val, int):
            start = start_val
        elif isinstance(start_val, str) and start_val.isdigit():
            start = int(start_val)
        
        if isinstance(end_val, int):
            end = end_val
        elif isinstance(end_val, str) and end_val.isdigit():
            end = int(end_val)

        # 5. start, end가 성공적으로 변환되었는지, 그리고 유효한 범위인지 최종 확인
        # (예: start가 end보다 크거나, 텍스트 길이를 벗어나는 경우 등)
        if start is None or end is None or start >= end or end > len(text):
            continue


        # if label is None or start is None or end is None:
        #     # print("None")
        #     continue
        
        if start < len(text) and end <= len(text):
            char_tags[start] = f'B-{label}'
            for i in range(start + 1, end):
                char_tags[i] = f'I-{label}'

    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']
    # print(f'offsets: {offsets}')
    # print(f'text: {text}')
    # print(f'char_tags: {char_tags}')
    token_tags = []
    for (start, end) in offsets:
        # offset이 (0, 0)인 특수 토큰은 이미 add_special_tokens=False로 제외됨
        # if start == end: continue
        token_tags.append(char_tags[start])
        
    return token_tags

    # 2. 토큰화하고, 각 토큰에 해당하는 BIO 태그 매핑
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']
    
    token_tags = []
    for token, (start, end) in zip(tokens, offsets):
        # [CLS], [SEP], [PAD] 등 특수 토큰은 제외
        if start == end:
            continue
        
        # 각 토큰의 태그는 해당 토큰의 시작 문자의 태그를 따름
        token_tags.append(char_tags[start])
        
    return token_tags
# --- 2. 메인 스크립트 실행 ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kanana-8B PII NER Fine-tuning Script with Caching")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_path', type=str, default="/mnt/data3/Korean_abstraction/python/coreference/models/kanana-1.5-2.1b-instruct-2505")
    parser.add_argument('--dataset_path', type=str, default='./datasets/pii_ner_3merged_08061_v3', help='Path to the raw dataset directory.')
    parser.add_argument('--processed_dataset_path', type=str, default='./processed_pii_dataset_kanana', help='Path to save/load the processed and split dataset.')
    parser.add_argument('--output_dir', type=str, default='./kanana_pii_finetuned')
    parser.add_argument('--lora_adapter_path', type=str, default='./kanana_pii_finetuned', help='학습된 LoRA 어댑터 가중치가 저장된 경로 (예: ./kanana_pii_finetuned)')
    
    
    args = parser.parse_args()

    # --- 공통 설정: 토크나이저 로드 ---
    # `add_eos_token=True`는 입력의 끝을 명확히 알려주어 모델의 답변 생성을 돕습니다.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token # PAD 토큰을 EOS 토큰으로 설정
    args.dataset_path = './datasets/pii_ner_3ds_norm'
    # args.dataset_path = '/mnt/data3/Korean_abstraction/python/coreference/datasets/pii_ner_only_thunder_kluebert'
    # --- 학습 모드 ---
    if args.mode == 'train':
        print("--- 파인튜닝 모드를 시작합니다 ---")
        
        # 1. 데이터셋 로드 및 프롬프트 형식으로 변환
        if os.path.exists(args.processed_dataset_path):
            print(f"전처리된 데이터셋을 '{args.processed_dataset_path}'에서 로드합니다...")
            train_dataset = load_from_disk(os.path.join(args.processed_dataset_path, 'train'))
            eval_dataset = load_from_disk(os.path.join(args.processed_dataset_path, 'validation'))
        else:
            print("전처리된 데이터셋이 없습니다. 새로 생성합니다...")

            # 2. 프롬프트 형식으로 변환
            def formatting_prompts_func(example):
                instruction = "주어진 문장에서 모든 개인 식별 정보(PII)를 찾아서, 각 PII의 종류, 시작 인덱스, 끝 인덱스를 JSON 형식으로 추출하세요."
                input_text = example["sentence"]
                response_text = example["spans"]
                prompt = f"""### 지시: {instruction}\n\n### 입력:\n{input_text}\n\n### 답변:\n{response_text}{tokenizer.eos_token}"""
                return {"text": prompt}
            
            # dataset = dataset.map(formatting_prompts_func, num_proc=os.cpu_count()) # 병렬 처리로 속도 향상
            full_train_dataset = load_from_disk(os.path.join(args.dataset_path, 'train'))
            train_dataset = full_train_dataset.map(formatting_prompts_func, remove_columns=full_train_dataset.column_names)
            train_dataset.save_to_disk(os.path.join(args.processed_dataset_path, 'train'))
            
            full_eval_dataset = load_from_disk(os.path.join(args.dataset_path, 'validation'))
            eval_dataset = full_eval_dataset.map(formatting_prompts_func, remove_columns=full_eval_dataset.column_names)
            eval_dataset.save_to_disk(os.path.join(args.processed_dataset_path, 'validation'))
            
            full_test_dataset = load_from_disk(os.path.join(args.dataset_path, 'test'))
            test_dataset = full_test_dataset.map(formatting_prompts_func, remove_columns=full_test_dataset.column_names)
            test_dataset.save_to_disk(os.path.join(args.processed_dataset_path, 'test'))
            
            # # 4. 다음 실행을 위해 디스크에 저장
            # print(f"전처리된 데이터셋을 '{args.processed_dataset_path}' 경로에 저장합니다...")
            # train_dataset.save_to_disk(os.path.join(args.processed_dataset_path, 'train'))
            # eval_dataset.save_to_disk(os.path.join(args.processed_dataset_path, 'validation'))


        def formatting_prompts_func(example):
            """ SFTTrainer에 맞는 형식으로 데이터를 변환하는 함수 """
            instruction = "주어진 문장에서 모든 개인 식별 정보(PII)를 찾아서, 각 PII의 종류, 시작 인덱스, 끝 인덱스를 JSON 형식으로 추출하세요."
            input_text = example["sentence"]
            response_text = example["spans"]

            # SFTTrainer는 이 형식에서 '### 답변:' 뒷부분을 정답(label)으로 인식하여 학습합니다.
            # EOS 토큰을 답변 끝에 추가하여 모델이 답변을 마치는 시점을 학습하도록 합니다.
            prompt = f"""### 지시: {instruction}\n\n### 입력:\n{input_text}\n\n### 답변:\n{response_text}{tokenizer.eos_token}"""
            return {"text": prompt}
        # print(train_dataset[0]); quit()
        # dataset = dataset.map(formatting_prompts_func)
        # train_dataset = train_dataset.map(formatting_prompts_func)
        # eval_dataset = eval_dataset.map(formatting_prompts_func)

        # 2. 모델 로드 (8-bit 양자화 적용)
        # BitsAndBytesConfig를 사용하여 GPU 메모리 사용량을 줄입니다.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # 4비트 양자화 타입 (NF4가 성능 저하가 가장 적음)
            bnb_4bit_compute_dtype=torch.bfloat16, # 계산 시 사용할 데이터 타입
            bnb_4bit_use_double_quant=True,    # 2차 양자화로 메모리 추가 절약
        )
        
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            # device_map="auto",
            trust_remote_code=True
        )
        
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

        # 4. 트레이너 설정
        training_args = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=2,                     # 전체 데이터셋에 대한 학습 횟수
            per_device_train_batch_size=16,          # 장치당 학습 배치 크기
            per_device_eval_batch_size=1,           # 장치당 검증 배치 크기
            gradient_accumulation_steps=4,          # 그래디언트 축적 단계 (메모리 부족 시 유용)
            learning_rate=2e-5,                     # 학습률
            bf16=True,                              # bfloat16 사용 (A100 이상 GPU에서 효율적)
            logging_strategy="steps",
            logging_steps=10,       
            eval_strategy="steps",
            eval_steps=6000,
            eval_accumulation_steps=3, 
            save_strategy="steps",
            save_steps=6000,                          # 50 스텝마다 체크포인트 저장
            load_best_model_at_end=True,            # 학습 종료 후 최적 모델 로드
            metric_for_best_model="eval_f1_score",  # 최적 모델 선택 기준
            save_total_limit=4,                     # 최대 4개의 체크포인트만 저장
            report_to="none",                       # WandB 등 로깅 비활성화
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=1024,
            dataloader_num_workers=0,
        )
        
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer = CustomStreamTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        
        # 5. 학습 시작
        print("파인튜닝을 시작합니다...")
        trainer.train()
        print("학습 완료! 최종 모델을 저장합니다.")
        trainer.save_model(args.output_dir)

    elif args.mode == 'test':
        # 1. 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token

        # 2. 모델 로딩 (베이스 모델 + 학습된 LoRA 어댑터 병합)
        print("베이스 모델 및 LoRA 어댑터 로딩 중...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # 저장된 LoRA 가중치를 베이스 모델에 병합
        model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
        model = model.merge_and_unload() # 추론 속도 향상을 위해 병합
        model.eval()
        print("모델 로딩 및 병합 완료.")

        # 3. 테스트 데이터셋 로드 (전처리 전 원본 데이터셋)
        try:
            test_dataset = load_from_disk(os.path.join(args.dataset_path, 'test'))
            test_dataset = Subset(test_dataset, range(1500, 5000))
            print(f"테스트 데이터셋 로드 완료. 샘플 수: {len(test_dataset)}")
        except FileNotFoundError:
            print(f"오류: '{args.dataset_path}/test' 경로에서 테스트 데이터셋을 찾을 수 없습니다.")
            quit()
            

        # 4. 전체 테스트 데이터셋에 대해 추론 실행
        all_true_entities = []
        all_pred_entities = []
        all_true_bio_tags_nested = [] 
        all_pred_bio_tags_nested = []
        all_true_bio_tags = []
        all_pred_bio_tags = []
        instruction = "주어진 문장에서 모든 개인 식별 정보(PII)를 찾아서, 각 PII의 종류, 시작 인덱스, 끝 인덱스를 JSON 형식으로 추출하세요."

        for example in tqdm(test_dataset, desc="Test 데이터셋 추론 중"):
            input_text = example["sentence"]
            true_bio_tags = []
            pred_bio_tags = []
            # 정답(spans)을 파싱하여 정답 리스트에 추가
            try:
                spans_data = example.get("spans")
                if spans_data: # spans 데이터가 None이거나 비어있지 않은 경우에만 처리
                    true_entities = json.loads(spans_data) if isinstance(spans_data, str) else spans_data
                    # print(f"Original true_entities: {true_entities}")
                    if true_entities: # 파싱 후에도 엔티티가 실제로 있는 경우
                        true_bio_tags = convert_json_to_bio_tags(input_text, true_entities, tokenizer)
                    # print("true:", true_bio_tags)
                    # true_bio_tags = true_entities
                # # spans_data의 타입이 문자열(str)인지 확인
                # true_entities = json.loads(spans_data) if isinstance(spans_data, str) else spans_data
                
                # # all_true_entities.append(true_entities)
                
                # true_bio_tags = convert_json_to_bio_tags(input_text, true_entities, tokenizer)

            except (json.JSONDecodeError, TypeError): # TypeError도 처리
                print("Error parsing spans_data:", spans_data)
                continue
                # all_true_entities.append([]) # 실패 시 빈 리스트 추가
                
            if true_bio_tags is None:
                # continue
                tokens = tokenizer(input_text, add_special_tokens=False)['input_ids']
                true_bio_tags = ['O'] * len(tokens)
            # print('true:',all_true_entities)
            # 추론을 위한 프롬프트 생성
            prompt = f"""### 지시: {instruction}\n\n### 입력:\n{input_text}\n\n### 답변:\n"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # 모델 생성
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1024, 
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 생성된 결과에서 프롬프트를 제외한 답변 부분만 디코딩
            response_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            parsed_json = extract_json_from_text(response_text)
            if parsed_json and isinstance(parsed_json, dict):
                pred_entities = parsed_json.get("PII", [])
                normalized_pred_entities = normalize_pii_types(pred_entities)
                pred_bio_tags = convert_json_to_bio_tags(input_text, normalized_pred_entities, tokenizer)
            else:
                try:
                    start_idx = response_text.find('```json')
                    end_idx = response_text.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                        # 정확한 JSON 문자열 부분만 추출합니다.
                        pred_json_str = response_text[start_idx : end_idx + 1]
                        pred_json_str = pred_json_str[7:].strip()
                        # print(f'pred_json_str: >>>{pred_json_str}<<<')
                        # 2. 추출된 문자열을 JSON으로 파싱합니다.
                        parsed_json = json.loads(pred_json_str)
                        
                        # 3. {"PII": [...]} 구조에서 실제 엔티티 리스트를 가져옵니다.
                        # 만약 'PII' 키가 없으면 빈 리스트를 반환합니다.
                        pred_entities = parsed_json.get("PII", [])
                        # print(f'pred_entities: {pred_entities}')
                        normalized_pred_entities = normalize_pii_types(pred_entities)
                        # all_pred_entities.append(normalized_pred_entities)
                        pred_bio_tags = convert_json_to_bio_tags(input_text, pred_entities, tokenizer)
                    # else:
                    #     # JSON 객체를 찾지 못한 경우
                    #     all_pred_entities.append([])

                except (json.JSONDecodeError, KeyError):
                    # 파싱에 실패하거나 "PII" 키가 없는 경우
                    pred_bio_tags = ['O'] * len(true_bio_tags)
                    
                # all_pred_entities.append([])
            # print('pred:', all_pred_entities)
            # if  true_bio_tags is not None and len(true_bio_tags) == len(pred_bio_tags):
            if len(true_bio_tags) == len(pred_bio_tags):
                all_true_bio_tags.extend(true_bio_tags)
                all_pred_bio_tags.extend(pred_bio_tags)
                all_true_bio_tags_nested.append(true_bio_tags)
                all_pred_bio_tags_nested.append(pred_bio_tags)
        
        # # 길이가 다를 경우 평가의 일관성을 위해 제외 (일반적으로 발생하지 않음)
        # if len(true_bio_tags) == len(pred_bio_tags):
        #     all_pred_bio_tags.extend(pred_bio_tags)
        # else:
        #     # 길이가 다를 경우, 정답 태그 리스트에서도 해당 샘플을 제거
        #     del all_true_bio_tags[-1*len(true_bio_tags):]
        # # for 반복문 이후, F1 계산 전
        
        
        print("\nEntity-level 추론 결과를 'entity_results.jsonl' 파일로 저장합니다...")
        with open("b_entity_results.jsonl", "w", encoding="utf-8") as f:
            for true_ents, pred_ents in zip(all_true_bio_tags_nested, all_pred_bio_tags_nested):
                line = json.dumps({
                    "true_entities": true_ents,
                    "pred_entities": pred_ents
                }, ensure_ascii=False)
                f.write(line + "\n")
        print("저장 완료.")

        # 4-2. Token-level 결과를 JSONL로 저장
        print("Token-level 추론 결과를 'bio_tag_results.jsonl' 파일로 저장합니다...")
        with open("bio_token_tag_results.jsonl", "w", encoding="utf-8") as f:
            for true_tag, pred_tag in zip(all_true_bio_tags, all_pred_bio_tags):
                line = json.dumps({"true_token": true_tag, "pred_token": pred_tag}, ensure_ascii=False)
                f.write(line + "\n")
        print("저장 완료.")
        
        # print(f"all_true_bio_tags_nested: {all_true_bio_tags_nested}")
        # print(f"all_pred_bio_tags_nested: {all_pred_bio_tags_nested}")
        # print(f"all_true_bio_tags: {all_true_bio_tags}")
        # print(f"all_pred_bio_tags: {all_pred_bio_tags}")
        
        if not all_true_bio_tags_nested:
            print("평가할 유효한 샘플이 없습니다.")
            
        else:
            entity_micro_f1 = seq_f1_score(all_true_bio_tags_nested, all_pred_bio_tags_nested, average="micro", zero_division=0)
            print(f"\nEntity-level Micro F1-Score: {entity_micro_f1:.4f}")

            binary_true_nested = [
                ['O' if tag == 'O' else f"{tag.split('-')[0]}-PII" for tag in seq]
                for seq in all_true_bio_tags_nested
            ]
            binary_pred_nested = [
                ['O' if tag == 'O' else f"{tag.split('-')[0]}-PII" for tag in seq]
                for seq in all_pred_bio_tags_nested
            ]
            
            # 단일 'PII' 클래스에 대한 F1 점수만 추출
            report = seq_classification_report(binary_true_nested, binary_pred_nested, output_dict=True, zero_division=0)
            binary_f1 = report.get('PII', {}).get('f1-score', 0.0)
            print(f"Entity-level Binary F1-Score (PII 전체): {binary_f1:.4f}")

            # # ✨ (참고) 기존의 클래스별 상세 리포트 ✨
            # print("\n--- Classification Report (per PII type) ---")
            # full_report = seq_classification_report(all_true_bio_tags_nested, all_pred_bio_tags_nested, zero_division=0, digits=4)
            # print(full_report)        
        
            print("\n\n--- 최종 성능 평가 (Token-level) ---")
            # 1. Token-level Micro F1 Score 계산
            micro_f1 = sk_f1_score(all_true_bio_tags, all_pred_bio_tags, average='micro', zero_division=0)
            print(f"Token-level Micro F1: {micro_f1:.4f}")

            # 2. Token-level Binary F1 Score 계산
            # 'O' 태그는 0, 나머지(B-*, I-*)는 1로 변환
            binary_true = ["O" if tag == 'O' else "PII" for tag in all_true_bio_tags]
            binary_pred = ["O" if tag == 'O' else "PII" for tag in all_pred_bio_tags]

            binary_f1 = sk_f1_score(binary_true, binary_pred, pos_label="PII", average='binary', zero_division=0)
            print(f"Token-level Binary F1: {binary_f1:.4f}")
            O_binary_f1 = sk_f1_score(binary_true, binary_pred, pos_label="O", average='binary', zero_division=0)
            print(f"O - Token-level Binary F1: {O_binary_f1:.4f}")

    elif args.mode == 'infer':
        print("--- 테스트 모드를 시작합니다 ---")

        # 1. 모델 로딩 (베이스 모델 + 학습된 LoRA 어댑터)
        print("베이스 모델 및 LoRA 어댑터 로딩 중...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # 저장된 LoRA 가중치를 베이스 모델에 병합
        model = PeftModel.from_pretrained(base_model, args.output_dir)
        model = model.merge_and_unload() # 추론 속도 향상을 위해 병합
        model.eval()

        # 2. 추론할 프롬프트 준비
        instruction = "주어진 문장에서 모든 개인 식별 정보(PII)를 찾아서, 각 PII의 종류, 시작 인덱스, 끝 인덱스를 JSON 형식으로 추출하세요."
        input_text = "담당자는 홍길동이며, 연락처는 010-1234-5678, 이메일 주소는 gildong.hong@example.com 입니다."
        
        prompt = f"""### 지시: {instruction}\n\n### 입력:\n{input_text}\n\n### 답변:\n"""
        
        # 3. 모델 추론 실행
        print("\n--- 추론 입력 ---")
        print(prompt)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id # pad_token_id 명시
            )
        
        # 생성된 결과에서 프롬프트를 제외한 답변 부분만 디코딩
        response_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        print("\n--- 추론 결과 ---")
        try:
            parsed_json = json.loads(response_text)
            print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
        except json.JSONDecodeError:
            print("JSON 파싱에 실패했습니다. 원본 출력:")
            print(response_text)