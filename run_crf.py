import argparse
import torch
import torch.nn as nn
from transformers import BertModel, BertForTokenClassification, AutoModelForCausalLM, AutoTokenizer, \
    Trainer, TrainingArguments,  StoppingCriteria, StoppingCriteriaList, DataCollatorForTokenClassification, RobertaForTokenClassification, PreTrainedTokenizerFast, AutoModelForTokenClassification, RobertaPreTrainedModel, RobertaModel
import custom
from datasets import load_dataset, Dataset, load_from_disk
import os
from seqeval.metrics import  precision_score, recall_score#, classification_report, f1_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.metrics import classification_report as sk_classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
from new_token_list import new_tokens
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
from matplotlib.colors import LogNorm, Normalize
from seqeval.metrics.sequence_labeling import get_entities # for overlapped, intermediated f1

USE_ADDITIONAL_TOKENS = False

def calculate_custom_f1(true_seqs, pred_seqs, mode="strict"):
    """
    mode options:
    - "strict": Exact boundary & type match.
    - "overlap": Any overlap & type match.
    - "intermediate": Overlap >= 50% of ground-truth length & type match.
    """
    tp_p = 0  # True Positives for Precision (Predicted entity 기준)
    total_p = 0 # Total Predicted Entities
    tp_r = 0  # True Positives for Recall (Ground Truth entity 기준)
    total_t = 0 # Total Ground Truth Entities

    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        true_ents = get_entities(true_seq) # [(tag, start, end), ...]
        pred_ents = get_entities(pred_seq)
        
        total_p += len(pred_ents)
        total_t += len(true_ents)

        # 1. Calculate Precision (Predicted Entity가 정답과 매칭되는지 확인)
        for p_ent in pred_ents:
            p_type, p_start, p_end = p_ent
            match_found = False
            for t_ent in true_ents:
                t_type, t_start, t_end = t_ent
                
                if p_type != t_type: # Type은 무조건 일치해야 함 (Note조건)
                    continue
                
                # 교집합 길이 계산
                intersect_start = max(p_start, t_start)
                intersect_end = min(p_end, t_end)
                intersection = max(0, intersect_end - intersect_start)
                
                if intersection == 0: 
                    continue
                
                # 모드별 조건 확인
                if mode == "strict":
                    if p_start == t_start and p_end == t_end:
                        match_found = True
                elif mode == "overlap":
                    match_found = True # 겹치기만 하면 OK
                elif mode == "intermediate":
                    # 정답 엔티티(ground-truth) 길이의 50% 이상 겹쳐야 함
                    gt_len = t_end - t_start
                    if intersection >= 0.5 * gt_len:
                        match_found = True
                
                if match_found: 
                    break
            if match_found: 
                tp_p += 1

        # 2. Calculate Recall (Ground Truth Entity가 예측되었는지 확인)
        for t_ent in true_ents:
            t_type, t_start, t_end = t_ent
            match_found = False
            for p_ent in pred_ents:
                p_type, p_start, p_end = p_ent
                
                if p_type != t_type: 
                    continue
                
                intersect_start = max(p_start, t_start)
                intersect_end = min(p_end, t_end)
                intersection = max(0, intersect_end - intersect_start)
                
                if intersection == 0: 
                    continue

                if mode == "strict":
                    if p_start == t_start and p_end == t_end:
                        match_found = True
                elif mode == "overlap":
                    match_found = True
                elif mode == "intermediate":
                    gt_len = t_end - t_start
                    if intersection >= 0.5 * gt_len:
                        match_found = True
                
                if match_found: 
                    break
            if match_found: 
                tp_r += 1
    
    precision = tp_p / total_p if total_p > 0 else 0.0
    recall = tp_r / total_t if total_t > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1

def calculate_robust_f1(true_seqs, pred_seqs, mode="strict"):
    """
    seqeval의 get_entities를 사용하여 정교하게 F1을 계산하는 함수
    
    Args:
        true_seqs: List[List[str]] - 정답 BIO 태그 리스트
        pred_seqs: List[List[str]] - 예측 BIO 태그 리스트
        mode: "strict", "overlap", "intermediate"
    """
    tp = 0
    fp = 0
    fn = 0

    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        # 1. seqeval의 파서를 사용하여 엔티티 추출 (Type, Start, End)
        # 리스트로 변환하여 인덱싱 가능하게 함
        true_ents = list(get_entities(true_seq))
        pred_ents = list(get_entities(pred_seq))
        
        # 중복 매칭 방지를 위한 집합 (이미 매칭된 정답 엔티티 인덱스 저장)
        hit_true_indices = set()

        # 2. Precision 계산 (예측된 엔티티가 정답에 있는지 확인)
        for p_ent in pred_ents:
            p_type, p_start, p_end = p_ent
            is_match = False
            
            for t_idx, t_ent in enumerate(true_ents):
                # 이미 다른 예측값과 매칭된 정답은 패스 (1:1 매칭 원칙)
                if t_idx in hit_true_indices:
                    continue
                
                t_type, t_start, t_end = t_ent
                
                # [공통 조건] 타입(Type)은 무조건 일치해야 함
                if p_type != t_type:
                    continue
                
                # [교집합 계산] (Python Slice Indexing 기준)
                intersect_start = max(p_start, t_start)
                intersect_end = min(p_end, t_end)
                intersection = max(0, intersect_end - intersect_start)
                
                # 아예 겹치지 않으면 패스
                if intersection == 0:
                    continue
                
                # [모드별 조건]
                match_condition = False
                if mode == "strict":
                    # 경계가 완전히 일치해야 함
                    if p_start == t_start and p_end == t_end:
                        match_condition = True
                        
                elif mode == "overlap":
                    # 1개 토큰이라도 겹치면 정답 (이미 intersection > 0은 위에서 확인)
                    match_condition = True
                    
                elif mode == "intermediate":
                    # 정답 엔티티 길이의 50% 이상 겹쳐야 함
                    gt_len = t_end - t_start
                    if intersection / gt_len >= 0.5:
                        match_condition = True
                
                if match_condition:
                    is_match = True
                    hit_true_indices.add(t_idx) # 정답 엔티티 소모(Consumed) 처리
                    break # 매칭 성공 시, 해당 예측에 대한 탐색 종료
            
            if is_match:
                tp += 1
            else:
                fp += 1 # 매칭되는 정답이 없으면 False Positive
        
        # 3. Recall 계산을 위한 FN (찾지 못한 정답 엔티티 수)
        # 전체 정답 수 - 찾은 정답 수(hit_true_indices)
        fn += len(true_ents) - len(hit_true_indices)

    # 4. 최종 F1 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1

def compute_advanced_metrics_crf(p):
    """
    반환값:
     {
       "f1": 다중 클래스 seqeval F1 (엔티티 단위),
       "binary_f1": PII vs O 토큰 단위 F1,
       - Cohen's Kappa (token-level agreement),
       - MUC-style Analysis (Correct, Spurious, Missing entities),
       "report": seqeval classification_report
     }
    """
    preds, labels = p
    
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

            true_tag = id2label.get(int(l_id), "O")
            pred_tag = id2label.get(int(p_id), "O")
            
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
            true_bin_tag = true_tag.replace(true_tag.split('-')[-1], 'PII') if true_tag != 'O' else 'O'
            pred_bin_tag = pred_tag.replace(pred_tag.split('-')[-1], 'PII') if pred_tag != 'O' else 'O'
            true_seq_binary_current.append(true_bin_tag)
            pred_seq_binary_current.append(pred_bin_tag)
            # true_seq_binary_current.append("O" if true_tag == "O" else "PII")
            # pred_seq_binary_current.append("O" if pred_tag == "O" else "PII")

        true_seqs.append(true_seq_current)
        pred_seqs.append(pred_seq_current)
        true_seqs_binary.append(true_seq_binary_current)
        pred_seqs_binary.append(pred_seq_binary_current)

    # 다중 클래스 F1 (엔티티 단위, seqeval)
    multiclass_micro_f1 = seq_f1_score(true_seqs, pred_seqs, average="micro")
    multiclass_weighted_f1 = seq_f1_score(true_seqs, pred_seqs, average="weighted")
    entity_level_binary_f1 = seq_f1_score(true_seqs_binary, pred_seqs_binary, average="micro", zero_division=0)
    report_str = seq_classification_report(true_seqs, pred_seqs, digits=4, zero_division=0)
    
    # 토큰 단위 메트릭 계산 (sklearn)
    binary_f1 = sk_f1_score(y_true_binary_flat, y_pred_binary_flat, average="weighted")
    # binary_f1 = sk_f1_score(y_true_binary_flat, y_pred_binary_flat, pos_label="PII", average="binary", zero_division=0)
    all_bio_labels = list(LABEL2ID.keys())
    all_merged_labels = []
    for label in all_bio_labels:
        if label != 'O' and label[2:] not in all_merged_labels:
            all_merged_labels.append(label[2:])
                
    token_merged_report_str = sk_classification_report(y_true_merged, y_pred_merged, labels=all_merged_labels, digits=4, zero_division=0)

    tokenlevel_micro_f1 = sk_f1_score(y_true_flat, y_pred_flat, average="micro", zero_division=0)
    kappa = cohen_kappa_score(y_true_flat, y_pred_flat)

    # --- 2. New Custom F1 Metrics ---
    # strict_f1 = calculate_custom_f1(true_seqs, pred_seqs, mode="strict")
    # overlap_f1 = calculate_custom_f1(true_seqs, pred_seqs, mode="overlap")
    # intermediate_f1 = calculate_custom_f1(true_seqs, pred_seqs, mode="intermediate")
    strict_f1 = calculate_robust_f1(all_true_bio, all_pred_bio, mode="strict")
    overlap_f1 = calculate_robust_f1(all_true_bio, all_pred_bio, mode="overlap")
    intermediate_f1 = calculate_robust_f1(all_true_bio, all_pred_bio, mode="intermediate")

    
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
            spurious = int(round(correct / precision - correct)) if precision > 0 else (support if recall == 0 else 0)
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
        "strict_f1": strict_f1,                       # Custom Strict
        "overlap_f1": overlap_f1,                     # Custom Overlap
        "intermediate_f1": intermediate_f1,           # Custom Intermediate
        "cohen_kappa": kappa,
        "muc_report": compact_muc_report,
        "entity_level_report": report_str,
        "token_level_report": token_merged_report_str
    }
    
try:
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    print("경고: NanumGothic 폰트를 찾을 수 없습니다. Confusion Matrix의 한글이 깨질 수 있습니다.")
    print("해결 방법: sudo apt-get install fonts-nanum*")
    
def plot_confusion_matrix(y_true, y_pred, labels, output_path="confusion_matrix.png"):
    # 'O' 태그를 제외한 PII 라벨만 필터링
    pii_labels = [label for label in labels if label != 'O']
    
    # PII 라벨에 대해서만 confusion matrix 계산
    cm = confusion_matrix(y_true, y_pred, labels=pii_labels)
    
    # pandas DataFrame으로 변환하여 인덱스와 컬럼 라벨 사용
    cm_df = pd.DataFrame(cm, index=pii_labels, columns=pii_labels)
    
    plt.figure(figsize=(16, 12))
    ax = plt.gca() # 현재 Axes 객체를 가져옵니다.

    # 1. 색상 맵 및 정규화 설정
    cmap = plt.get_cmap("Blues")
    
    vmax = cm.max()
    # 로그 스케일: 1 이상의 값에만 적용
    log_norm = LogNorm(vmin=1, vmax=vmax) if vmax > 0 else None
    
    # 선형 스케일: 모든 값에 대해 색상을 매핑 (0인 값의 배경색을 위해 사용)
    # vmin을 0으로 설정하여 0도 포함시키고, 0에 대한 색상은 수동으로 처리합니다.
    linear_norm = Normalize(vmin=0, vmax=vmax) # 0도 포함하는 선형 정규화
    
    # 2. 이미지(배경색) 그리기
    # imshow를 사용하면 norm이 0 이하의 값도 처리할 수 있도록 해야 합니다.
    # 그래서 0을 포함하는 선형 정규화(linear_norm)를 사용하여 데이터 전체의 색상 매핑을 결정합니다.
    im = ax.imshow(cm, cmap=cmap, norm=linear_norm, aspect='auto')
    
    # 3. 0인 셀의 배경색을 수동으로 변경
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                # 0인 셀에만 특정 색상을 칠합니다. (linear_norm이 0을 흰색으로 만들 경우)
                # 현재 cmap이 'Blues'이므로, 0은 가장 연한 파란색이 될 것입니다.
                # 'whitesmoke'를 원하면 아래처럼 명시적으로 칠해줍니다.
                ax.add_patch(plt.Rectangle(xy=(j-0.5, i-0.5), width=1, height=1, color='whitesmoke', lw=0))

    # 4. 숫자(Annotation) 그리기
    # 모든 셀에 대해 숫자를 직접 그립니다.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            # 글자색 결정 로직 (배경색 대비)
            # 여기서는 단순히 모두 검은색으로 고정합니다.
            color = "black" 
            
            ax.text(j, i, str(val),
                    ha='center', va='center', color=color, fontsize=8) # 글자 크기 조절 가능

    # 5. 컬러바 추가 (로그 스케일)
    # 컬러바는 1 이상의 값에 대한 로그 스케일을 보여주는 것이 더 적합합니다.
    # 따라서 로그 정규화를 사용하여 컬러바를 만듭니다.
    # LogNorm은 0을 직접 다루지 못하므로, 컬러바 데이터는 1 이상의 값만 고려합니다.
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=log_norm, cmap=cmap), ax=ax, extend='min')
    cbar.ax.set_ylabel('Count (Log Scale)', rotation=-90, va="bottom")
    
    # 6. 축 라벨 설정
    ax.set_xticks(np.arange(len(pii_labels)))
    ax.set_yticks(np.arange(len(pii_labels)))
    ax.set_xticklabels(pii_labels, rotation=45, ha='right')
    ax.set_yticklabels(pii_labels, rotation=0)
    
    ax.set_title('Confusion Matrix (PII Classes Only)')
    ax.set_ylabel('True Labels')
    ax.set_xlabel('Predicted Labels')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Confusion Matrix가 '{output_path}'에 저장되었습니다.")

class RobertaCrfForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # 1. Roberta 모델 통과 및 Logits 계산 (기존과 동일)
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            mask = attention_mask.bool() if attention_mask is not None else None
            # active_labels = labels.clone()
            # valid_max_label_id = self.num_labels - 1
            # active_labels[active_labels == -100] = 22
            safe_labels = labels.clone().masked_fill(labels == -100, 22)
            # assert safe_labels.shape == mask.shape == logits.shape[:2]
            # print(safe_labels.min(), safe_labels.max(), "mask sum:", mask.sum())
            # active_labels = torch.clamp(active_labels, min=0, max=valid_max_label_id)
            # print(f"Shape - Logits: {logits.shape}, Labels: {active_labels.shape}, Mask: {mask.shape}")
            # print(f"Device - Logits: {logits.device}, Labels: {active_labels.device}, Mask: {mask.device}")
            # print(f"Label Range - Min: {active_labels.min().item()}, Max: {active_labels.max().item()}")
            # print(f"Config num_labels: {self.num_labels}")
            
            # with torch.no_grad():
            #     logits64 = logits.double().cpu()
            #     labels64 = safe_labels.cpu()
            #     mask64   = mask.cpu()
            #     self.crf.cpu()
            #     loss64   = -self.crf(logits64, labels64, mask=mask64, reduction='mean')
            #     self.crf.cuda()
            #     print("CPU FP64 CRF loss:", loss64)
            loss = -self.crf(logits, safe_labels, mask=mask, reduction='token_mean')
            # loss = -self.crf(logits, safe_labels, mask=mask, reduction='sum')
            # loss = loss / mask.sum()
            if torch.isnan(loss):
                print(f'loss is {loss}')
                ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, self.num_labels), labels.view(-1))
                loss = 0.5*loss.nan_to_num(0.0) + 0.5*ce_loss
            else:
                loss = loss

        # if not self.training:
        #     mask = attention_mask.bool() if attention_mask is not None else None
        #     decoded_paths = self.crf.decode(logits, mask=mask)
            
        #     max_len = logits.size(1)
        #     padded_paths = [p + [-100] * (max_len - len(p)) for p in decoded_paths]
        #     predictions = torch.tensor(padded_paths, device=logits.device)
        #     return TokenClassifierOutput(loss=loss, logits=predictions)

        # 4. 훈련 모드일 때 결과 반환
        return TokenClassifierOutput(loss=loss, logits=logits)
    
class CRFTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # Forward로 emissions 얻기
            outputs = model(**inputs)
            loss = outputs.loss
            emissions = outputs.logits  # float tensor
            
            # CRF decode 수행
            mask = inputs.get('attention_mask', None)
            if mask is not None:
                mask = mask.bool()
                try:
                    decoded_paths = model.crf.decode(emissions, mask=mask)
                    # 패딩처리
                    max_len = emissions.size(1)
                    padded_paths = []
                    for path in decoded_paths:
                        padded_path = path + [-100] * (max_len - len(path))
                        padded_paths.append(padded_path)
                    predictions = torch.tensor(padded_paths, device=emissions.device)
                except:
                    # CRF decode 실패 시 argmax 사용
                    predictions = emissions.argmax(dim=-1)
            else:
                predictions = emissions.argmax(dim=-1)
            
            labels = inputs.get('labels')
            
        return (loss, predictions, labels) if not prediction_loss_only else loss

model_checkpoint = "klue/roberta-base"

train_path = './datasets/pii_ner_merged_08061_v2/train'
eval_path = './datasets/pii_ner_merged_08061_v2/validation'
test_path = './datasets/pii_ner_merged_08061_v2/test'
train_path = './datasets/pii_ner_3merged_08061_v3/train'
eval_path = './datasets/pii_ner_3merged_08061_v3/validation'
test_path = './datasets/pii_ner_3merged_08061_v3/test'
# test_path = './datasets/pii_ner_half/test' 
# load dataset
train_dataset = load_from_disk(train_path)
eval_dataset = load_from_disk(eval_path)
test_dataset = load_from_disk(test_path)


LABEL2ID = {
    "B-이름": 0, "I-이름": 1,
    "B-학교": 2, "I-학교": 3,      # '학교'로 통합 ('학과' 없음)
    "B-회사": 4, "I-회사": 5,      # '회사'로 통합 ('업무부서' 없음)
    "B-주소": 6, "I-주소": 7,
    "B-번호": 8, "I-번호": 9,
    "B-URL": 10, "I-URL": 11,     # 'URL'로 통합 ('웹주소' 없음)
    "B-계좌번호": 12, "I-계좌번호": 13,
    "B-은행명": 14, "I-은행명": 15,
    "B-보안코드": 16, "I-보안코드": 17,
    "B-이메일": 18, "I-이메일": 19,
    "B-아이디": 20, "I-아이디": 21,
    "O": 22
}
id2label = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

class FilteredDataset:
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                sample = self.dataset[idx]
                return {
                    'input_ids': sample['input_ids'],
                    'attention_mask': sample['attention_mask'], 
                    'labels': sample['labels']
                }

def find_latest_checkpoint_folder(directory):
    prefix = "checkpoint-"
    max_num = -1
    latest_path = None

    if not os.path.exists(directory):
        return None

    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)

        if not os.path.isdir(full_path):
            continue

        if name.startswith(prefix):
            num_part = name[len(prefix):]
            
            if num_part.isdigit():
                num = int(num_part)
                
                if num > max_num:
                    max_num = num
                    latest_path = full_path

    return latest_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train/test/infer")
    parser.add_argument("--b", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="./results/PII_klueBERT")
    parser.add_argument("--dataset_path", type=str, default="./datasets")
    # parser.add_argument("--ckpt_path", type=str, default="/mnt/data3/Korean_abstraction/python/coreference/results/PII_klueBERT/backup_b32_lr3e5_v3_kluebert_crf/checkpoint-54710")
    parser.add_argument("--ckpt_path", type=str, default="/mnt/data3/Korean_abstraction/python/coreference/results/PII_klueBERT/crf_1024_1e5_b16/checkpoint-109420")
    parser.add_argument("--fig_path", type=str, default="3datasets")
    parser.add_argument("--add_token", type=bool, default=False)
   
    
    args = parser.parse_args()
    
    if args.dataset_path == "./datasets":
        train_path = './datasets/pii_ner_3merged_08061_v3/train'
        eval_path = './datasets/pii_ner_3merged_08061_v3/validation'
        test_path = './datasets/pii_ner_3merged_08061_v3/test'

        train_dataset = load_from_disk(train_path)
        eval_dataset = load_from_disk(eval_path)
        test_dataset = load_from_disk(test_path)
    else:
        train_path = args.dataset_path + '/train'
        eval_path = args.dataset_path + '/validation'
        test_path = args.dataset_path + '/test'

        train_dataset = load_from_disk(train_path)
        eval_dataset = load_from_disk(eval_path)
        test_dataset = load_from_disk(test_path)
        
    # train_dataset = Subset(train_dataset, list(range(0, 100)))
    # eval_dataset = Subset(eval_dataset, list(range(0, 50)))
    from transformers import AutoConfig
    max_len = 512
    if args.mode == "train":
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        max_len = 512 # max(len(x['input_ids']) for x in train_dataset)
        # print(max_len)
        data_collator = DataCollatorForTokenClassification(tokenizer, padding='max_length',  max_length=max_len, label_pad_token_id=-100)
        

        config = AutoConfig.from_pretrained(model_checkpoint, num_labels=NUM_LABELS, id2label=id2label, label2id=LABEL2ID)
        # full_pipeline_test()

        model = RobertaCrfForTokenClassification.from_pretrained(
            model_checkpoint,
            config=config
        ).to("cuda")
        
        with torch.no_grad():
            model.crf.transitions.clamp_(-1.0, 1.0)
            model.crf.start_transitions.clamp_(-1.0, 1.0)
            model.crf.end_transitions.clamp_(-1.0, 1.0)

        if args.add_token == True:
            tokenizer.add_special_tokens({"additional_special_tokens" : new_tokens})
            num_added_toks = tokenizer.add_tokens(new_tokens)
            print(f"새로운 토큰 {num_added_toks}개를 어휘 사전에 추가했습니다.")
        model.resize_token_embeddings(len(tokenizer))
        total_params = sum(p.numel() for p in model.parameters())
        
        # print(f"총 파라미터 수: {total_params}"); quit()
        training_args = TrainingArguments(
            output_dir=args.save_dir,
            num_train_epochs=args.epoch,
            per_device_train_batch_size=args.b,
            per_device_eval_batch_size=args.b,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            # warmup_steps=300,
            # weight_decay=0.01,
            warmup_steps=500,
            weight_decay=0.02,
            learning_rate = args.lr,
            remove_unused_columns=True, # custom dataset사용시 true
            save_steps=10_000,
            save_total_limit=2,
            seed = 43,
            load_best_model_at_end=True,
            # bf16 = True,
            max_grad_norm=0.5, 
            metric_for_best_model="entity_level_micro_f1",
        )
        
        # sub_trainset = FilteredDataset(Subset(train_dataset, list(range(0, 100))))
        # sub_testset = FilteredDataset(Subset(test_dataset, list(range(0, 50))))
        sub_trainset = FilteredDataset(train_dataset)
        sub_evalset = FilteredDataset(eval_dataset)
        
        trainer = CRFTrainer(
            model=model,
            args=training_args,
            train_dataset=sub_trainset,
            eval_dataset=sub_evalset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_advanced_metrics_crf,
        )
        trainer.train()
        trainer.evaluate()
        
    elif args.mode == "test":
        torch.cuda.empty_cache()
        # ckpt_path = "/mnt/data3/Korean_abstraction/python/coreference/results/PII_klueBERT/backup_b32_lr3e5_v3_kluebert_crf/checkpoint-54710"

        # args.ckpt_path = find_latest_checkpoint_folder(args.ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=True)
        

        
        max_len = 512 # max(len(x['input_ids']) for x in train_dataset)
        data_collator = DataCollatorForTokenClassification(tokenizer, padding='max_length',  max_length=max_len, label_pad_token_id=-100)

        model = RobertaCrfForTokenClassification.from_pretrained(
            args.ckpt_path
        ).to("cuda")

        # 3) Trainer 생성 후 평가
        test_args = TrainingArguments(
            output_dir=args.save_dir + '/test',
            per_device_eval_batch_size=1,
            bf16=True,
            dataloader_drop_last=False,
            report_to="none",
        )

        trainer = CRFTrainer(
            model=model,
            args=test_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_advanced_metrics_crf,
        )
        prediction_result = trainer.predict(test_dataset)
        metrics = prediction_result.metrics
        print("-------- Test Metrics --------")
        print(f"Entity-level_Micro F1 (23-class) : {metrics['test_entity_level_micro_f1']:.4f}")
        print(f"Entity-level_Weighted F1 (23-class) : {metrics['test_entity_level_weighted_f1']:.4f}")
        print(f"Entity_level_binary_f1 (PII/O) : {metrics['test_entity_level_binary_f1']:.4f}")
        print(f"Token-level Micro F1 (23-class) : {metrics['test_token_level_micro_f1']:.4f}")
        print(f"Token-level Binary  F1 (PII/O)  : {metrics['test_token_level_binary_f1']:.4f}")
        print(f"Strict F1 : {metrics['test_strict_f1']:.4f}")
        print(f"overlap F1 : {metrics['test_overlap_f1']:.4f}")
        print(f"intermediate F1 : {metrics['test_intermediate_f1']:.4f}")
        print(f"Cohen keppa : {metrics['test_cohen_kappa']:.4f}")
        print("\nMUC report:\n")
        print(metrics["test_muc_report"])
        print("\n--- Detailed Report (Entity-level) ---\n")
        print(metrics["test_entity_level_report"])
        print("\n--- Detailed Report (Token-level, Merged) ---\n")
        print(metrics["test_token_level_report"])
        
        predictions = prediction_result.predictions
        if predictions.ndim == 3:
            preds = np.argmax(predictions, axis=2)
        else:
            preds = predictions
        labels = prediction_result.label_ids
        
        y_true_bio, y_pred_bio = [], []
        y_true_merged, y_pred_merged = [], []

        for pred_row, label_row in zip(preds, labels):
            for p_id, l_id in zip(pred_row, label_row):
                if l_id == -100:
                    continue
                
                true_tag = id2label.get(int(l_id), "O")
                pred_tag = id2label.get(int(p_id), "O")


                y_true_bio.append(true_tag)
                y_pred_bio.append(pred_tag)
                
                # 리스트 2: B-/I- 태그 통합
                # 'O'가 아닐 경우, 'B-' 또는 'I-' (2글자)를 제거
                y_true_merged.append(true_tag[2:] if true_tag != 'O' else 'O')
                y_pred_merged.append(pred_tag[2:] if pred_tag != 'O' else 'O')

        # 2. 두 가지 버전의 Confusion Matrix를 각각 생성
        all_bio_labels = list(LABEL2ID.keys())
        all_merged_labels = []
        for label in all_bio_labels:
            if label != 'O' and label[2:] not in all_merged_labels:
                all_merged_labels.append(label[2:])
        # all_merged_labels = sorted(list(set([label[2:] if label != 'O' else 'O' for label in all_bio_labels])))


        # dataset_name = args.dataset_path.split('/')[-1].split('_')[-2]
        # # 방법 1: B-/I- 태그를 유지
        # plot_confusion_matrix(y_true_bio, y_pred_bio, all_bio_labels, output_path=f"./Confusion_matrics/{args.fig_path}/bio/KluebertCRF_{dataset_name}-dataset_cm_bio.png")
        import json
        print("Token-level 추론 결과를 'bio_tag_results.jsonl' 파일로 저장합니다...")
        with open("result_for_confusion_matrix.jsonl", "w", encoding="utf-8") as f:
            line = json.dumps({"y_true_merged": y_true_merged, "y_pred_merged": y_pred_merged}, ensure_ascii=False)
            # for true_tag, pred_tag in zip(y_true_merged, y_pred_merged):
            #     line = json.dumps({"y_true_merged": true_tag, "y_pred_merged": pred_tag}, ensure_ascii=False)
            f.write(line + "\n")
        print("저장 완료.")
        import seaborn as sns

        cnf_matrix = confusion_matrix(y_true_merged, y_pred_merged, labels=["이름", "학교", "회사", "주소", "번호", "URL", "계좌번호", "은행명", "보안코드", "이메일", "아이디"])
        sns.heatmap(cnf_matrix, annot=True, cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig("confusion_matrix.png")
        # 방법 2: B-/I- 태그를 통합
        # plot_confusion_matrix(y_true_merged, y_pred_merged, all_merged_labels, output_path=f"./Confusion_matrics/{args.fig_path}/merged/KluebertCRF_{dataset_name}-dataset_cm_merged.png")


        # ## data test start
        # sub_dataset = Subset(test_dataset, list(range(0, 20)))
        # prediction_result = trainer.predict(sub_dataset)
        # predictions = prediction_result.predictions
        # if predictions.ndim == 3:
        #     preds = np.argmax(predictions, axis=2)
        # else:
        #     preds = predictions

        # # 2. 분석할 샘플 개수 설정
        # num_samples_to_analyze = 20
        # # ckpt_path = "/mnt/data3/Korean_abstraction/python/coreference/results/PII_klueBERT/backup_b32_lr3e5_v3_kluebert_crf/checkpoint-54710"
        # tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, use_fast=True)
        # for i in range(num_samples_to_analyze):
        #     # sub_dataset에서 샘플 하나씩 가져오기
        #     sample = sub_dataset[i]
        #     input_ids = sample['input_ids']
        #     true_label_ids = sample['labels']
        #     pred_ids_for_sample = preds[i]

        #     # ID를 토큰과 레이블 텍스트로 변환
        #     tokens = tokenizer.convert_ids_to_tokens(input_ids)

        #     print(f"\n-----------[ Sample #{i+1} ]-----------")
        #     # 원본 문장 출력 (특수 토큰 제외)
        #     original_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        #     print(f"  Original Text: {original_text}\n")
            
        #     # 테이블 헤더 출력
        #     print(f"  {'Token':<15} | {'True Label':<20} | {'Predicted Label':<20}")
        #     print("  " + "-"*65)

        #     # 각 토큰별로 정답과 예측 비교
        #     for token, true_id, pred_id in zip(tokens, true_label_ids, pred_ids_for_sample):
        #         # 패딩(-100)된 레이블은 건너뛰기
        #         if true_id != -100:
        #             # ID를 실제 레이블 이름으로 변환
        #             # (주의: 스크립트에 정의된 대로 정답과 예측의 id2label 맵을 다르게 사용)
        #             true_label = id2label.get(true_id, "N/A")
        #             pred_label = id2label.get(pred_id, "N/A")

        #             # 예측이 틀렸을 경우 하이라이트
        #             mismatch_highlight = " <--- ★" if true_label != pred_label else ""
                    
        #             # 토큰의 특수문자( ) 제거 후 출력
        #             display_token = token.replace(' ', '')
                    
        #             print(f"  {display_token:<15} | {true_label:<20} | {pred_label:<20}{mismatch_highlight}")

      