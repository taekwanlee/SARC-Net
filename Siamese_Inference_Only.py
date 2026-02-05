import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_v2_s
from PIL import Image
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from sklearn.preprocessing import StandardScaler
import joblib  # scaler 저장/로드용
import seaborn as sns
from scipy.stats import pearsonr
from decimal import Decimal, getcontext
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
os.environ["OMP_NUM_THREADS"] = "4"

from torch.amp import autocast

# Optional: C-index 계산을 위한 import (없으면 무시)
try:
    from sksurv.metrics import concordance_index_censored
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    print("[WARNING] scikit-survival이 설치되지 않았습니다.")


# -------------------------------------------------------------
# Args
# -------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Siamese Age Regression - Inference Only")
    
    # 필수 인자
    parser.add_argument("--model_path", type=str, required=True,
                        help="학습된 모델 가중치 파일 경로 (.pth)")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="추론할 데이터의 CSV 파일 경로")
    
    # CSV 컬럼명 설정
    parser.add_argument("--image_path_col", type=str, default="file_path",
                        help="이미지 경로 컬럼명 (기본값: file_path)")
    parser.add_argument("--label_col", type=str, default="chronoage",
                        help="나이 레이블 컬럼명 (기본값: chronoage)")
    
    # 선택적 인자
    parser.add_argument("--output_csv", type=str, default="predictions.csv",
                        help="예측 결과를 저장할 CSV 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./siamese_inference_results",
                        help="결과 이미지 저장 디렉토리")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="배치 크기")
    parser.add_argument("--img_size", type=int, default=512,
                        help="입력 이미지 크기")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="사용할 디바이스 (cuda:0, cuda:1, cpu 등)")
    
    # Scaler 
    parser.add_argument("--scaler_path", type=str, default=None,
                        help="저장된 StandardScaler 파일 경로 (.pkl 또는 .joblib) - 가장 권장되는 방법")
    
    # 모델 관련
    parser.add_argument("--backbone_weights", type=str, default=None,
                        help="EfficientNetV2-S 사전학습 가중치 경로 (None이면 사용 안함)")
    
    # 평가 관련
    parser.add_argument("--has_labels", action="store_true",
                        help="CSV에 chronoage 레이블이 있는 경우 (평가 수행)")
    parser.add_argument("--plot_results", action="store_true",
                        help="Delta Scatter Plot 생성 여부 (동일 환자 내 여러 이미지가 있는 경우만 분석한 plot)")
    
    # Delta 분석용 (동일 환자 내 여러 이미지가 있는 경우)
    parser.add_argument("--patient_id_col", type=str, default="patient_id",
                        help="환자 ID 컬럼명 (Delta 분석용)")
    
    # C-index 계산용 (survival 데이터)
    parser.add_argument("--survival_time_col", type=str, default="survival.time",
                        help="생존 시간 컬럼명 (기본값: survival.time)")
    parser.add_argument("--death_col", type=str, default="death",
                        help="사망 이벤트 컬럼명 (기본값: death)")
    
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------
# Attention pooling module (기존 코드와 동일)
# -------------------------------------------------------------
class AttentionPool2D(nn.Module):
    def __init__(self, in_ch: int, alpha_roi: float = 1.0):
        super().__init__()
        self.logit_conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.alpha_roi = alpha_roi
        self.eps = 1e-8
        self.last_attention_map = None

    def forward(self, feat: torch.Tensor, roi_img: torch.Tensor | None = None):
        B, C, H, W = feat.shape
        logits = self.logit_conv(feat)                           # (B,1,H,W)
        log_attn = F.log_softmax(logits.flatten(2), dim=2)       # (B,1,H*W)
        attn = log_attn.exp()
        self.last_attention_map = attn.view(B, 1, H, W)
        hidden = (feat.flatten(2) * attn).sum(-1)                # (B,C)
        attn_loss = torch.zeros((), device=feat.device, dtype=feat.dtype)
        return hidden, attn_loss


# -------------------------------------------------------------
# Model (기존 코드와 동일)
# -------------------------------------------------------------
class SiameseAgeModel(nn.Module):
    def __init__(self, alpha_roi: float = 0.0, fuse_mode: str = 'concat', 
                 backbone_weights_path: str = None):
        super().__init__()
        self.backbone = efficientnet_v2_s(weights=None)
        
        # 사전학습 가중치 로드 (선택적)
        if backbone_weights_path and os.path.exists(backbone_weights_path):
            state_dict = torch.load(backbone_weights_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)
            print(f"[INFO] Backbone weights loaded from: {backbone_weights_path}")
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.backbone_features = self.backbone.features

        self.fuse_mode = 'concat'
        self.alpha_roi = alpha_roi
        self.attn = AttentionPool2D(in_features, alpha_roi)
        self.last_attn_loss = torch.tensor(0.0)
        self.gap = nn.AdaptiveAvgPool2d(1)

        fuse_c = in_features * 2 if self.fuse_mode == "concat" else in_features
        self.age_head = nn.Linear(fuse_c, 1)

    def _forward_one(self, img: torch.Tensor, roi_img: torch.Tensor | None = None):
        feat = self.backbone_features(img)          # (B, C, H, W)
        gap_vec = self.gap(feat).flatten(1)
        att_vec, attn_loss = self.attn(feat, roi_img)
        if self.fuse_mode == 'concat':
            z = torch.cat([gap_vec, att_vec], dim=1)
        else:
            z = 0.5 * (gap_vec + att_vec)
        y = self.age_head(z)
        return y, attn_loss

    def forward(self, x1, x2=None, roi1=None, roi2=None):
        y1, attn1 = self._forward_one(x1, roi1)
        if x2 is None:
            self.last_attn_loss = attn1
            return y1
        y2, attn2 = self._forward_one(x2, roi2)
        self.last_attn_loss = 0.5 * (attn1 + attn2)
        return y1, y2


# -------------------------------------------------------------
# Dataset for Inference (기존 SingleCXRDataset 기반, 단순화)
# -------------------------------------------------------------
class InferenceDataset(Dataset):
    """
    Inference용 Dataset.
    CSV 파일에서 file_path를 읽어 이미지를 로드합니다.
    chronoage가 있으면 레이블로 사용, 없으면 None 반환.
    """
    def __init__(self, df: pd.DataFrame, transforms=None, has_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.has_labels = has_labels and ('chronoage' in df.columns)
        
        # 필수 컬럼 체크
        if 'file_path' not in df.columns:
            raise ValueError("CSV에 'file_path' 컬럼이 필요합니다.")

    def _read_np(self, path):
        return np.array(Image.open(path).convert('RGB'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['file_path'].iloc[idx]
        img = self._read_np(img_path)
        
        if self.transforms:
            out = self.transforms(image=img)
            img_t = out["image"]
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if self.has_labels:
            label = torch.tensor(self.df['chronoage'].iloc[idx], dtype=torch.float32)
            return img_t, label
        else:
            return img_t


# -------------------------------------------------------------
# Transforms (기존 코드와 동일)
# -------------------------------------------------------------
def build_inference_transform(size: int):
    """추론용 Transform (augmentation 없음)"""
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])


# -------------------------------------------------------------
# Metrics
# -------------------------------------------------------------
def bootstrap_mae_ci(y_true, y_pred, n_bootstrap=2000, alpha=0.05, random_seed=42):
    """Bootstrap confidence interval for MAE"""
    np.random.seed(random_seed)
    n_samples = len(y_true)
    mae_values = []
    mae = mean_absolute_error(y_true, y_pred)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        mae_bs = mean_absolute_error(y_true[idx], y_pred[idx])
        mae_values.append(mae_bs)
    mae_values = np.array(mae_values)
    lower = np.percentile(mae_values, 100 * alpha / 2)
    upper = np.percentile(mae_values, 100 * (1 - alpha / 2))
    return mae, (lower, upper)


def bootstrap_rmse_ci(y_true, y_pred, n_bootstrap=2000, alpha=0.05, random_seed=42):
    """Bootstrap confidence interval for RMSE"""
    np.random.seed(random_seed)
    n_samples = len(y_true)
    rmse_values = []
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        rmse_bs = math.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))
        rmse_values.append(rmse_bs)
    rmse_values = np.array(rmse_values)
    lower = np.percentile(rmse_values, 100 * alpha / 2)
    upper = np.percentile(rmse_values, 100 * (1 - alpha / 2))
    return rmse, (lower, upper)


def bootstrap_cindex_ci(durations, risks, events, n_bootstrap=2000, alpha=0.05, random_seed=42):
    if not HAS_SKSURV:
        return None, (None, None)
    np.random.seed(random_seed)
    n_samples = len(durations)
    c_indices = []
    c_index = concordance_index_censored(events.astype(bool), durations, risks)[0]
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        c_index_bs = concordance_index_censored(events[idx].astype(bool), durations[idx], risks[idx])[0]
        c_indices.append(c_index_bs)
    c_indices = np.array(c_indices)
    lower = np.percentile(c_indices, 100 * alpha / 2)
    upper = np.percentile(c_indices, 100 * (1 - alpha / 2))
    return c_index, (lower, upper)


# -------------------------------------------------------------
# Main Inference Function
# -------------------------------------------------------------
def run_inference(args):
    set_seed()
    
    # Device 설정
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("[WARNING] CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transform
    transform = build_inference_transform(args.img_size)
    
    # 데이터 로드
    print(f"[INFO] Loading data from: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    print(f"[INFO] Total samples: {len(df)}")
    
    # 컬럼명 검증 및 내부 표준 컬럼명으로 변환
    image_col = args.image_path_col
    label_col = args.label_col
    
    # 이미지 경로 컬럼 확인
    if image_col not in df.columns:
        raise ValueError(f"[ERROR] 이미지 경로 컬럼 '{image_col}'이 CSV에 없습니다.\n"
                        f"        사용 가능한 컬럼: {list(df.columns)}\n"
                        f"        --image_path_col 인자로 올바른 컬럼명을 지정하세요.")
    
    # 내부 표준 컬럼명으로 변환 (file_path)
    if image_col != 'file_path':
        df['file_path'] = df[image_col]
        print(f"[INFO] 이미지 경로 컬럼: '{image_col}' -> 'file_path'로 매핑")
    
    # 레이블 컬럼 확인 (has_labels가 True인 경우에만)
    if args.has_labels:
        if label_col not in df.columns:
            raise ValueError(f"[ERROR] 레이블 컬럼 '{label_col}'이 CSV에 없습니다.\n"
                            f"        사용 가능한 컬럼: {list(df.columns)}\n"
                            f"        --label_col 인자로 올바른 컬럼명을 지정하거나, --has_labels를 제거하세요.")
        # 내부 표준 컬럼명으로 변환 (chronoage)
        if label_col != 'chronoage':
            df['chronoage'] = df[label_col]
            print(f"[INFO] 레이블 컬럼: '{label_col}' -> 'chronoage'로 매핑")
    
    # Scaler 설정 (우선순위: scaler_path > scaler_csv > scaler_mean/std)
    if args.scaler_path:
        # 방법 1: 저장된 scaler 파일 로드 (가장 권장)
        print(f"[INFO] Loading scaler from: {args.scaler_path}")
        if not os.path.exists(args.scaler_path):
            raise FileNotFoundError(f"Scaler 파일을 찾을 수 없습니다: {args.scaler_path}")
        scaler = joblib.load(args.scaler_path)
        print(f"[INFO] Scaler loaded - mean: {scaler.mean_[0]:.4f}, std: {scaler.scale_[0]:.4f}")
    elif args.scaler_csv:
        # 방법 2: 학습 데이터 CSV로 scaler fit
        print(f"[INFO] Fitting scaler from: {args.scaler_csv}")
        train_df = pd.read_csv(args.scaler_csv)
        scaler = StandardScaler()
        scaler.fit(train_df[['chronoage']])
        print(f"[INFO] Scaler fitted - mean: {scaler.mean_[0]:.4f}, std: {scaler.scale_[0]:.4f}")
    elif args.scaler_mean is not None and args.scaler_std is not None:
        # 방법 3: 수동으로 mean/std 설정
        print(f"[INFO] Using provided scaler parameters: mean={args.scaler_mean}, std={args.scaler_std}")
        scaler = StandardScaler()
        scaler.mean_ = np.array([args.scaler_mean])
        scaler.scale_ = np.array([args.scaler_std])
        scaler.var_ = np.array([args.scaler_std ** 2])
        scaler.n_features_in_ = 1
    else:
        print("[WARNING] Scaler 정보가 제공되지 않았습니다. 원본 스케일로 출력합니다.")
        print("[WARNING] 다음 중 하나를 사용하세요:")
        print("          --scaler_path <path_to_scaler.pkl>  (권장)")
        print("          --scaler_csv <path_to_train_data.csv>")
        print("          --scaler_mean <mean> --scaler_std <std>")
        scaler = None
    
    # Dataset & DataLoader
    dataset = InferenceDataset(df, transforms=transform, has_labels=args.has_labels)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 모델 로드
    print(f"[INFO] Loading model from: {args.model_path}")
    model = SiameseAgeModel(
        alpha_roi=0,  # Inference 시 ROI loss 비활성화
        fuse_mode='concat',
        backbone_weights_path=None  # 학습된 가중치에서 로드하므로 불필요
    ).to(device)
    
    # 가중치 로드
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Model loaded successfully!")
    
    # Inference
    print("[INFO] Running inference...")
    all_preds = []
    all_labels = [] if args.has_labels else None
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            if args.has_labels:
                images, labels = batch
                all_labels.append(labels.numpy())
            else:
                images = batch
            
            images = images.to(device, non_blocking=True)
            
            with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
    
    # 결과 병합
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    
    # 스케일 역변환
    if scaler is not None:
        preds_real = scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    else:
        preds_real = all_preds
    
    # 결과 DataFrame 생성
    result_df = df.copy()
    result_df['predicted_age'] = preds_real
    if scaler is not None:
        result_df['predicted_age_normalized'] = all_preds
    
    # 결과 저장
    output_path = os.path.join(args.output_dir, args.output_csv)
    result_df.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to: {output_path}")
    
    # 평가 (레이블이 있는 경우)
    if args.has_labels and all_labels is not None:
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        
        if scaler is not None:
            labels_real = all_labels  # 원본 스케일
        else:
            labels_real = all_labels
        
        # 메트릭 계산
        r2 = r2_score(labels_real, preds_real)
        mae, (mae_lo, mae_hi) = bootstrap_mae_ci(labels_real, preds_real)
        rmse, (rmse_lo, rmse_hi) = bootstrap_rmse_ci(labels_real, preds_real)
        
        print("\n" + "="*60)
        print("🚩 Evaluation Results 🚩")
        print("="*60)
        print(f"R² Score: {r2:.5f}")
        print(f"MAE: {mae:.5f} years (95% CI: {mae_lo:.5f}-{mae_hi:.5f})")
        print(f"RMSE: {rmse:.5f} years (95% CI: {rmse_lo:.5f}-{rmse_hi:.5f})")
        
        # Pearson correlation (p-value만 출력)
        r_val, p_val = pearsonr(labels_real, preds_real)
        if p_val < 0.01:
            print("p < 0.01")
        elif p_val < 0.05:
            print("p < 0.05")
        else:
            print(f"p = {p_val:.4f}")
        
        # C-index (survival 데이터가 있는 경우)
        survival_col = args.survival_time_col
        death_col = args.death_col
        if HAS_SKSURV and {survival_col, death_col}.issubset(df.columns):
            durations = df[survival_col].to_numpy(dtype=float)
            events = df[death_col].to_numpy()
            c_index, (ci_lo, ci_hi) = bootstrap_cindex_ci(durations, preds_real, events)
            if c_index is not None:
                print(f"C-index: {c_index:.4f} (95% CI: {ci_lo:.4f}-{ci_hi:.4f})")
        elif HAS_SKSURV:
            print(f"[INFO] C-index 계산 스킵: '{survival_col}' 또는 '{death_col}' 컬럼이 CSV에 없습니다.")
        print("="*60)
        
        # Delta Scatter Plot 생성 (동일 환자 내 여러 이미지가 있는 경우)
        if args.plot_results:
            patient_col = args.patient_id_col
            
            if patient_col not in result_df.columns:
                print(f"[WARNING] '{patient_col}' 컬럼이 없어서 Delta Scatter Plot을 생성할 수 없습니다.")
                print(f"[INFO] --patient_id_col 인자로 환자 ID 컬럼명을 지정하세요.")
            else:
                # 환자별로 이미지 쌍 생성
                delta_pairs = []
                
                for patient_id, group in result_df.groupby(patient_col):
                    if len(group) < 2:
                        continue  # 이미지가 2개 미만이면 쌍을 만들 수 없음
                    
                    group = group.sort_values('chronoage').reset_index(drop=True)
                    
                    # 모든 쌍 조합 생성
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            gt_delta = group.loc[j, 'chronoage'] - group.loc[i, 'chronoage']
                            pred_delta = group.loc[j, 'predicted_age'] - group.loc[i, 'predicted_age']
                            
                            delta_pairs.append({
                                'patient_id': patient_id,
                                'gt_delta': gt_delta,
                                'pred_delta': pred_delta
                            })
                
                if len(delta_pairs) == 0:
                    print("[WARNING] 동일 환자 내 여러 이미지를 가진 환자가 없어서 Delta Scatter Plot을 생성할 수 없습니다.")
                else:
                    delta_df = pd.DataFrame(delta_pairs)
                    
                    # 통계 계산
                    n_pairs = len(delta_df)
                    n_patients = delta_df['patient_id'].nunique()
                    gt_delta = delta_df['gt_delta'].values
                    pred_delta = delta_df['pred_delta'].values
                    
                    # Pearson 상관계수
                    corr, p_val = pearsonr(gt_delta, pred_delta)
                    r_squared = corr ** 2
                    
                    # MAE
                    delta_mae = np.mean(np.abs(pred_delta - gt_delta))
                    
                    print(f"\n[INFO] Delta Analysis:")
                    print(f"  - 환자 수: {n_patients}명")
                    print(f"  - 총 이미지 쌍 수: {n_pairs}개")
                    # print(f"  - Pearson r: {corr:.4f}")
                    print(f"  - R²: {r_squared:.4f}")
                    print(f"  - Δ MAE: {delta_mae:.2f} years")
                    
                    # Delta Scatter Plot
                    fig, ax = plt.subplots(figsize=(8, 7))
                    
                    ax.scatter(gt_delta, pred_delta, alpha=0.4, s=15, color='blue', edgecolors='none')
                    
                    # 이상적인 선 (y = x)
                    max_val = max(gt_delta.max(), pred_delta.max())
                    min_val = min(gt_delta.min(), pred_delta.min())
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7)
                    
                    # 통계 정보 표시
                    if p_val < 0.01:
                        p_str = "p < 0.01"
                    else:
                        p_str = f"p = {p_val:.4f}"
                    
                    stats_text = (#f"Pearson r = {corr:.4f}\n"
                                 f"R² = {r_squared:.4f}\n"
                                 f"{p_str}\n"
                                 f"Δ = {delta_mae:.2f} years")
                    
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           family='monospace')
                    
                    ax.set_xlabel('Actual Age Difference', fontsize=12)
                    ax.set_ylabel('Predicted Age Difference', fontsize=12)
                    ax.set_title('Siamese_Network_Attention', 
                                fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal', adjustable='box')
                    ax.set_xlim([min_val - 1, max_val + 1])
                    ax.set_ylim([min_val - 1, max_val + 1])
                    
                    plt.tight_layout()
                    delta_plot_path = os.path.join(args.output_dir, "delta_scatter_plot.png")
                    plt.savefig(delta_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"[INFO] Delta Scatter Plot saved: {delta_plot_path}")
                    
                    # Delta pairs CSV 저장
                    delta_csv_path = os.path.join(args.output_dir, "delta_pairs.csv")
                    delta_df.to_csv(delta_csv_path, index=False)
                    print(f"[INFO] Delta pairs saved: {delta_csv_path}")
    
    print("\n[INFO] Inference completed!")
    return result_df


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    run_inference(args)

