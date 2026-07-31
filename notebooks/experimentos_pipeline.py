# -*- coding: utf-8 -*-
"""
Experimental pipeline — hierarchical task formulations + class imbalance.

Uniform protocol for all models in the parliamentary offensive-speech study:
  TF-IDF+LR | mBERT | RoBERTa | BERTimbau

Phases:
  0) Data hygiene
  1) Hierarchy (H1/H2), flat-3/4, cascade
  2) Focal Loss + WeightedRandomSampler
  3) Back-translation (training only)
  4) StratifiedGroupKFold by speaker id
  5) Report: timings, confusion matrices, per-class F1, ROC/PR, heatmaps

Used by ``notebooks/experiments_hierarchy_imbalance.ipynb``.
Published result tables/figures live under ``docs/results`` and ``docs/figures``.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = Dataset = WeightedRandomSampler = object  # type: ignore
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config (ajuste no Colab antes de rodar)
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 8
EPOCHS = 12
PATIENCE = 3
LR = 2e-5
MAX_LENGTH = 128

QUICK_MODE = False
RUN_TRANSFORMERS = TORCH_AVAILABLE
RUN_BACKTRANSLATION = TORCH_AVAILABLE
RUN_GROUP_SPLIT = True
SAVE_FIGURES = True
OUTPUT_DIR = Path("outputs")

# Modelos da pesquisa (mesmo protocolo experimental para todos)
FULL_MODELS: Dict[str, str] = {
    "mBERT": "bert-base-multilingual-cased",
    "RoBERTa": "roberta-base",
    "BERTimbau": "neuralmind/bert-base-portuguese-cased",
}

MODELS: Dict[str, str] = dict(FULL_MODELS) if TORCH_AVAILABLE else {}

if QUICK_MODE:
    N_SPLITS = 2
    EPOCHS = 3
    RUN_BACKTRANSLATION = False
    MODELS = {"BERTimbau": FULL_MODELS["BERTimbau"]} if TORCH_AVAILABLE else {}

DATA_URL = (
    "https://raw.githubusercontent.com/alissonf216/"
    "deteccao_discurso_odio_deputados/main/dados/"
    "discursos_deputados_classificados.xlsx"
)

LABEL_DESC = {
    0: "Discurso neutro ou nao ofensivo",
    1: (
        "Discurso potencialmente ofensivo, mas nao diretamente "
        "relacionado a grupos protegidos"
    ),
    2: (
        "Discurso ofensivo direcionado a grupos protegidos, "
        "mas sem incitacao a violencia"
    ),
    3: (
        "Discurso de odio que incita violencia, odio ou "
        "discriminacao contra grupos protegidos"
    ),
}

FLAT3_NAMES = {0: "Neutro", 1: "Ofensa_geral", 2: "Grupos_protegidos"}
BIN_NAMES = {0: "Neutro", 1: "Ofensivo"}
FINE_NAMES = {0: "Ofensa_geral", 1: "Grupos_protegidos"}
FLAT4_NAMES = {0: "Neutro", 1: "Ofensa_geral", 2: "Protegidos_sem_violencia", 3: "Odio"}

DESC_TO_LABEL = {
    "discurso neutro ou nao ofensivo": 0,
    (
        "discurso potencialmente ofensivo, mas nao diretamente "
        "relacionado a grupos protegidos"
    ): 1,
    (
        "discurso ofensivo direcionado a grupos protegidos, "
        "mas sem incitacao a violencia"
    ): 2,
    (
        "discurso de odio que incita violencia, odio ou "
        "discriminacao contra grupos protegidos"
    ): 3,
}

BT_MODELS = {
    "pt_en": "Helsinki-NLP/opus-mt-ROMANCE-en",
    "en_pt": "Helsinki-NLP/opus-mt-en-ROMANCE",
}

if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

ALL_RESULTS: List[dict] = []
ALL_ARTIFACTS: List[dict] = []  # predicoes agregadas para figuras
_bt_cache = {
    "ready": False,
    "tok_pe": None,
    "mdl_pe": None,
    "tok_ep": None,
    "mdl_ep": None,
}
_GLOBAL_T0 = None


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch/transformers nao instalados. "
            "No Colab: !pip install torch transformers sentencepiece openpyxl"
        )


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figuras").mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _as_numpy_1d(values, dtype=None) -> np.ndarray:
    """Materializa Series/Arrow/ChunkedArray em ndarray NumPy puro.

    Necessario no Colab (pandas/pyarrow): `.values` de colunas string pode
    retornar ArrowStringArray/ChunkedArray, que nao aceita indexing com
    arrays de inteiros dos folds (`texts[tr]` -> TypeError).

    Use dtype=object para textos; deixe dtype=None para labels numericos
    (numpy infere int/float a partir da lista Python).
    """
    if isinstance(values, np.ndarray) and getattr(values.dtype, "kind", None) in (
        "i",
        "u",
        "f",
        "b",
    ):
        return np.asarray(values, dtype=dtype) if dtype is not None else values
    seq = values.tolist() if hasattr(values, "tolist") else list(values)
    if dtype is not None:
        return np.asarray(seq, dtype=dtype)
    return np.asarray(seq)


def _save_fig(name: str) -> None:
    if not SAVE_FIGURES:
        return
    ensure_output_dir()
    path = OUTPUT_DIR / "figuras" / f"{name}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"  figura: {path}")


# ---------------------------------------------------------------------------
# Fase 0 — dados
# ---------------------------------------------------------------------------
def normalize_desc(text) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower()
    repl = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "é": "e",
        "ê": "e",
        "í": "i",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ú": "u",
        "ü": "u",
        "ç": "c",
    }
    for a, b in repl.items():
        t = t.replace(a, b)
    return " ".join(t.split())


def load_and_clean(url: str = DATA_URL) -> pd.DataFrame:
    raw = pd.read_excel(url, sheet_name="Dados_completos")
    df = raw[["id_deputado", "frase", "label", "label_descricao"]].copy()
    df["frase"] = df["frase"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    print("Antes da limpeza:", len(df))
    print(df["label"].value_counts().sort_index())

    df["desc_norm"] = df["label_descricao"].map(normalize_desc)
    df["label_from_desc"] = df["desc_norm"].map(DESC_TO_LABEL)
    inconsist = df[df["label_from_desc"].notna() & (df["label"] != df["label_from_desc"])]
    print(f"\nInconsistencias label vs descricao: {len(inconsist)}")
    if len(inconsist):
        print(
            inconsist[["frase", "label", "label_descricao", "label_from_desc"]]
            .head(10)
            .to_string()
        )

    mask_fix = df["label_from_desc"].notna() & (df["label"] != df["label_from_desc"])
    df.loc[mask_fix, "label"] = df.loc[mask_fix, "label_from_desc"].astype(int)
    print(f"Labels corrigidos: {int(mask_fix.sum())}")

    n_before = len(df)
    df = df.drop_duplicates(subset=["frase"], keep="first").reset_index(drop=True)
    print(f"Duplicatas removidas: {n_before - len(df)} | Apos limpeza: {len(df)}")
    print(df["label"].value_counts().sort_index())

    df["y_bin"] = (df["label"] != 0).astype(int)
    df["y_fine"] = np.where(
        df["label"] == 1,
        0,
        np.where(df["label"].isin([2, 3]), 1, np.nan),
    )
    df["y_flat3"] = np.where(df["label"] == 0, 0, np.where(df["label"] == 1, 1, 2))
    df["y_flat4"] = df["label"].astype(int)
    df["label_nome"] = df["label"].map(LABEL_DESC)

    print("\nDerivados:", {
        "y_bin": df["y_bin"].value_counts().to_dict(),
        "y_fine": df["y_fine"].dropna().astype(int).value_counts().to_dict(),
        "y_flat3": df["y_flat3"].value_counts().sort_index().to_dict(),
    })
    return df


# ---------------------------------------------------------------------------
# Model utils
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length: int = MAX_LENGTH):
            self.texts = list(texts)
            self.labels = list(labels)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoding.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            return item

    class FocalLoss(nn.Module):
        def __init__(self, gamma: float = 2.0, weight=None, reduction: str = "mean"):
            super().__init__()
            self.gamma = gamma
            self.weight = weight
            self.reduction = reduction

        def forward(self, logits, targets):
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            targets = targets.long()
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = -((1.0 - pt) ** self.gamma) * log_pt
            if self.weight is not None:
                loss = loss * self.weight.gather(0, targets)
            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss

else:
    TextDataset = None  # type: ignore
    FocalLoss = None  # type: ignore


def make_weighted_sampler(labels: Sequence[int]):
    _require_torch()
    counts = Counter(int(y) for y in labels)
    class_w = {c: 1.0 / n for c, n in counts.items()}
    sample_w = [class_w[int(y)] for y in labels]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def compute_metrics(y_true, y_pred, average_positive=None, y_proba=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
    }
    if average_positive is not None and set(np.unique(y_true)).issubset({0, 1}):
        p, r, f, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=average_positive,
            zero_division=0,
        )
        out["pos_precision"] = p
        out["pos_recall"] = r
        out["pos_f1"] = f
        if y_proba is not None:
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                out["avg_precision"] = float(average_precision_score(y_true, y_proba))
            except ValueError:
                out["roc_auc"] = float("nan")
                out["avg_precision"] = float("nan")
    return out


def summarize_folds(fold_metrics: List[dict]) -> Tuple[dict, dict]:
    keys = fold_metrics[0].keys()
    means, stds = {}, {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None and not (
            isinstance(m[k], float) and np.isnan(m[k])
        )]
        if not vals:
            means[k] = float("nan")
            stds[k] = float("nan")
        else:
            means[k] = float(np.mean(vals))
            stds[k] = float(np.std(vals))
    return means, stds


def register_result(
    experiment,
    model_name,
    means,
    stds,
    extra=None,
    tempo_segundos=None,
) -> dict:
    row = {
        "experimento": experiment,
        "modelo": model_name,
        "macro_f1": means.get("macro_f1"),
        "macro_f1_std": stds.get("macro_f1"),
        "weighted_f1": means.get("weighted_f1"),
        "mcc": means.get("mcc"),
        "mcc_std": stds.get("mcc"),
        "accuracy": means.get("accuracy"),
        "pos_f1": means.get("pos_f1"),
        "pos_precision": means.get("pos_precision"),
        "pos_recall": means.get("pos_recall"),
        "roc_auc": means.get("roc_auc"),
        "avg_precision": means.get("avg_precision"),
        "tempo_segundos": tempo_segundos,
        "tempo_minutos": None if tempo_segundos is None else tempo_segundos / 60.0,
    }
    if extra:
        row.update(extra)
    ALL_RESULTS.append(row)
    t_msg = f" | tempo={tempo_segundos:.1f}s" if tempo_segundos is not None else ""
    print(
        f"[RESULT] {experiment} | {model_name} | "
        f"Macro-F1={row['macro_f1']:.3f}±{row['macro_f1_std']:.3f}{t_msg}"
    )
    return row


def store_artifact(
    experiment: str,
    model_name: str,
    y_true,
    y_pred,
    y_proba=None,
    label_names: Optional[Dict[int, str]] = None,
    texts=None,
):
    ALL_ARTIFACTS.append(
        {
            "experimento": experiment,
            "modelo": model_name,
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "y_proba": None if y_proba is None else np.asarray(y_proba),
            "label_names": label_names,
            "texts": None if texts is None else list(texts),
        }
    )


def _label_names_for_experiment(experiment: str, num_labels: Optional[int] = None):
    e = experiment.lower()
    if "binario" in e or "h1_" in e:
        return BIN_NAMES
    if "fino" in e or "h2_" in e:
        return FINE_NAMES
    if "flat4" in e:
        return FLAT4_NAMES
    if num_labels == 2:
        return BIN_NAMES
    if num_labels == 4:
        return FLAT4_NAMES
    return FLAT3_NAMES


# ---------------------------------------------------------------------------
# Treino / predicao transformers
# ---------------------------------------------------------------------------
def train_transformer_fold(
    model_path: str,
    X_train,
    y_train,
    X_val,
    y_val,
    num_labels: int,
    loss_type: str = "ce_weighted",
    use_sampler: bool = False,
    focal_gamma: float = 2.0,
):
    _require_torch()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_ds = TextDataset(X_train, y_train, tokenizer)
    val_ds = TextDataset(X_val, y_val, tokenizer)

    if use_sampler:
        sampler = make_weighted_sampler(y_train)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    ).to(device)

    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_vec = np.ones(num_labels, dtype=np.float32)
    for c, w in zip(classes, cw):
        weight_vec[int(c)] = w
    class_weights = torch.tensor(weight_vec, dtype=torch.float, device=device)

    if loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
    elif loss_type == "ce_weighted":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = max(1, len(train_loader) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_f1 = -1.0
    best_state = None
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = criterion(outputs.logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
                pred = torch.argmax(outputs.logits, dim=1)
                preds.extend(pred.cpu().numpy())
                true.extend(batch["labels"].cpu().numpy())
        val_f1 = f1_score(true, preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch + 1} | Val Macro-F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer


def predict_transformer(model, tokenizer, texts, labels_dummy=None, return_proba=False):
    if labels_dummy is None:
        labels_dummy = [0] * len(texts)
    loader = DataLoader(
        TextDataset(texts, labels_dummy, tokenizer), batch_size=BATCH_SIZE
    )
    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            if return_proba:
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                # probabilidade da classe positiva (indice 1) se binario
                if prob.shape[1] == 2:
                    probas.extend(prob[:, 1])
                else:
                    probas.extend(prob.max(axis=1))
    if return_proba:
        return np.asarray(preds), np.asarray(probas)
    return np.asarray(preds)


def _iter_splits(texts, labels, groups=None):
    if groups is None:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        return list(splitter.split(texts, labels))
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(splitter.split(texts, labels, groups))


# ---------------------------------------------------------------------------
# CV runners
# ---------------------------------------------------------------------------
def run_tfidf_cv(
    texts,
    labels,
    experiment_name: str,
    binary_pos=None,
    groups=None,
    texts_for_error=None,
):
    texts = _as_numpy_1d(texts, dtype=object)
    labels = _as_numpy_1d(labels)
    groups = None if groups is None else _as_numpy_1d(groups, dtype=object)
    fold_metrics = []
    all_true, all_pred, all_proba = [], [], []
    t0 = time.perf_counter()

    for fold, (tr, te) in enumerate(_iter_splits(texts, labels, groups), 1):
        t_fold = time.perf_counter()
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=10000, class_weight="balanced", solver="lbfgs"
                    ),
                ),
            ]
        )
        pipe.fit(texts[tr], labels[tr])
        pred = pipe.predict(texts[te])
        proba = None
        if binary_pos is not None and hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(texts[te])[:, 1]
                all_proba.extend(proba)
            except Exception:
                proba = None
        m = compute_metrics(labels[te], pred, average_positive=binary_pos, y_proba=proba)
        fold_metrics.append(m)
        all_true.extend(labels[te])
        all_pred.extend(pred)
        print(
            f"[TF-IDF] {experiment_name} fold {fold}: "
            f"Macro-F1={m['macro_f1']:.4f} ({time.perf_counter() - t_fold:.1f}s)"
        )

    tempo = time.perf_counter() - t0
    means, stds = summarize_folds(fold_metrics)
    register_result(experiment_name, "TF-IDF+LR", means, stds, tempo_segundos=tempo)
    print(classification_report(all_true, all_pred, digits=4, zero_division=0))
    store_artifact(
        experiment_name,
        "TF-IDF+LR",
        all_true,
        all_pred,
        y_proba=np.asarray(all_proba) if all_proba else None,
        label_names=_label_names_for_experiment(experiment_name),
        texts=None if texts_for_error is None else list(np.asarray(texts_for_error)),
    )
    return means, stds, np.asarray(all_true), np.asarray(all_pred)


def run_transformer_cv(
    texts,
    labels,
    experiment_name: str,
    num_labels: int,
    binary_pos=None,
    groups=None,
    loss_type: str = "ce_weighted",
    use_sampler: bool = False,
    focal_gamma: float = 2.0,
    augment_fn: Optional[Callable] = None,
):
    if not RUN_TRANSFORMERS:
        print("RUN_TRANSFORMERS=False — pulando transformers")
        return None
    if not MODELS:
        print("MODELS vazio — pulando transformers")
        return None

    texts = _as_numpy_1d(texts, dtype=object)
    labels = _as_numpy_1d(labels)
    groups = None if groups is None else _as_numpy_1d(groups, dtype=object)
    splits = _iter_splits(texts, labels, groups)
    label_names = _label_names_for_experiment(experiment_name, num_labels)

    for model_name, model_path in MODELS.items():
        print(
            f"\n=== {experiment_name} | {model_name} | "
            f"loss={loss_type} sampler={use_sampler} ==="
        )
        fold_metrics = []
        all_true, all_pred, all_proba = [], [], []
        t0 = time.perf_counter()

        for fold, (tr, te) in enumerate(splits, 1):
            t_fold = time.perf_counter()
            print(f"Fold {fold}/{len(splits)}")
            X_tr_full, y_tr_full = texts[tr], labels[tr]
            X_te, y_te = texts[te], labels[te]

            try:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_tr_full,
                    y_tr_full,
                    test_size=0.1,
                    stratify=y_tr_full,
                    random_state=SEED,
                )
            except ValueError:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_tr_full, y_tr_full, test_size=0.1, random_state=SEED
                )

            if augment_fn is not None:
                X_tr, y_tr = augment_fn(list(X_tr), list(y_tr))

            set_seed(SEED + fold)
            model, tokenizer = train_transformer_fold(
                model_path,
                X_tr,
                y_tr,
                X_val,
                y_val,
                num_labels=num_labels,
                loss_type=loss_type,
                use_sampler=use_sampler,
                focal_gamma=focal_gamma,
            )
            want_proba = binary_pos is not None and num_labels == 2
            if want_proba:
                pred, proba = predict_transformer(
                    model, tokenizer, X_te, y_te, return_proba=True
                )
                all_proba.extend(proba)
            else:
                pred = predict_transformer(model, tokenizer, X_te, y_te)
                proba = None

            m = compute_metrics(
                y_te, pred, average_positive=binary_pos, y_proba=proba
            )
            fold_metrics.append(m)
            all_true.extend(y_te)
            all_pred.extend(pred)
            print(
                f"  Test Macro-F1={m['macro_f1']:.4f} MCC={m['mcc']:.4f} "
                f"({time.perf_counter() - t_fold:.1f}s)"
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tempo = time.perf_counter() - t0
        means, stds = summarize_folds(fold_metrics)
        register_result(
            experiment_name,
            model_name,
            means,
            stds,
            extra={"loss": loss_type, "sampler": use_sampler},
            tempo_segundos=tempo,
        )
        print(classification_report(all_true, all_pred, digits=4, zero_division=0))
        store_artifact(
            experiment_name,
            model_name,
            all_true,
            all_pred,
            y_proba=np.asarray(all_proba) if all_proba else None,
            label_names=label_names,
        )
    return True


# ---------------------------------------------------------------------------
# Cascata hierarquica
# ---------------------------------------------------------------------------
def map_cascade_to_flat3(y_bin_pred, y_fine_pred_for_offensive):
    out = []
    j = 0
    for b in y_bin_pred:
        if int(b) == 0:
            out.append(0)
        else:
            fine = int(y_fine_pred_for_offensive[j])
            j += 1
            out.append(1 if fine == 0 else 2)
    return np.asarray(out)


def run_hierarchical_cascade_tfidf(
    df_in: pd.DataFrame,
    experiment_name: str = "Cascade_H_tfidf",
    groups=None,
):
    texts = _as_numpy_1d(df_in["frase"].astype(str), dtype=object)
    y_bin = _as_numpy_1d(df_in["y_bin"])
    y_flat3 = _as_numpy_1d(df_in["y_flat3"])
    y_fine_all = _as_numpy_1d(df_in["y_fine"])
    groups_arr = None if groups is None else _as_numpy_1d(groups, dtype=object)

    fold_metrics = []
    all_true, all_pred = [], []
    t0 = time.perf_counter()

    for fold, (tr, te) in enumerate(_iter_splits(texts, y_flat3, groups_arr), 1):
        t_fold = time.perf_counter()
        pipe1 = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=10000, class_weight="balanced", solver="lbfgs"
                    ),
                ),
            ]
        )
        pipe1.fit(texts[tr], y_bin[tr])

        tr_off = [i for i in tr if y_bin[i] == 1]
        pipe2 = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=10000, class_weight="balanced", solver="lbfgs"
                    ),
                ),
            ]
        )
        pipe2.fit(texts[tr_off], y_fine_all[tr_off].astype(int))

        bin_pred = pipe1.predict(texts[te])
        te_off_idx = np.where(bin_pred == 1)[0]
        fine_pred = (
            pipe2.predict(texts[te][te_off_idx])
            if len(te_off_idx)
            else np.array([], dtype=int)
        )
        flat_pred = map_cascade_to_flat3(bin_pred, fine_pred)
        m = compute_metrics(y_flat3[te], flat_pred)
        fold_metrics.append(m)
        all_true.extend(y_flat3[te])
        all_pred.extend(flat_pred)
        print(
            f"[Cascade TF-IDF] fold {fold}: Macro-F1={m['macro_f1']:.4f} "
            f"({time.perf_counter() - t_fold:.1f}s)"
        )

    tempo = time.perf_counter() - t0
    means, stds = summarize_folds(fold_metrics)
    register_result(
        experiment_name, "TF-IDF+LR cascade", means, stds, tempo_segundos=tempo
    )
    print(classification_report(all_true, all_pred, digits=4, zero_division=0))
    store_artifact(
        experiment_name,
        "TF-IDF+LR cascade",
        all_true,
        all_pred,
        label_names=FLAT3_NAMES,
    )
    return means, stds


def run_hierarchical_cascade_bert(
    df_in: pd.DataFrame,
    experiment_name: str = "Cascade_H_bert",
    groups=None,
    loss_type: str = "ce_weighted",
    use_sampler: bool = False,
):
    if not RUN_TRANSFORMERS or not MODELS:
        print("Cascata BERT pulada")
        return None

    texts = _as_numpy_1d(df_in["frase"].astype(str), dtype=object)
    y_bin = _as_numpy_1d(df_in["y_bin"])
    y_flat3 = _as_numpy_1d(df_in["y_flat3"])
    y_fine_all = _as_numpy_1d(df_in["y_fine"])
    groups_arr = None if groups is None else _as_numpy_1d(groups, dtype=object)
    splits = _iter_splits(texts, y_flat3, groups_arr)

    for model_name, model_path in MODELS.items():
        print(f"\n=== Cascata {experiment_name} | {model_name} ===")
        fold_metrics = []
        all_true, all_pred = [], []
        t0 = time.perf_counter()

        for fold, (tr, te) in enumerate(splits, 1):
            t_fold = time.perf_counter()
            print(f"Fold {fold}")
            try:
                X1_tr, X1_val, y1_tr, y1_val = train_test_split(
                    texts[tr],
                    y_bin[tr],
                    test_size=0.1,
                    stratify=y_bin[tr],
                    random_state=SEED,
                )
            except ValueError:
                X1_tr, X1_val, y1_tr, y1_val = train_test_split(
                    texts[tr], y_bin[tr], test_size=0.1, random_state=SEED
                )
            set_seed(SEED + fold)
            m1, tok1 = train_transformer_fold(
                model_path,
                X1_tr,
                y1_tr,
                X1_val,
                y1_val,
                num_labels=2,
                loss_type=loss_type,
                use_sampler=use_sampler,
            )

            tr_off = [i for i in tr if y_bin[i] == 1]
            X2 = texts[tr_off]
            y2 = y_fine_all[tr_off].astype(int)
            try:
                X2_tr, X2_val, y2_tr, y2_val = train_test_split(
                    X2, y2, test_size=0.1, stratify=y2, random_state=SEED
                )
            except ValueError:
                X2_tr, X2_val, y2_tr, y2_val = train_test_split(
                    X2, y2, test_size=0.1, random_state=SEED
                )
            set_seed(SEED + fold + 17)
            m2, tok2 = train_transformer_fold(
                model_path,
                X2_tr,
                y2_tr,
                X2_val,
                y2_val,
                num_labels=2,
                loss_type=loss_type,
                use_sampler=use_sampler,
            )

            bin_pred = predict_transformer(m1, tok1, texts[te])
            te_off_local = np.where(bin_pred == 1)[0]
            fine_pred = (
                predict_transformer(m2, tok2, texts[te][te_off_local])
                if len(te_off_local)
                else np.array([], dtype=int)
            )
            flat_pred = map_cascade_to_flat3(bin_pred, fine_pred)
            m = compute_metrics(y_flat3[te], flat_pred)
            fold_metrics.append(m)
            all_true.extend(y_flat3[te])
            all_pred.extend(flat_pred)
            print(
                f"  Cascade Macro-F1={m['macro_f1']:.4f} "
                f"({time.perf_counter() - t_fold:.1f}s)"
            )

            del m1, m2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tempo = time.perf_counter() - t0
        means, stds = summarize_folds(fold_metrics)
        register_result(
            experiment_name,
            f"{model_name} cascade",
            means,
            stds,
            extra={"loss": loss_type, "sampler": use_sampler},
            tempo_segundos=tempo,
        )
        print(classification_report(all_true, all_pred, digits=4, zero_division=0))
        store_artifact(
            experiment_name,
            f"{model_name} cascade",
            all_true,
            all_pred,
            label_names=FLAT3_NAMES,
        )
    return True


# ---------------------------------------------------------------------------
# Back-translation
# ---------------------------------------------------------------------------
def _load_bt_models() -> bool:
    if _bt_cache["ready"]:
        return True
    try:
        print("Carregando MarianMT para back-translation...")
        _bt_cache["tok_pe"] = AutoTokenizer.from_pretrained(BT_MODELS["pt_en"])
        _bt_cache["mdl_pe"] = AutoModelForSeq2SeqLM.from_pretrained(
            BT_MODELS["pt_en"]
        ).to(device)
        _bt_cache["tok_ep"] = AutoTokenizer.from_pretrained(BT_MODELS["en_pt"])
        _bt_cache["mdl_ep"] = AutoModelForSeq2SeqLM.from_pretrained(
            BT_MODELS["en_pt"]
        ).to(device)
        _bt_cache["ready"] = True
        print("Back-translation pronto.")
        return True
    except Exception as e:
        print("Falha ao carregar MarianMT:", e)
        return False


def _translate_batch(texts, tokenizer, model, max_length=128, prefix=None):
    outs = []
    bs = 8
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        if prefix:
            chunk = [prefix + t for t in chunk]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(**enc, max_length=max_length, num_beams=2)
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs


def back_translate_texts(texts):
    en = _translate_batch(texts, _bt_cache["tok_pe"], _bt_cache["mdl_pe"])
    pt = _translate_batch(
        en, _bt_cache["tok_ep"], _bt_cache["mdl_ep"], prefix=">>pt<< "
    )
    return pt


def make_minority_augment_fn(minority_labels, max_per_class=200, copies=1):
    def augment_fn(X, y):
        if not _load_bt_models():
            return X, y
        X, y = list(X), list(y)
        by_class: Dict[int, List[str]] = {}
        for t, lab in zip(X, y):
            lab = int(lab)
            if lab in minority_labels:
                by_class.setdefault(lab, []).append(t)
        new_X, new_y = [], []
        for lab, samples in by_class.items():
            take = samples[:max_per_class]
            for _ in range(copies):
                try:
                    synth = back_translate_texts(take)
                except Exception as e:
                    print("BT falhou:", e)
                    continue
                for s in synth:
                    s = (s or "").strip()
                    if s and s not in X and s not in new_X:
                        new_X.append(s)
                        new_y.append(lab)
        print(f"  Augment BT adicionou {len(new_X)} frases")
        return X + new_X, y + new_y

    return augment_fn


# ---------------------------------------------------------------------------
# Fases
# ---------------------------------------------------------------------------
def run_fase1(df_clean: pd.DataFrame) -> None:
    print("=" * 60)
    print("FASE 1 — Hierarquia + flat (TODOS os modelos)")
    print("=" * 60)

    run_tfidf_cv(
        df_clean["frase"], df_clean["y_bin"], "H1_binario", binary_pos=1
    )
    df_off = df_clean[df_clean["y_fine"].notna()].copy()
    df_off["y_fine"] = df_off["y_fine"].astype(int)
    run_tfidf_cv(
        df_off["frase"], df_off["y_fine"], "H2_fino_ofensivos", binary_pos=1
    )
    run_tfidf_cv(df_clean["frase"], df_clean["y_flat3"], "Flat3")
    run_tfidf_cv(df_clean["frase"], df_clean["y_flat4"], "Flat4")

    run_transformer_cv(
        df_clean["frase"], df_clean["y_bin"], "H1_binario", num_labels=2, binary_pos=1
    )
    run_transformer_cv(
        df_off["frase"], df_off["y_fine"], "H2_fino_ofensivos", num_labels=2, binary_pos=1
    )
    run_transformer_cv(df_clean["frase"], df_clean["y_flat3"], "Flat3", num_labels=3)
    run_transformer_cv(df_clean["frase"], df_clean["y_flat4"], "Flat4", num_labels=4)

    run_hierarchical_cascade_tfidf(df_clean, "Cascade_H")
    run_hierarchical_cascade_bert(df_clean, "Cascade_H")


def run_fase2(df_clean: pd.DataFrame) -> None:
    print("=" * 60)
    print("FASE 2 — Focal Loss + Sampler (TODOS os transformers)")
    print("=" * 60)

    run_transformer_cv(
        df_clean["frase"],
        df_clean["y_flat3"],
        "Flat3_focal",
        num_labels=3,
        loss_type="focal",
    )
    run_transformer_cv(
        df_clean["frase"],
        df_clean["y_flat3"],
        "Flat3_sampler",
        num_labels=3,
        loss_type="ce_weighted",
        use_sampler=True,
    )
    run_transformer_cv(
        df_clean["frase"],
        df_clean["y_flat3"],
        "Flat3_focal_sampler",
        num_labels=3,
        loss_type="focal",
        use_sampler=True,
    )
    run_transformer_cv(
        df_clean["frase"],
        df_clean["y_bin"],
        "H1_binario_focal_sampler",
        num_labels=2,
        binary_pos=1,
        loss_type="focal",
        use_sampler=True,
    )
    run_hierarchical_cascade_bert(
        df_clean,
        "Cascade_H_focal_sampler",
        loss_type="focal",
        use_sampler=True,
    )


def run_fase3(df_clean: pd.DataFrame) -> None:
    if not (RUN_BACKTRANSLATION and RUN_TRANSFORMERS and MODELS):
        print("Fase 3 pulada")
        return
    print("=" * 60)
    print("FASE 3 — Back-translation (TODOS os transformers)")
    print("=" * 60)
    if _load_bt_models():
        print("Demo BT:", back_translate_texts(["Discurso potencialmente ofensivo."]))

    aug3 = make_minority_augment_fn({1, 2}, max_per_class=150, copies=1)
    run_transformer_cv(
        df_clean["frase"],
        df_clean["y_flat3"],
        "Flat3_focal_sampler_BT",
        num_labels=3,
        loss_type="focal",
        use_sampler=True,
        augment_fn=aug3,
    )
    run_transformer_cv(
        df_clean["frase"],
        df_clean["y_bin"],
        "H1_binario_focal_sampler_BT",
        num_labels=2,
        binary_pos=1,
        loss_type="focal",
        use_sampler=True,
        augment_fn=make_minority_augment_fn({1}, max_per_class=200, copies=1),
    )


def run_fase4(df_clean: pd.DataFrame) -> None:
    if not RUN_GROUP_SPLIT:
        print("Fase 4 pulada")
        return
    print("=" * 60)
    print("FASE 4 — Split por deputado (TODOS os modelos)")
    print("=" * 60)
    groups = _as_numpy_1d(df_clean["id_deputado"].astype(str), dtype=object)

    run_tfidf_cv(df_clean["frase"], df_clean["y_flat3"], "Flat3_group", groups=groups)
    run_tfidf_cv(
        df_clean["frase"],
        df_clean["y_bin"],
        "H1_binario_group",
        binary_pos=1,
        groups=groups,
    )
    run_hierarchical_cascade_tfidf(df_clean, "Cascade_H_group", groups=groups)

    run_transformer_cv(
        df_clean["frase"],
        df_clean["y_flat3"],
        "Flat3_focal_sampler_group",
        num_labels=3,
        groups=groups,
        loss_type="focal",
        use_sampler=True,
    )
    run_transformer_cv(
        df_clean["frase"],
        df_clean["y_bin"],
        "H1_binario_focal_sampler_group",
        num_labels=2,
        binary_pos=1,
        groups=groups,
        loss_type="focal",
        use_sampler=True,
    )
    run_hierarchical_cascade_bert(
        df_clean,
        "Cascade_H_focal_sampler_group",
        groups=groups,
        loss_type="focal",
        use_sampler=True,
    )


# ---------------------------------------------------------------------------
# Relatorio rico (alem da matriz de confusao)
# ---------------------------------------------------------------------------
def _names_list(label_names: Optional[Dict[int, str]], labels_present) -> List[str]:
    labels_present = sorted(set(int(x) for x in labels_present))
    if not label_names:
        return [str(i) for i in labels_present]
    return [label_names.get(i, str(i)) for i in labels_present]


def plot_confusion_pair(y_true, y_pred, title: str, label_names=None, fname=None):
    labels = sorted(set(y_true) | set(y_pred))
    names = _names_list(label_names, labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=names, yticklabels=names)
    axes[0].set_title(f"{title} (contagem)")
    axes[0].set_xlabel("Predito")
    axes[0].set_ylabel("Real")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", ax=axes[1],
                xticklabels=names, yticklabels=names)
    axes[1].set_title(f"{title} (normalizada)")
    axes[1].set_xlabel("Predito")
    axes[1].set_ylabel("Real")
    plt.tight_layout()
    if fname:
        _save_fig(fname)
    plt.show()


def plot_per_class_f1(y_true, y_pred, title: str, label_names=None, fname=None):
    labels = sorted(set(y_true) | set(y_pred))
    names = _names_list(label_names, labels)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    dfp = pd.DataFrame({"classe": names, "precision": p, "recall": r, "f1": f, "support": s})
    melt = dfp.melt(id_vars=["classe", "support"], value_vars=["precision", "recall", "f1"],
                    var_name="metrica", value_name="valor")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=melt, x="classe", y="valor", hue="metrica")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.tight_layout()
    if fname:
        _save_fig(fname)
    plt.show()
    return dfp


def plot_roc_pr(y_true, y_proba, title: str, fname_prefix=None):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(np.unique(y_true)) < 2:
        print("ROC/PR indisponivel (uma classe apenas)")
        return
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"AUC={auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title(f"ROC — {title}")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].plot(rec, prec, label=f"AP={ap:.3f}")
    axes[1].set_title(f"Precision-Recall — {title}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    plt.tight_layout()
    if fname_prefix:
        _save_fig(fname_prefix)
    plt.show()


def plot_error_examples(artifact: dict, n: int = 5):
    texts = artifact.get("texts")
    if not texts:
        return
    y_true = artifact["y_true"]
    y_pred = artifact["y_pred"]
    names = artifact.get("label_names") or {}
    print(f"\n--- Exemplos de erro: {artifact['experimento']} | {artifact['modelo']} ---")
    fp_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == 0 and p != 0]
    fn_idx = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != 0 and p == 0]
    for tag, idxs in [("FP (neutro->ofensivo)", fp_idx), ("FN (ofensivo->neutro)", fn_idx)]:
        print(f"\n{tag}:")
        for i in idxs[:n]:
            print(f"  true={names.get(int(y_true[i]), y_true[i])} "
                  f"pred={names.get(int(y_pred[i]), y_pred[i])} | {texts[i][:160]}")


def build_heatmaps(results_df: pd.DataFrame):
    if results_df.empty:
        return
    for metric, title, fname in [
        ("macro_f1", "Heatmap Macro-F1 (experimento x modelo)", "heatmap_macro_f1"),
        ("tempo_minutos", "Heatmap tempo (minutos)", "heatmap_tempo_min"),
        ("mcc", "Heatmap MCC", "heatmap_mcc"),
    ]:
        if metric not in results_df.columns:
            continue
        pivot = results_df.pivot_table(
            index="experimento", columns="modelo", values=metric, aggfunc="mean"
        )
        if pivot.empty:
            continue
        plt.figure(figsize=(max(6, 1.4 * pivot.shape[1]), max(4, 0.4 * pivot.shape[0])))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(title)
        plt.tight_layout()
        _save_fig(fname)
        plt.show()


def build_timing_bars(results_df: pd.DataFrame):
    if "tempo_minutos" not in results_df.columns:
        return
    dfp = results_df.dropna(subset=["tempo_minutos"]).copy()
    if dfp.empty:
        return
    dfp["label"] = dfp["experimento"].astype(str) + " | " + dfp["modelo"].astype(str)
    dfp = dfp.sort_values("tempo_minutos", ascending=True)
    plt.figure(figsize=(10, max(4, 0.32 * len(dfp))))
    sns.barplot(data=dfp, y="label", x="tempo_minutos", orient="h", color="#6b4f3a")
    plt.xlabel("Tempo total (minutos)")
    plt.title("Custo computacional por experimento/modelo")
    plt.tight_layout()
    _save_fig("barras_tempo_minutos")
    plt.show()


def report_artifacts(max_cm: int = 12):
    """Gera figuras a partir das predicoes agregadas."""
    print("\n===== RELATORIO DE ARTEFATOS =====")
    shown = 0
    for art in ALL_ARTIFACTS:
        if shown >= max_cm:
            break
        key = f"{art['experimento']}__{art['modelo']}".replace(" ", "_").replace("/", "-")
        title = f"{art['experimento']} | {art['modelo']}"
        plot_confusion_pair(
            art["y_true"], art["y_pred"], title, art.get("label_names"), fname=f"cm_{key}"
        )
        plot_per_class_f1(
            art["y_true"], art["y_pred"], f"P/R/F1 — {title}",
            art.get("label_names"), fname=f"prf1_{key}",
        )
        if art.get("y_proba") is not None and len(np.unique(art["y_true"])) == 2:
            plot_roc_pr(art["y_true"], art["y_proba"], title, fname_prefix=f"rocpr_{key}")
        shown += 1


def build_final_table(
    csv_path: Optional[str] = None,
    generate_plots: bool = True,
):
    ensure_output_dir()
    if csv_path is None:
        csv_path = str(OUTPUT_DIR / "resultados_experimentos_hierarquia_desbalanceamento.csv")

    results_df = pd.DataFrame(ALL_RESULTS)
    if results_df.empty:
        print("Nenhum resultado acumulado.")
        return results_df

    results_df = results_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    results_df["macro_f1_str"] = results_df.apply(
        lambda r: f"{r['macro_f1']:.3f} ± {r['macro_f1_std']:.3f}"
        if pd.notna(r.get("macro_f1"))
        else "",
        axis=1,
    )
    if "tempo_segundos" in results_df.columns:
        def _fmt_tempo(s):
            if pd.isna(s):
                return ""
            if s < 60:
                return f"{s:.1f} s"
            return f"{s/60:.1f} min"

        results_df["tempo_str"] = results_df["tempo_segundos"].apply(_fmt_tempo)

    print("\n===== TABELA COMPARATIVA FINAL =====")
    cols = [
        c
        for c in [
            "experimento",
            "modelo",
            "macro_f1_str",
            "mcc",
            "weighted_f1",
            "pos_f1",
            "roc_auc",
            "avg_precision",
            "tempo_str",
            "loss",
            "sampler",
        ]
        if c in results_df.columns
    ]
    print(results_df[cols].to_string(index=False))
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV: {csv_path}")

    # Compact JSON summary for reports / docs
    summary_path = OUTPUT_DIR / "resumo_resultados.json"
    summary_path.write_text(
        results_df[cols].to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON: {summary_path}")

    if generate_plots:
        plot_df = results_df.dropna(subset=["macro_f1"]).copy()
        plot_df["label"] = (
            plot_df["experimento"].astype(str) + " | " + plot_df["modelo"].astype(str)
        )
        plt.figure(figsize=(10, max(4, 0.32 * len(plot_df))))
        sns.barplot(data=plot_df, y="label", x="macro_f1", orient="h", color="#2c6eae")
        plt.xlabel("Macro-F1 (media CV)")
        plt.title("Comparativo de experimentos x modelos")
        plt.tight_layout()
        _save_fig("barras_macro_f1")
        plt.show()

        build_heatmaps(results_df)
        build_timing_bars(results_df)
        report_artifacts()

    if _GLOBAL_T0 is not None:
        total = time.perf_counter() - _GLOBAL_T0
        print(f"\nTempo TOTAL da execucao: {total/60:.1f} min ({total:.0f}s)")
        meta = {
            "tempo_total_segundos": total,
            "tempo_total_minutos": total / 60.0,
            "modelos": list(MODELS.keys()),
            "n_splits": N_SPLITS,
            "epochs": EPOCHS,
            "device": str(device),
        }
        (OUTPUT_DIR / "meta_execucao.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return results_df


def print_config():
    print("Device:", device)
    print("TORCH_AVAILABLE:", TORCH_AVAILABLE)
    print("QUICK_MODE:", QUICK_MODE)
    print("RUN_TRANSFORMERS:", RUN_TRANSFORMERS)
    print("RUN_BACKTRANSLATION:", RUN_BACKTRANSLATION)
    print("RUN_GROUP_SPLIT:", RUN_GROUP_SPLIT)
    print("Modelos:", list(MODELS.keys()) if MODELS else ["(nenhum — so TF-IDF)"])
    print("N_SPLITS:", N_SPLITS, "| EPOCHS:", EPOCHS, "| BATCH:", BATCH_SIZE)


def run_all(phases: Optional[Sequence[int]] = None):
    """
    Executa o pipeline completo.
    phases: ex. [0,1,2,3,4] ou [0,1] para parcial. None = todas.
    """
    global _GLOBAL_T0
    _GLOBAL_T0 = time.perf_counter()
    set_seed(SEED)
    ensure_output_dir()
    ALL_RESULTS.clear()
    ALL_ARTIFACTS.clear()
    print_config()

    if phases is None:
        phases = [0, 1, 2, 3, 4]

    df_clean = None
    if 0 in phases or any(p in phases for p in [1, 2, 3, 4]):
        df_clean = load_and_clean()

    if 1 in phases:
        run_fase1(df_clean)
    if 2 in phases:
        run_fase2(df_clean)
    if 3 in phases:
        run_fase3(df_clean)
    if 4 in phases:
        run_fase4(df_clean)

    return build_final_table()


if __name__ == "__main__":
    run_all()
