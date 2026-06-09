"""
Context-Aware Spelling Correction System
==========================================
A production-ready, end-to-end transformer-based seq2seq spelling correction pipeline.

Usage:
    python spell_checker.py         
    python spell_checker.py --correct "Ths is a tst sentance"
"""

import os
import argparse
import glob
import json
import warnings
import difflib
from dataclasses import dataclass, field
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset as HFDataset
import evaluate
from Levenshtein import distance as levenshtein_distance
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS  (override via CLI or edit here)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Files
    train_file:          str   = "train.csv"
    val_file:            str   = "val.csv"
    test_file:           str   = "test.csv"
    input_col:           str   = "augmented_text"
    target_col:          str   = "text"
    save_dir:            str   = "./spelling_correction_model"
    output_dir:          str   = "./checkpoints"
    plots_dir:           str   = "./plots"
    cache_dir:           str   = "./tokenized_cache"

    # Model
    model_name:          str   = "t5-small"

    # Training
    batch_size:          int   = 32
    gradient_acc_steps:  int   = 4
    learning_rate:       float = 2e-4
    max_epochs:          int   = 10
    warmup_ratio:        float = 0.05
    max_input_length:    int   = 32
    max_target_length:   int   = 32
    early_stopping_patience: int = 2

    # Decoding
    beam_width:          int   = 5
    confidence_threshold: float = 0.75

    # Misc
    seed:                int   = 42
    sample_size:         int   = 5     # examples shown in comparison table
    attention_samples:   int   = 3     # sentences used for attention plots


CFG = Config()


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and validate required columns exist."""
    df = pd.read_csv(path)
    assert CFG.target_col in df.columns and CFG.input_col in df.columns, (
        f"Expected columns '{CFG.target_col}' and '{CFG.input_col}' in {path}."
    )
    df = df[[CFG.target_col, CFG.input_col]].dropna().reset_index(drop=True)
    return df


def analyse_errors(df: pd.DataFrame, label: str = "train") -> None:
    """
    Perform character-level diff analysis to log the distribution of
    substitution, insertion, and deletion error types in the dataset.
    """
    subs, ins, dels = 0, 0, 0
    for _, row in df.iterrows():
        src = row[CFG.input_col]
        tgt = row[CFG.target_col]
        ops = difflib.SequenceMatcher(None, src, tgt).get_opcodes()
        for tag, *_ in ops:
            if tag == "replace":
                subs += 1
            elif tag == "insert":
                ins += 1
            elif tag == "delete":
                dels += 1

    total = subs + ins + dels or 1
    print(f"\n[Error Analysis — {label}]")
    print(f"  Substitutions : {subs:>6}  ({100*subs/total:.1f}%)")
    print(f"  Insertions    : {ins:>6}  ({100*ins/total:.1f}%)")
    print(f"  Deletions     : {dels:>6}  ({100*dels/total:.1f}%)")
    print(f"  Total ops     : {total:>6}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TOKENIZATION & DATASET
# ─────────────────────────────────────────────────────────────────────────────

def get_cache_path(split_name: str) -> str:
    """Generate a cache directory path for a given split."""
    # A modell nevéből biztonságos könyvtárnevet csinálunk
    model_safe_name = CFG.model_name.replace("/", "_")
    return os.path.join(
        CFG.cache_dir,
        f"{split_name}_{model_safe_name}_in{CFG.max_input_length}_out{CFG.max_target_length}"
    )


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer,
    split_name: str = "data",
    use_cache: bool = True,
) -> HFDataset:
    """
    Convert a pandas DataFrame into a HuggingFace Dataset with tokenized
    input_ids, attention_mask, and labels.
    
    If use_cache=True, attempts to load from disk first; saves after tokenization.
    """
    cache_path = get_cache_path(split_name)
    
    # ── Próbáljuk betölteni a cache-ből ─────────────────────────────────────
    if use_cache and os.path.isdir(cache_path):
        print(f"  [Cache] Loading {split_name} from: {cache_path}")
        return HFDataset.load_from_disk(cache_path)
    
    # ── Ha nincs cache, tokenizálunk ────────────────────────────────────────
    print(f"  [Tokenize] Processing {split_name}...")
    hf = HFDataset.from_pandas(df)

    prefix = "correct spelling: " if "t5" in CFG.model_name.lower() else ""

    def tokenize_fn(batch):
        inputs = [prefix + t for t in batch[CFG.input_col]]
        model_inputs = tokenizer(
            inputs,
            max_length=CFG.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[CFG.target_col],
            max_length=CFG.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = hf.map(tokenize_fn, batched=True, remove_columns=hf.column_names)
    
    # ── Mentés cache-be ─────────────────────────────────────────────────────
    if use_cache:
        os.makedirs(CFG.cache_dir, exist_ok=True)
        tokenized.save_to_disk(cache_path)
        print(f"  [Cache] Saved {split_name} to: {cache_path}")
    
    return tokenized


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Compute Word Error Rate using the evaluate library."""
    wer_metric = evaluate.load("wer")
    return wer_metric.compute(predictions=predictions, references=references)


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Compute Character Error Rate using the evaluate library."""
    cer_metric = evaluate.load("cer")
    return cer_metric.compute(predictions=predictions, references=references)


def compute_mean_levenshtein(predictions: list[str], references: list[str]) -> float:
    """Compute mean Levenshtein distance between prediction and reference strings."""
    return float(np.mean([
        levenshtein_distance(p, r) for p, r in zip(predictions, references)
    ]))


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score with smoothing."""
    smoothie = SmoothingFunction().method4
    refs  = [[r.split()] for r in references]
    hyps  = [p.split() for p in predictions]
    return corpus_bleu(refs, hyps, smoothing_function=smoothie)


def compute_all_metrics(
    predictions: list[str],
    references:  list[str],
    inputs:      list[str] = None,
    label:       str = "",
) -> dict:
    """
    Compute WER, CER, Levenshtein, and BLEU.
    Returns a dict and prints a formatted summary.
    """
    metrics = {
        "WER":         compute_wer(predictions, references),
        "CER":         compute_cer(predictions, references),
        "Levenshtein": compute_mean_levenshtein(predictions, references),
        "BLEU":        compute_bleu(predictions, references),
        "F0.5":        compute_f05_score(predictions, references)
    }
    print(f"\n[Metrics — {label}]")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    return metrics

def perform_statistical_tests(inputs: list[str], predictions: list[str], references: list[str]) -> None:
    baseline_distances = [levenshtein_distance(inp, ref) for inp, ref in zip(inputs, references)]
    model_distances = [levenshtein_distance(pred, ref) for pred, ref in zip(predictions, references)]
    
    differences = [b - m for b, m in zip(baseline_distances, model_distances)]
    
    # Páros t-próba
    t_stat, p_val_t = stats.ttest_rel(baseline_distances, model_distances)
    
    # Wilcoxon-próba – hibakezelés hozzáadva
    try:
        if all(d == 0 for d in differences):
            print("  Wilcoxon-próba: Nem alkalmazható (minden különbség nulla)")
            p_val_w = None
        else:
            w_stat, p_val_w = stats.wilcoxon(baseline_distances, model_distances)
    except ValueError as e:
        print(f"  Wilcoxon-próba: Nem alkalmazható ({e})")
        p_val_w = None
    
    print("\n[Statisztikai Szignifikancia Tesztek (Levenshtein-távolság alapján)]")
    print(f"  Páros t-próba p-érték       : {p_val_t:.4e} {'(Szignifikáns)' if p_val_t < 0.05 else '(Nem szignifikáns)'}")
    if p_val_w is not None:
        print(f"  Wilcoxon-próba p-érték      : {p_val_w:.4e} {'(Szignifikáns)' if p_val_w < 0.05 else '(Nem szignifikáns)'}")

def compute_f05_score(predictions: list[str], references: list[str]) -> float:
    """
    Kiszámítja a szószintű F0.5 pontszámot.
    
    Indoklás: Helyesírás-javításnál a pontosság (precision) fontosabb, mint a visszahívás (recall).
    A felhasználók számára rendkívül zavaró, ha a modell egy helyesen leírt szót 
    indokolatlanul "kijavít" (false positive). Az F0.5 metrika ezt a preferenciát 
    tükrözi azáltal, hogy a pontosságot nagyobb súllyal veszi figyelembe.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.split())
        ref_words = set(ref.split())
        
        true_positives += len(pred_words.intersection(ref_words))
        false_positives += len(pred_words - ref_words)
        false_negatives += len(ref_words - pred_words)
        
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    if precision == 0 and recall == 0:
        return 0.0
        
    # F0.5 formula
    f05 = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)
    return float(f05)

# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def generate_corrections(
    texts:     list[str],
    model,
    tokenizer,
    device:    str,
    batch_size: int = 32, # Új paraméter a kötegelt feldolgozáshoz
) -> tuple[list[str], list[float]]:
    """
    Run beam search decoding on a list of input texts using BATCHING.
    Returns (corrected_texts, confidence_scores).
    """
    model.eval()
    corrections  = []
    confidences  = []
    prefix       = "correct spelling: " if "t5" in CFG.model_name.lower() else ""

    # Iterálás batch-eken keresztül a tqdm folyamatjelzővel
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating corrections"):
        batch_texts = texts[i : i + batch_size]
        batch_inputs = [prefix + t for t in batch_texts]

        # Tokenizálás egyszerre a teljes batch-re, dinamikus paddinggel
        enc = tokenizer(
            batch_inputs,
            return_tensors="pt",
            max_length=CFG.max_input_length,
            truncation=True,
            padding=True, # Dinamikusan a batch leghosszabb mondatához igazít
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                num_beams=CFG.beam_width,
                max_length=CFG.max_target_length,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # A teljes batch dekódolása egyszerre (sokkal gyorsabb, mint a sima decode)
        batch_decoded = tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        corrections.extend(batch_decoded)

        # Confidence kinyerése az egész batch-re (a korábban megbeszélt hibajavítással)
        if hasattr(outputs, "sequences_scores"):
            batch_probs = torch.exp(outputs.sequences_scores).tolist()
            confidences.extend(batch_probs)
        else:
            confidences.extend([1.0] * len(batch_texts))

    return corrections, confidences

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(history: list[dict]) -> None:
    """Plot training and validation loss curves over exact epochs."""
    os.makedirs(CFG.plots_dir, exist_ok=True)

    # Kinyerjük az értékeket és a hozzájuk tartozó pontos epoch számot
    train_epochs = [e["epoch"] for e in history if "loss" in e and "eval_loss" not in e]
    train_losses = [e["loss"]  for e in history if "loss" in e and "eval_loss" not in e]

    val_epochs   = [e["epoch"] for e in history if "eval_loss" in e]
    val_losses   = [e["eval_loss"] for e in history if "eval_loss" in e]

    plt.figure(figsize=(9, 5))

    # Sima vonal a sűrűbb train loss-nak
    if train_epochs:
        plt.plot(train_epochs, train_losses, "b-", alpha=0.7, label="Train Loss")

    # Pöttyökkel jelölt vonal a ritkább val loss-nak
    if val_epochs:
        plt.plot(val_epochs, val_losses, "r-o", linewidth=2, markersize=6, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(CFG.plots_dir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def plot_metric_comparison(
    baseline_metrics: dict,
    model_metrics:    dict,
    split_label:      str = "Validation",
) -> None:
    """
    Plot grouped bar chart comparing baseline vs. model metrics
    (WER, CER, Levenshtein, BLEU) and save to disk.
    """
    os.makedirs(CFG.plots_dir, exist_ok=True)
    metrics   = list(baseline_metrics.keys())
    baseline  = [baseline_metrics[m] for m in metrics]
    model_out = [model_metrics[m]    for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline,  width, label="Before Model", color="salmon")
    ax.bar(x + width / 2, model_out, width, label="After Model",  color="steelblue")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title(f"Metric Comparison — {split_label} Set")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CFG.plots_dir, f"metric_comparison_{split_label.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_attention_heatmap(
    sentence:  str,
    model,
    tokenizer,
    device:    str,
    idx:       int = 0,
) -> None:
    """
    Visualize the cross-attention weights of the model for a single sentence,
    showing which input tokens the model attends to when generating each output token.
    """
    os.makedirs(CFG.plots_dir, exist_ok=True)
    prefix = "correct spelling: " if "t5" in CFG.model_name.lower() else ""
    enc    = tokenizer(
        prefix + sentence,
        return_tensors="pt",
        max_length=CFG.max_input_length,
        truncation=True,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            num_beams=CFG.beam_width,
            max_length=CFG.max_target_length,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    decoded_tokens = tokenizer.convert_ids_to_tokens(
        outputs.sequences[0], skip_special_tokens=False
    )
    input_tokens = tokenizer.convert_ids_to_tokens(
        enc["input_ids"][0], skip_special_tokens=False
    )

    # Extract cross-attention from the last decoder layer, first beam, averaged over heads
    try:
        cross_attn = outputs.cross_attentions  # tuple of tuples
        # cross_attentions[step][layer] → (batch, heads, 1, src_len)
        last_layer_idx = -1
        attn_matrix = torch.cat(
            [step[last_layer_idx][0].mean(dim=0).squeeze(0).unsqueeze(0)
             for step in cross_attn],
            dim=0,
        ).cpu().numpy()   # (tgt_len, src_len)
    except Exception:
        print(f"  Attention extraction skipped for sample {idx}.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(input_tokens)), max(5, len(decoded_tokens) // 2)))
    sns.heatmap(
        attn_matrix,
        xticklabels=input_tokens,
        yticklabels=decoded_tokens,
        cmap="viridis",
        ax=ax,
        linewidths=0.3,
    )
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Output Tokens")
    ax.set_title(f"Cross-Attention Heatmap — Sample {idx + 1}")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    path = os.path.join(CFG.plots_dir, f"attention_heatmap_{idx}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CHECKPOINT AVERAGING
# ─────────────────────────────────────────────────────────────────────────────
# nincs használva még
def average_checkpoints(checkpoint_dirs: list[str], save_dir: str, tokenizer) -> None:
    """
    Load the top-k checkpoints iteratively to save memory, average their 
    weights, and save the result to the specified save directory.
    """
    k = len(checkpoint_dirs)
    print(f"\n [Checkpoint Averaging] Averaging {k} checkpoints...")
    
    avg_state = {}
    
    # 1. Iteratív betöltés a memóriatúlcsordulás elkerülésére
    for i, ckpt_dir in enumerate(checkpoint_dirs):
        print(f"  Loading checkpoint {i+1}/{k}: {ckpt_dir}")
        
        # Modell betöltése CPU-ra a VRAM kímélése érdekében
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir)
        state_dict = model.state_dict()
        
        for key in state_dict:
            if key not in avg_state:
                # Első modell esetén lemásoljuk a tenzort CPU-ra
                avg_state[key] = state_dict[key].clone().detach().cpu()
            else:
                # Későbbi modellek esetén hozzáadjuk a meglévőhöz
                avg_state[key] += state_dict[key].cpu()
                
        # Memória explicit felszabadítása iterációnként
        del model
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # 2. Átlagolás (osztás k-val)
    for key in avg_state:
        avg_state[key] = avg_state[key] / float(k)
        
    # 3. Új modell inicializálása az első checkpoint architektúrája alapján, 
    # majd az átlagolt súlyok betöltése
    print("  Saving averaged model...")
    final_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dirs[0])
    final_model.load_state_dict(avg_state)
    
    # Mentés a kimeneti mappába
    os.makedirs(save_dir, exist_ok=True)
    final_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  Averaged model saved to: {save_dir}")

def get_checkpoint_loss(ckpt_dir: str) -> float:
    """
    Kinyeri a legalacsonyabb eval_loss értékét a megadott checkpoint 
    trainer_state.json fájljából. Ha nem találja, végtelent (inf) ad vissza.
    """
    state_file = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(state_file):
        return float("inf")
    
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
            # A log_history-ban megkeressük az eval_loss-t
            for log in reversed(state.get("log_history", [])):
                if "eval_loss" in log:
                    return log["eval_loss"]
    except Exception as e:
        print(f"Hiba a {state_file} olvasásakor: {e}")
        
    return float("inf")

def run_checkpoint_averaging_pipeline(args: argparse.Namespace) -> None:
    """
    Összegyűjti a checkpointokat, kiválasztja a legjobb eval loss-szal rendelkezőket,
    végrehajtja az átlagolást, majd lefuttatja a teszt halmaz kiértékelését.
    """
    # 1. Checkpoint mappák megkeresése a kimeneti mappában
    checkpoint_pattern = os.path.join(CFG.output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print(f"Hiba: Nem találtam checkpointokat a {CFG.output_dir} mappában.")
        return
        
    # 2. Checkpointok párosítása a hozzájuk tartozó eval loss értékkel
    checkpoints_with_loss = [(ckpt, get_checkpoint_loss(ckpt)) for ckpt in checkpoints]
    
    # Kiszűrjük azokat, ahol nem találtunk érvényes loss-t
    checkpoints_with_loss = [c for c in checkpoints_with_loss if c[1] != float("inf")]
    
    if not checkpoints_with_loss:
        print("Hiba: Egyik checkpointban sem találtam 'eval loss' metrikát.")
        return
        
    # 3. Növekvő sorrendbe rendezés a veszteség alapján (legkisebb veszteség = legjobb)
    checkpoints_with_loss.sort(key=lambda x: x[1])
    
    # 4. Top K kiválasztása
    best_checkpoints = [ckpt for ckpt, loss in checkpoints_with_loss[:args.top_k]]
    
    print(f"\n[Top-{args.top_k} Checkpoint kiválasztva átlagolásra]")
    for ckpt, loss in checkpoints_with_loss[:args.top_k]:
        print(f"  {ckpt} (eval loss: {loss:.4f})")
        
    # 5. Tokenizer betöltése és a memóriahatékony átlagolás lefuttatása
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    
    average_checkpoints(best_checkpoints, CFG.save_dir, tokenizer)
    
    # 6. Automatikus kiértékelés az új, átlagolt modellen a teszt halmazon
    evaluate_on_test(CFG)


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config) -> None:
    """
    Full training pipeline:
      1. Load and analyse data
      2. Apply curriculum learning
      3. Tokenize datasets
      4. Fine-tune with early stopping
      5. Average top-k checkpoints
      6. Plot loss curves
    """
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Setup] Using device: {device}")
    print(f"[Setup] Model       : {cfg.model_name}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[Data] Loading CSVs...")
    train_df = load_csv(cfg.train_file)
    val_df   = load_csv(cfg.val_file)

    # ── Tokenizer & model ───────────────────────────────────────────────────
    print(f"\n[Model] Loading tokenizer and model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    # ── Tokenize ────────────────────────────────────────────────────────────
    print("[Data] Tokenizing (or loading from cache)...")
    train_hf = tokenize_dataset(train_df, tokenizer, split_name="train")
    val_hf   = tokenize_dataset(val_df,   tokenizer, split_name="val")

    # ── Training arguments ──────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.max_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_acc_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        fp16=True,
        #bf16=True, 
        seed=cfg.seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience
        )],
    )

    # ── Train ───────────────────────────────────────────────────────────────
    print("\n[Training] Starting fine-tuning...")
    train_result = trainer.train()

    # Save best model to save_dir directly
    os.makedirs(cfg.save_dir, exist_ok=True)
    trainer.save_model(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)
    print(f"\n[Training] Best model saved to {cfg.save_dir}")

    # ── Extract loss history ─────────────────────────────────────────────────
    history = trainer.state.log_history
    train_losses = [e["loss"]      for e in history if "loss"      in e and "eval_loss" not in e]
    val_losses   = [e["eval_loss"] for e in history if "eval_loss" in e]

    print("\n[Plots] Saving loss curves...")
    plot_loss_curves(trainer.state.log_history)

    # ── Validation evaluation ────────────────────────────────────────────────
    print("\n[Evaluation] Evaluating on validation set...")
    val_texts  = val_df[cfg.input_col].tolist()
    val_refs   = val_df[cfg.target_col].tolist()

    model_final = AutoModelForSeq2SeqLM.from_pretrained(cfg.save_dir).to(device)
    tok_final   = AutoTokenizer.from_pretrained(cfg.save_dir)

    val_preds, val_conf = generate_corrections(val_texts, model_final, tok_final, device, batch_size=cfg.batch_size)

    baseline_val  = compute_all_metrics(val_texts,  val_refs, "Validation — Baseline")
    model_val     = compute_all_metrics(val_preds,  val_refs, val_texts, "Validation — Model")
    plot_metric_comparison(baseline_val, model_val, split_label="Validation")

    # ── Attention heatmaps ───────────────────────────────────────────────────
    print("\n[Plots] Generating attention heatmaps...")
    for i, sent in enumerate(val_texts[:cfg.attention_samples]):
        plot_attention_heatmap(sent, model_final, tok_final, device, idx=i)

    # ── Confidence flagging ──────────────────────────────────────────────────
    flagged = [(t, p, c) for t, p, c in zip(val_texts, val_preds, val_conf)
               if c < cfg.confidence_threshold]
    print(f"\n[Confidence] {len(flagged)} / {len(val_preds)} corrections flagged "
          f"(threshold={cfg.confidence_threshold})")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(cfg: Config) -> None:
    """
    Load the saved model from disk and evaluate it on test.csv.
    Prints all metrics and a side-by-side comparison table.
    """
    assert os.path.isdir(cfg.save_dir), (
        f"No saved model found at '{cfg.save_dir}'. Run training first."
    )

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = AutoModelForSeq2SeqLM.from_pretrained(cfg.save_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.save_dir)

    print(f"\n[Test] Loading {cfg.test_file}...")
    test_df   = load_csv(cfg.test_file)
    test_texts = test_df[cfg.input_col].tolist()
    test_refs  = test_df[cfg.target_col].tolist()

    print("[Test] Generating corrections (beam search)...")
    test_preds, test_conf = generate_corrections(test_texts, model, tokenizer, device, batch_size=cfg.batch_size)

    # ── Metrics ──────────────────────────────────────────────────────────────
    baseline_test = compute_all_metrics(test_texts, test_refs, "Test — Baseline")
    model_test    = compute_all_metrics(test_preds, test_refs, "Test — Model")
    plot_metric_comparison(baseline_test, model_test, split_label="Test")

    # Statisztikai próbák lefuttatása az eredeti hibás szöveg, a javított szöveg és a referencia alapján
    perform_statistical_tests(inputs=test_texts, predictions=test_preds, references=test_refs)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print(f"{'CORRUPTED INPUT':<30} | {'MODEL OUTPUT':<30} | {'GROUND TRUTH':<25}")
    print(f"{'─'*90}")
    for i in range(min(cfg.sample_size, len(test_texts))):
        conf_flag = " ⚑" if test_conf[i] < cfg.confidence_threshold else ""
        print(
            f"{test_texts[i][:28]:<30} | "
            f"{test_preds[i][:28]:<30} | "
            f"{test_refs[i][:23]:<25}"
            f"  [conf={test_conf[i]:.2f}{conf_flag}]"
        )
    print(f"{'─'*90}")
    print("  ⚑ = flagged for human review (confidence below threshold)")


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def correct_sentence(sentence: str, cfg: Config) -> None:
    """
    Load the saved model and correct a single input sentence via beam search.
    Prints the correction and its confidence score.
    """
    assert os.path.isdir(cfg.save_dir), (
        f"No saved model found at '{cfg.save_dir}'. Run training first."
    )
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = AutoModelForSeq2SeqLM.from_pretrained(cfg.save_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.save_dir)

    corrections, confidences = generate_corrections([sentence], model, tokenizer, device)
    corrected   = corrections[0]
    confidence  = confidences[0]
    flagged     = confidence < cfg.confidence_threshold

    print(f"\n  Input     : {sentence}")
    print(f"  Corrected : {corrected}")
    print(f"  Confidence: {confidence:.4f}", "⚑ (low confidence)" if flagged else "")

    print("\n  Generating attention heatmap...")
    plot_attention_heatmap(
        sentence, model, tokenizer, device, idx=0
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and apply them onto the global Config."""
    parser = argparse.ArgumentParser(
        description="Context-aware transformer spelling correction system."
    )
    parser.add_argument("--model",       type=str,   default=CFG.model_name)
    parser.add_argument("--batch-size",  type=int,   default=CFG.batch_size)
    parser.add_argument("--lr",          type=float, default=CFG.learning_rate)
    parser.add_argument("--max-epochs",  type=int,   default=CFG.max_epochs)
    parser.add_argument("--beam-width",  type=int,   default=CFG.beam_width)
    parser.add_argument("--patience",    type=int,   default=CFG.early_stopping_patience)
    parser.add_argument("--conf-thresh", type=float, default=CFG.confidence_threshold)
    parser.add_argument("--save-dir",    type=str,   default=CFG.save_dir)
    parser.add_argument("--correct",     type=str,   default=None,
                        help="Skip training and correct a single sentence.")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Skip training and evaluate the saved model on test.csv.")
    parser.add_argument("--average-checkpoints", action="store_true",
                        help="Skip training and average existing checkpoints.")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of latest checkpoints to average.")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    """Apply parsed CLI arguments onto the global CFG dataclass."""
    CFG.model_name               = args.model
    CFG.batch_size               = args.batch_size
    CFG.learning_rate            = args.lr
    CFG.max_epochs               = args.max_epochs
    CFG.beam_width               = args.beam_width
    CFG.early_stopping_patience  = args.patience
    CFG.confidence_threshold     = args.conf_thresh
    CFG.save_dir                 = args.save_dir


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    apply_args(args)

    if args.correct:
        # Interactive correction mode
        correct_sentence(args.correct, CFG)

    elif args.eval_only:
        # Evaluation only — skip training
        evaluate_on_test(CFG)

    elif args.average_checkpoints:
        # Checkpoint averaging mode — skip training
        run_checkpoint_averaging_pipeline(args)

    else:
        # Full pipeline: train → evaluate validation → evaluate test
        train(CFG)
        evaluate_on_test(CFG)


if __name__ == "__main__":
    main()
