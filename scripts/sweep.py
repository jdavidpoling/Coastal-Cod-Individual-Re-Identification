"""Optuna hyperparameter search for the cod-reid fine-tunes.

TPE (multivariate) Bayesian search with median pruning. Each trial runs the
model's training script as an isolated subprocess so CUDA state never bleeds
across trials. The script flushes per-epoch oos_mAP to sweep_metrics.jsonl,
which we tail to prune weak trials mid-run. Objective = best oos_mAP from
history.json; oos_top1 at that same epoch is stored as a trial user attr.

MegaDescriptor and MiewID load differently (timm vs transformers AutoModel), so
each has its own training script and its own study — Swin and EfficientNetV2
have different optima, so keeping the TPE models separate is correct anyway.

Run:   python sweep.py --model megadescriptor --trials 30
       python sweep.py --model miewid --trials 30
Resume: same command -- the SQLite study is reused (load_if_exists).
Parallel: launch the same command on another GPU with CUDA_VISIBLE_DEVICES set;
          agents share the study via the SQLite file.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import optuna

HERE = Path(__file__).parent
SWEEP_ROOT = Path("/.. param_search/results")
STORAGE = f"sqlite:///{SWEEP_ROOT / 'reid_hpo.db'}"

# Each model maps to its own training script (model identity is baked into the
# script; nothing about the backbone is injected here).
MODELS = {
    "megadescriptor": HERE / "/.. /fine-tuning_MD.py",
    "miewid":         HERE / "/.. /fine-tuning_MW.py",
    "dinov3":         HERE / "/.. /fine-tuning_DINO.py",
}


def suggest(trial):
    """Search space. ft_lr is the single most important knob (Swin wants the
    low end); the rest are scoped to ranges sane for small-set reid fine-tuning."""
    return {
        "SWEEP_LP_LR":          f"{trial.suggest_float('lp_lr', 1e-4, 5e-3, log=True):.6g}",
        "SWEEP_FT_LR":          f"{trial.suggest_float('ft_lr', 1e-6, 1e-4, log=True):.6g}",
        "SWEEP_LP_EPOCHS":      str(trial.suggest_int('lp_epochs', 1, 4)),
        "SWEEP_WEIGHT_DECAY":   f"{trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True):.6g}",
        "SWEEP_ARCFACE_MARGIN": f"{trial.suggest_float('arcface_margin', 0.2, 0.6):.4g}",
        "SWEEP_ARCFACE_SCALE":  str(trial.suggest_categorical('arcface_scale', [16, 32, 48])),
    }


def objective(trial):
    run_dir = SWEEP_ROOT / args.model / f"trial_{trial.number:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "sweep_metrics.jsonl"

    env = {**os.environ, **suggest(trial),
           "SWEEP_MODE": "1", "SWEEP_RUN_DIR": str(run_dir)}

    proc = subprocess.Popen([os.environ.get("PYTHON", "python"), str(MODELS[args.model])],
                            env=env, stdout=open(run_dir / "train.log", "w"),
                            stderr=subprocess.STDOUT)

    # Tail the metrics file: report each FT-phase oos_mAP to the pruner using an
    # FT-relative step (so trials with different lp_epochs are compared like for
    # like), and kill the subprocess if Optuna says to prune.
    seen, ft_step = 0, 0
    while proc.poll() is None:
        time.sleep(15)
        if not metrics_file.exists():
            continue
        lines = metrics_file.read_text().splitlines()
        for line in lines[seen:]:
            rec = json.loads(line)
            if rec["phase"] == "FT" and rec["oos_mAP"] is not None:
                trial.report(rec["oos_mAP"], ft_step)
                ft_step += 1
                if trial.should_prune():
                    proc.terminate()
                    proc.wait(timeout=60)
                    raise optuna.TrialPruned()
        seen = len(lines)

    if proc.returncode != 0:
        print(f"[trial {trial.number}] training exited {proc.returncode}; scoring 0.")
        return 0.0

    history = json.loads((run_dir / "history.json").read_text())
    oos = [m for m in history.get("oos_mAP", []) if m is not None]
    if oos:
        best_map = max(oos)
        # top-1 at the epoch that gave the best mAP (the checkpoint you'd deploy),
        # not its independent max -- logged for the mAP-vs-top1 comparison.
        i = history["oos_mAP"].index(best_map)
        trial.set_user_attr("oos_top1", history["oos_top1"][i])
        return best_map
    return max(history.get("val_mAP", [0.0]) or [0.0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    (SWEEP_ROOT / args.model).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=f"{args.model}_oosmap4",
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42),
        # Don't prune during LP recovery: n_warmup_steps counts FT-relative steps,
        # so a trial gets >=2 FT validations before it can be cut.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=args.trials)

    print(f"\nBest oos_mAP: {study.best_value*100:.2f}%")
    _t1 = study.best_trial.user_attrs.get("oos_top1")
    print(f"oos_top1 at that trial: {_t1*100:.2f}%" if _t1 is not None else "oos_top1: n/a")
    print("Best params:", json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(SWEEP_ROOT / args.model / "trials.csv", index=False)
