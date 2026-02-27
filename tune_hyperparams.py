"""
Hyperparameter Tuning for FDA Training using Optuna.

This script performs Bayesian optimization to find the best combination of:
- GRL: weight, lr
- Distillation: conf_thres, lambda_weight  
- Teacher EMA: alpha

Disk Management:
    - Automatically cleans up trial outputs after each trial
    - Only keeps best.pt weights (removes last.pt, debug images, logs)
    - Monitors disk space and aborts if below threshold
    - Disables heavy logging (TensorBoard, DomainMonitor) during tuning

Usage:
    python tune_hyperparams.py --n-trials 20 --epochs-per-trial 100
    python tune_hyperparams.py --n-trials 20 --epochs-per-trial 100 --min-disk-gb 3
"""

import argparse
import gc
import shutil
import yaml
import optuna
from pathlib import Path
from datetime import datetime
import torch
import optuna.visualization as vis
from train import train
from utils.config_loader import load_config, config_to_namespace


# ============================================================
# Disk Management Utilities
# ============================================================

def get_disk_free_gb(path='.'):
    """Get free disk space in GB for the drive containing `path`."""
    usage = shutil.disk_usage(Path(path).resolve().anchor)
    return usage.free / (1024 ** 3)


def get_dir_size_mb(path):
    """Get total size of a directory in MB."""
    total = 0
    path = Path(path)
    if path.exists():
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 ** 2)


def cleanup_trial_output(trial_dir, keep_best=True):
    """Clean up trial output directory to save disk space.
    
    Removes:
    - last.pt (only keep best.pt)
    - Debug images
    - TensorBoard logs  
    - Domain monitor outputs (UMAP plots, features)
    - CSV logs
    
    Keeps:
    - best.pt (if keep_best=True)
    """
    trial_dir = Path(trial_dir)
    if not trial_dir.exists():
        return 0.0
    
    freed_mb = 0.0
    
    # Remove last.pt (biggest single file, ~200MB for YOLOv8l)
    last_pt = trial_dir / 'weights' / 'last.pt'
    if last_pt.exists():
        freed_mb += last_pt.stat().st_size / (1024 ** 2)
        last_pt.unlink()
    
    # Remove debug images directory
    debug_dir = trial_dir / 'debug_images'
    if debug_dir.exists():
        freed_mb += get_dir_size_mb(debug_dir)
        shutil.rmtree(debug_dir, ignore_errors=True)
    
    # Remove TensorBoard logs
    for tb_dir in trial_dir.glob('**/events.out.tfevents.*'):
        freed_mb += tb_dir.stat().st_size / (1024 ** 2)
        tb_dir.unlink(missing_ok=True)
    
    # Remove domain monitor outputs
    for pattern in ['domain_monitor', 'umap_*', 'tsne_*', 'mmd_*', 'domain_acc*']:
        for p in trial_dir.rglob(pattern):
            if p.is_dir():
                freed_mb += get_dir_size_mb(p)
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                freed_mb += p.stat().st_size / (1024 ** 2)
                p.unlink(missing_ok=True)
    
    # Remove CSV logs (small but many)
    for csv_file in trial_dir.rglob('*.csv'):
        freed_mb += csv_file.stat().st_size / (1024 ** 2)
        csv_file.unlink(missing_ok=True)
    
    # If not keeping best either, remove entire directory
    if not keep_best:
        remaining = get_dir_size_mb(trial_dir)
        freed_mb += remaining
        shutil.rmtree(trial_dir, ignore_errors=True)
    
    return freed_mb


def create_trial_config(trial, base_config_path):
    """Create a config namespace with sampled hyperparameters."""
    
    # Load base config
    with open(base_config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Sample hyperparameters
    
    # === GRL Parameters ===
    grl_weight = trial.suggest_float('grl_weight', 0.01, 0.2, log=True)
    grl_lr = trial.suggest_float('grl_lr', 1e-5, 1e-3, log=True)
    grl_max_alpha = trial.suggest_float('grl_max_alpha', 0.1, 0.5)
    
    # === Distillation Parameters ===
    distill_conf_min = trial.suggest_float('distill_conf_min', 0.3, 0.6)
    distill_conf_max = trial.suggest_float('distill_conf_max', 0.6, 0.9)
    distill_lambda = trial.suggest_float('distill_lambda', 0.05, 0.5, log=True)
    
    # === EMA Parameters ===
    ema_alpha = trial.suggest_float('ema_alpha', 0.99, 0.9999, log=True)
    
    # Update config
    config['grl']['weight'] = grl_weight
    config['grl']['lr'] = grl_lr
    config['grl']['max_alpha'] = grl_max_alpha
    
    config['distillation']['conf_thres_min'] = distill_conf_min
    config['distillation']['conf_thres_max'] = distill_conf_max
    config['distillation']['lambda_weight'] = distill_lambda
    
    config['teacher']['alpha'] = ema_alpha
    
    return config


def objective(trial, args):
    """Optuna objective function with disk management."""
    
    trial_name = f"trial_{trial.number:03d}"
    trial_dir = Path(args.trial_project) / trial_name
    
    try:
        # === Disk space check before trial ===
        free_gb = get_disk_free_gb(args.trial_project)
        print(f"\n[Trial {trial.number}] Disk free: {free_gb:.1f} GB")
        
        if free_gb < args.min_disk_gb:
            print(f"[Trial {trial.number}] ABORT: Only {free_gb:.1f}GB free "
                  f"(minimum: {args.min_disk_gb}GB). Stopping study.")
            study = trial.study
            study.stop()  # Gracefully stop the study
            return 0.0
        
        # Create trial config
        config = create_trial_config(trial, args.base_config)
        
        # Override epochs for faster trials
        config['training']['epochs'] = args.epochs_per_trial
        
        # Unique output name for this trial
        config['output']['name'] = trial_name
        config['output']['project'] = args.trial_project
        
        # === DISABLE heavy features to save disk ===
        config['logging']['enable_monitoring'] = False  # No DomainMonitor (UMAP plots)
        config['logging']['log_interval'] = 9999        # Minimal iteration logging
        
        # Convert to namespace
        opt = config_to_namespace_from_dict(config)
        
        # Disable debug image saving and TensorBoard for tuning
        opt.tuning_mode = True  # Flag for train() to skip debug images
        opt.enable_monitoring = False  # No domain monitor
        
        # Run training
        best_map50 = train(opt)
        
        print(f"[Trial {trial.number}] mAP@50 = {best_map50:.4f}")
        
        # === Cleanup trial output to save disk ===
        freed_mb = cleanup_trial_output(trial_dir, keep_best=False)
        print(f"[Trial {trial.number}] Cleaned up {freed_mb:.1f} MB")
        
        # Force GPU + Python cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        return best_map50
        
    except Exception as e:
        print(f"[Trial {trial.number}] Failed: {e}")
        
        # Cleanup even on failure
        try:
            if trial_dir.exists():
                freed_mb = cleanup_trial_output(trial_dir, keep_best=False)
                print(f"[Trial {trial.number}] Cleaned up {freed_mb:.1f} MB after failure")
        except Exception:
            pass
        
        gc.collect()
        torch.cuda.empty_cache()
        return 0.0  # Return worst score on failure


def config_to_namespace_from_dict(config):
    """Convert config dict to namespace for train()."""
    from argparse import Namespace
    
    opt = Namespace()
    
    # Model
    opt.weights = config['model']['weights']
    opt.imgsz = config['model']['imgsz']
    
    # Data
    opt.data = config['data']['config']
    opt.workers = config['data']['workers']
    opt.batch = config['data']['batch_size']
    
    # Training
    opt.epochs = config['training']['epochs']
    opt.warmup_epochs = config['training']['warmup_epochs']
    opt.lr0 = config['training']['lr0']
    opt.lrf = config['training']['lrf']
    opt.device = config['training']['device']
    
    # Teacher
    opt.freeze_teacher = config['teacher']['freeze_teacher']
    opt.teacher_alpha = config['teacher']['alpha']
    
    # Distillation
    opt.conf_thres = config['distillation']['conf_thres_min']
    opt.conf_thres_max = config['distillation']['conf_thres_max']
    opt.iou_thres = config['distillation']['iou_thres']
    opt.lambda_weight = config['distillation']['lambda_weight']
    opt.use_progressive_lambda = config['distillation']['use_progressive_lambda']
    opt.class_mapping = config['distillation']['class_mapping']
    
    # GRL
    opt.use_grl = config['grl']['enabled']
    opt.grl_warmup = config['grl']['warmup_epochs']
    opt.grl_max_alpha = config['grl']['max_alpha']
    opt.grl_weight = config['grl']['weight']
    opt.grl_hidden_dim = config['grl']['hidden_dim']
    opt.grl_lr = config['grl'].get('lr', 0.0001)
    
    # Output
    opt.project = config['output']['project']
    opt.name = config['output']['name']
    
    # Logging
    opt.enable_monitoring = config['logging']['enable_monitoring']
    
    # Performance
    opt.amp = config['performance']['amp']
    
    return opt


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for FDA')
    parser.add_argument('--base-config', type=str, default='configs/train_config.yaml',
                        help='Base config file')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials to run')
    parser.add_argument('--epochs-per-trial', type=int, default=20,
                        help='Epochs per trial (shorter for faster tuning)')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume existing study')
    parser.add_argument('--min-disk-gb', type=float, default=3.0,
                        help='Minimum free disk space (GB) before aborting')
    parser.add_argument('--keep-trial-outputs', action='store_true',
                        help='Keep all trial outputs (WARNING: uses lots of disk)')
    args = parser.parse_args()
    
    # Study name
    if args.study_name is None:
        args.study_name = f"fda_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Trial output directory (reused and cleaned between trials)
    args.trial_project = f"runs/tuning/{args.study_name}"
    
    # Storage
    if args.storage is None:
        Path('optuna_studies').mkdir(exist_ok=True)
        args.storage = f"sqlite:///optuna_studies/{args.study_name}.db"
    
    print("=" * 70)
    print("FDA Hyperparameter Tuning (Disk-Managed)")
    print("=" * 70)
    print(f"Study: {args.study_name}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs/trial: {args.epochs_per_trial}")
    print(f"Storage: {args.storage}")
    print(f"Trial outputs: {args.trial_project}")
    print(f"Min disk space: {args.min_disk_gb} GB")
    print(f"Disk free now: {get_disk_free_gb('.'):.1f} GB")
    print(f"Keep outputs: {args.keep_trial_outputs}")
    print("=" * 70)
    
    # Pre-flight disk check
    free_gb = get_disk_free_gb('.')
    if free_gb < args.min_disk_gb:
        print(f"\nERROR: Only {free_gb:.1f}GB free. Need at least {args.min_disk_gb}GB.")
        print("Free up disk space or use --min-disk-gb to lower the threshold.")
        return
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.resume,
        direction='maximize',  # Maximize mAP@50
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best mAP@50: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best config
    best_config = create_trial_config(study.best_trial, args.base_config)
    best_config_path = f"configs/best_config_{args.study_name}.yaml"
    with open(best_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
    print(f"\nBest config saved to: {best_config_path}")
    
    # Final cleanup of any remaining trial directories
    tuning_dir = Path(args.trial_project)
    if tuning_dir.exists() and not args.keep_trial_outputs:
        remaining_size = get_dir_size_mb(tuning_dir)
        if remaining_size > 0:
            shutil.rmtree(tuning_dir, ignore_errors=True)
            print(f"\nFinal cleanup: removed {remaining_size:.1f} MB from {tuning_dir}")
    
    print(f"\nDisk free after tuning: {get_disk_free_gb('.'):.1f} GB")
    
    # Generate importance plot
    try:      
        fig = vis.plot_param_importances(study)
        fig.write_html(f"optuna_studies/{args.study_name}_importance.html")
        print(f"Importance plot: optuna_studies/{args.study_name}_importance.html")
    except Exception as e:
        print(f"Could not generate importance plot: {e}")


if __name__ == '__main__':
    main()
