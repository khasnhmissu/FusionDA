"""
Training Visualization for FusionDA
====================================
Generate static plots for training analysis.

Features:
- Loss curves (box, cls, distill, domain)
- Metric curves (mAP50, mAP50-95)
- Multi-run comparison
- Automatic subplot layout

Usage:
    # Single run
    python utils/visualize_training.py --log-dir runs/fda/exp
    
    # Compare multiple runs
    python utils/visualize_training.py --log-dir runs/fda/exp1 runs/fda/exp2 --names "baseline" "with_grl"
    
    # Demo mode
    python utils/visualize_training.py --demo
"""

import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44AF69']
FIGSIZE = (14, 10)
DPI = 150


def load_csv(path: str) -> list:
    """Load CSV file as list of dicts."""
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def load_training_log(log_dir: str) -> dict:
    """
    Load all training logs from a directory.
    
    Returns:
        dict with keys: 'epoch', 'iteration', 'config'
    """
    log_dir = Path(log_dir)
    result = {
        'epoch': load_csv(log_dir / 'epoch_metrics.csv'),
        'iteration': load_csv(log_dir / 'iteration_losses.csv'),
        'config': {},
        'name': log_dir.name,
    }
    
    config_file = log_dir / 'training_config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            result['config'] = json.load(f)
    
    # Also check for legacy format (training_losses.csv from YOLOv5)
    legacy_csv = log_dir / 'training_losses.csv'
    if legacy_csv.exists() and not result['epoch']:
        result['epoch'] = load_csv(legacy_csv)
    
    return result


def extract_numeric_columns(data: list) -> dict:
    """Extract numeric columns from CSV data."""
    if not data:
        return {}
    
    result = defaultdict(list)
    for row in data:
        for k, v in row.items():
            try:
                result[k].append(float(v))
            except (ValueError, TypeError):
                pass
    
    return dict(result)


def plot_loss_curves(
    logs: list,
    names: list = None,
    save_path: str = None,
    title: str = "Training Losses",
):
    """
    Plot loss curves for one or more training runs.
    
    Args:
        logs: List of training log dicts
        names: Display names for each run
        save_path: Path to save figure
        title: Plot title
    """
    if not logs:
        print("No data to plot")
        return
    
    if names is None:
        names = [log.get('name', f'run_{i}') for i, log in enumerate(logs)]
    
    # Determine which loss columns exist
    loss_keys = set()
    for log in logs:
        data = extract_numeric_columns(log.get('epoch', []))
        for k in data.keys():
            if 'loss' in k.lower() or k in ['box', 'cls', 'obj', 'dfl', 'distill', 'domain']:
                loss_keys.add(k)
    
    if not loss_keys:
        # Try iteration data
        for log in logs:
            data = extract_numeric_columns(log.get('iteration', []))
            for k in data.keys():
                if 'loss' in k.lower():
                    loss_keys.add(k)
    
    loss_keys = sorted(loss_keys)
    
    if not loss_keys:
        print("No loss columns found in data")
        return
    
    # Layout: 2 columns, N rows
    n_losses = len(loss_keys)
    ncols = 2
    nrows = (n_losses + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten() if n_losses > 1 else [axes]
    
    for idx, loss_key in enumerate(loss_keys):
        ax = axes[idx]
        
        for run_idx, log in enumerate(logs):
            # Try epoch data first
            data = extract_numeric_columns(log.get('epoch', []))
            
            if loss_key in data:
                epochs = data.get('epoch', list(range(len(data[loss_key]))))
                ax.plot(epochs, data[loss_key], 
                       color=COLORS[run_idx % len(COLORS)],
                       label=names[run_idx],
                       linewidth=2)
            else:
                # Try iteration data with smoothing
                data = extract_numeric_columns(log.get('iteration', []))
                if loss_key in data:
                    values = data[loss_key]
                    # Smooth with moving average
                    window = max(1, len(values) // 100)
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    x = np.linspace(0, len(data.get('epoch', [0])) or len(values), len(smoothed))
                    ax.plot(x, smoothed,
                           color=COLORS[run_idx % len(COLORS)],
                           label=names[run_idx],
                           linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(loss_key.replace('_', ' ').title())
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_losses, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_metrics(
    logs: list,
    names: list = None,
    save_path: str = None,
    title: str = "Training Metrics",
):
    """
    Plot metric curves (mAP, precision, recall).
    """
    if not logs:
        print("No data to plot")
        return
    
    if names is None:
        names = [log.get('name', f'run_{i}') for i, log in enumerate(logs)]
    
    # Determine which metric columns exist
    metric_keys = set()
    for log in logs:
        data = extract_numeric_columns(log.get('epoch', []))
        for k in data.keys():
            if any(m in k.lower() for m in ['map', 'precision', 'recall', 'f1', 'accuracy']):
                metric_keys.add(k)
    
    metric_keys = sorted(metric_keys)
    
    if not metric_keys:
        print("No metric columns found in data")
        return
    
    # Layout
    n_metrics = len(metric_keys)
    ncols = min(2, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric_key in enumerate(metric_keys):
        ax = axes[idx]
        
        for run_idx, log in enumerate(logs):
            data = extract_numeric_columns(log.get('epoch', []))
            
            if metric_key in data:
                epochs = data.get('epoch', list(range(len(data[metric_key]))))
                ax.plot(epochs, data[metric_key],
                       color=COLORS[run_idx % len(COLORS)],
                       label=names[run_idx],
                       linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title(metric_key.replace('_', ' ').replace('-', '@'))
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
    
    # Hide unused
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_combined(
    logs: list,
    names: list = None,
    save_path: str = None,
):
    """
    Create combined plot with losses and metrics.
    """
    if not logs:
        print("No data to plot")
        return
    
    if names is None:
        names = [log.get('name', f'run_{i}') for i, log in enumerate(logs)]
    
    # Get all numeric columns
    all_keys = set()
    for log in logs:
        data = extract_numeric_columns(log.get('epoch', []))
        all_keys.update(data.keys())
    
    # Categorize columns
    loss_keys = [k for k in all_keys if 'loss' in k.lower() or k in ['box', 'cls', 'obj', 'dfl']]
    metric_keys = [k for k in all_keys if any(m in k.lower() for m in ['map', 'precision', 'recall'])]
    other_keys = [k for k in all_keys if k not in loss_keys + metric_keys + ['epoch', 'timestamp']]
    
    # Create figure with subplots
    n_plots = len(loss_keys) + len(metric_keys) + (1 if other_keys else 0)
    if n_plots == 0:
        print("No data columns found")
        return
    
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
    
    plot_idx = 1
    
    # Plot losses
    for loss_key in sorted(loss_keys):
        ax = fig.add_subplot(nrows, ncols, plot_idx)
        plot_idx += 1
        
        for run_idx, log in enumerate(logs):
            data = extract_numeric_columns(log.get('epoch', []))
            if loss_key in data:
                epochs = data.get('epoch', list(range(len(data[loss_key]))))
                ax.plot(epochs, data[loss_key],
                       color=COLORS[run_idx % len(COLORS)],
                       label=names[run_idx], linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(loss_key.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot metrics
    for metric_key in sorted(metric_keys):
        ax = fig.add_subplot(nrows, ncols, plot_idx)
        plot_idx += 1
        
        for run_idx, log in enumerate(logs):
            data = extract_numeric_columns(log.get('epoch', []))
            if metric_key in data:
                epochs = data.get('epoch', list(range(len(data[metric_key]))))
                ax.plot(epochs, data[metric_key],
                       color=COLORS[run_idx % len(COLORS)],
                       label=names[run_idx], linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title(metric_key.replace('_', ' '))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
    
    # Plot other values (lr, alpha, etc)
    if other_keys:
        ax = fig.add_subplot(nrows, ncols, plot_idx)
        for run_idx, log in enumerate(logs):
            data = extract_numeric_columns(log.get('epoch', []))
            for k in sorted(other_keys)[:3]:  # Max 3 lines
                if k in data:
                    epochs = data.get('epoch', list(range(len(data[k]))))
                    ax.plot(epochs, data[k], label=f"{names[run_idx]}:{k}", linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_title('Other Parameters')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_report(log_dir: str, output_dir: str = None):
    """
    Generate complete training report with all plots.
    
    Args:
        log_dir: Path to training log directory
        output_dir: Output directory for plots (default: log_dir/plots)
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir) if output_dir else log_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating Training Report")
    print(f"Log dir: {log_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    logs = [load_training_log(log_dir)]
    
    if not logs[0]['epoch'] and not logs[0]['iteration']:
        print("ERROR: No training data found!")
        return
    
    # Generate plots
    plot_loss_curves(logs, save_path=output_dir / 'losses.png')
    plot_metrics(logs, save_path=output_dir / 'metrics.png')
    plot_combined(logs, save_path=output_dir / 'combined.png')
    
    print(f"\n✓ Report generated in: {output_dir}")


def compare_runs(log_dirs: list, names: list = None, output_dir: str = None):
    """
    Compare multiple training runs.
    
    Args:
        log_dirs: List of log directory paths
        names: Display names for each run
        output_dir: Output directory for plots
    """
    if names is None:
        names = [Path(d).name for d in log_dirs]
    
    if output_dir is None:
        output_dir = Path(log_dirs[0]).parent / 'comparison'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Comparing {len(log_dirs)} Training Runs")
    print(f"{'='*60}")
    for i, (d, n) in enumerate(zip(log_dirs, names)):
        print(f"  [{i+1}] {n}: {d}")
    print(f"Output: {output_dir}\n")
    
    # Load all logs
    logs = [load_training_log(d) for d in log_dirs]
    
    # Generate comparison plots
    plot_loss_curves(logs, names, save_path=output_dir / 'losses_comparison.png',
                    title="Loss Comparison")
    plot_metrics(logs, names, save_path=output_dir / 'metrics_comparison.png',
                title="Metrics Comparison")
    plot_combined(logs, names, save_path=output_dir / 'combined_comparison.png')
    
    print(f"\n✓ Comparison report generated in: {output_dir}")


def demo():
    """Generate demo plots with sample data."""
    import tempfile
    
    print("Generating demo plots...")
    
    # Create sample data
    epochs = list(range(50))
    
    sample_data = []
    for epoch in epochs:
        sample_data.append({
            'epoch': epoch,
            'loss_sr_avg': 2.0 * np.exp(-epoch / 20) + np.random.randn() * 0.1,
            'loss_sf_avg': 1.8 * np.exp(-epoch / 25) + np.random.randn() * 0.08,
            'loss_distill_avg': 0.5 * np.exp(-epoch / 30) + np.random.randn() * 0.05,
            'loss_domain_avg': 0.3 * (1 - np.exp(-epoch / 15)) + np.random.randn() * 0.03,
            'mAP50': 0.3 + 0.5 * (1 - np.exp(-epoch / 20)) + np.random.randn() * 0.02,
            'mAP50-95': 0.2 + 0.4 * (1 - np.exp(-epoch / 25)) + np.random.randn() * 0.02,
            'lr': 0.01 * (1 - epoch / 50),
        })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save sample data
        csv_path = Path(tmpdir) / 'epoch_metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sample_data[0].keys())
            writer.writeheader()
            writer.writerows(sample_data)
        
        # Generate plots
        output_dir = Path('demo_plots')
        output_dir.mkdir(exist_ok=True)
        
        logs = [load_training_log(tmpdir)]
        logs[0]['name'] = 'demo_run'
        
        plot_loss_curves(logs, ['Demo Run'], save_path=output_dir / 'demo_losses.png')
        plot_metrics(logs, ['Demo Run'], save_path=output_dir / 'demo_metrics.png')
        plot_combined(logs, ['Demo Run'], save_path=output_dir / 'demo_combined.png')
        
        print(f"\n✓ Demo plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='FusionDA Training Visualization')
    parser.add_argument('--log-dir', nargs='+', type=str, 
                       help='Training log directory(ies)')
    parser.add_argument('--names', nargs='+', type=str,
                       help='Display names for runs')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--demo', action='store_true',
                       help='Generate demo plots with sample data')
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
        return
    
    if not args.log_dir:
        parser.print_help()
        return
    
    if len(args.log_dir) == 1:
        generate_report(args.log_dir[0], args.output)
    else:
        compare_runs(args.log_dir, args.names, args.output)
    
    plt.show()


if __name__ == '__main__':
    main()
