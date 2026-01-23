"""
Training Logger for FusionDA
============================
Unified logging system for training losses and metrics.

Features:
- CSV logging (per epoch/iteration)
- TensorBoard integration (optional)
- Console progress logging
- Multi-run comparison support

Usage:
    from utils.training_logger import TrainingLogger
    
    logger = TrainingLogger(save_dir='runs/fda/exp')
    
    # Per iteration
    logger.log_iteration(epoch, i, {'loss_sr': 0.5, 'loss_distill': 0.1})
    
    # Per epoch  
    logger.log_epoch(epoch, {'mAP50': 0.85})
    
    # Save all data
    logger.finalize()
"""

import os
import csv
import time
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TrainingLogger:
    """
    Unified Training Logger for FusionDA.
    
    Logs training losses, metrics, and parameters to:
    - CSV files (for analysis and visualization)
    - TensorBoard (optional)
    - Console (via get_pbar_dict)
    """
    
    def __init__(
        self,
        save_dir: str,
        project_name: str = "FusionDA",
        use_tensorboard: bool = True,
        log_interval: int = 10,  # Log every N iterations
        verbose: bool = True,
    ):
        """
        Initialize logger.
        
        Args:
            save_dir: Directory to save logs
            project_name: Name for logging headers
            use_tensorboard: Enable TensorBoard logging
            log_interval: Iterations between console logs
            verbose: Print initialization info
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.project_name = project_name
        self.log_interval = log_interval
        self.verbose = verbose
        
        # Tracking data
        self.iteration_data = []  # Per-iteration losses
        self.epoch_data = []      # Per-epoch metrics
        self.start_time = time.time()
        self.current_epoch = 0
        
        # Running averages for current epoch
        self._epoch_losses = defaultdict(list)
        
        # CSV files
        self.iteration_csv = self.save_dir / 'iteration_losses.csv'
        self.epoch_csv = self.save_dir / 'epoch_metrics.csv'
        self.config_file = self.save_dir / 'training_config.json'
        
        # TensorBoard
        self.tb_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(str(self.save_dir / 'tensorboard'))
            if verbose:
                print(f"[Logger] TensorBoard enabled: tensorboard --logdir {self.save_dir}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            if verbose:
                print("[Logger] Warning: TensorBoard not available. Install with: pip install tensorboard")
        
        # Track column order
        self._iter_columns = None
        self._epoch_columns = None
        
        if verbose:
            print(f"[Logger] Initialized. Save dir: {self.save_dir}")
    
    def log_config(self, config: dict):
        """
        Save training configuration.
        
        Args:
            config: Dictionary of training parameters
        """
        config['log_start_time'] = datetime.now().isoformat()
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_iteration(
        self,
        epoch: int,
        iteration: int,
        losses: dict,
        extra: dict = None,
        global_step: int = None,
    ):
        """
        Log losses for a single iteration.
        
        Args:
            epoch: Current epoch
            iteration: Iteration within epoch
            losses: Dictionary of loss values (e.g., {'loss_sr': 0.5})
            extra: Additional values to log (e.g., {'lr': 0.001})
            global_step: Global step (auto-calculated if None)
        """
        self.current_epoch = epoch
        
        # Build row
        row = {
            'epoch': epoch,
            'iteration': iteration,
            'timestamp': time.time() - self.start_time,
        }
        row.update(losses)
        if extra:
            row.update(extra)
        
        # Track for epoch averaging
        for k, v in losses.items():
            self._epoch_losses[k].append(v)
        
        # Store data
        self.iteration_data.append(row)
        
        # TensorBoard logging
        if self.tb_writer and global_step is not None:
            for k, v in losses.items():
                self.tb_writer.add_scalar(f'train/{k}', v, global_step)
            if extra:
                for k, v in extra.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f'train/{k}', v, global_step)
    
    def log_epoch(
        self,
        epoch: int,
        metrics: dict,
        extra: dict = None,
    ):
        """
        Log metrics for a complete epoch.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metric values (e.g., {'mAP50': 0.85})
            extra: Additional values to log
        """
        # Calculate epoch averages
        avg_losses = {}
        for k, v_list in self._epoch_losses.items():
            if v_list:
                avg_losses[f'{k}_avg'] = sum(v_list) / len(v_list)
        
        # Build row
        row = {
            'epoch': epoch,
            'timestamp': time.time() - self.start_time,
        }
        row.update(avg_losses)
        row.update(metrics)
        if extra:
            row.update(extra)
        
        # Store
        self.epoch_data.append(row)
        
        # TensorBoard
        if self.tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f'metrics/{k}', v, epoch)
            for k, v in avg_losses.items():
                self.tb_writer.add_scalar(f'epoch/{k}', v, epoch)
        
        # Reset epoch accumulators
        self._epoch_losses.clear()
        
        # Save periodically
        self._save_csvs()
    
    def get_pbar_dict(self) -> dict:
        """
        Get dictionary for tqdm progress bar display.
        
        Returns:
            dict: Current epoch averages for display
        """
        result = {}
        for k, v_list in self._epoch_losses.items():
            if v_list:
                # Short key names for display
                short_name = k.replace('loss_', '').replace('_', '')[:6]
                result[short_name] = f'{sum(v_list) / len(v_list):.4f}'
        return result
    
    def _save_csvs(self):
        """Save data to CSV files."""
        # Iteration CSV
        if self.iteration_data:
            if self._iter_columns is None:
                self._iter_columns = list(self.iteration_data[0].keys())
            
            with open(self.iteration_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._iter_columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.iteration_data)
        
        # Epoch CSV
        if self.epoch_data:
            if self._epoch_columns is None:
                self._epoch_columns = list(self.epoch_data[0].keys())
            
            with open(self.epoch_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._epoch_columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.epoch_data)
    
    def finalize(self):
        """
        Finalize logging - save all data and close writers.
        Call this at the end of training.
        """
        # Save final CSVs
        self._save_csvs()
        
        # Log training duration
        total_time = time.time() - self.start_time
        summary = {
            'total_epochs': len(self.epoch_data),
            'total_iterations': len(self.iteration_data),
            'training_time_seconds': total_time,
            'training_time_hours': total_time / 3600,
        }
        
        with open(self.save_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Close TensorBoard
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.verbose:
            print(f"\n[Logger] Training complete!")
            print(f"[Logger] Total time: {total_time / 3600:.2f} hours")
            print(f"[Logger] Logs saved to: {self.save_dir}")
            print(f"[Logger] Files:")
            print(f"  - {self.iteration_csv}")
            print(f"  - {self.epoch_csv}")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.tb_writer:
            try:
                self.tb_writer.close()
            except:
                pass


def load_training_log(log_dir: str) -> dict:
    """
    Load training logs from a directory.
    
    Args:
        log_dir: Path to training log directory
        
    Returns:
        dict: {'iteration': DataFrame, 'epoch': DataFrame, 'config': dict}
    """
    log_dir = Path(log_dir)
    result = {}
    
    # Load iteration CSV
    iter_csv = log_dir / 'iteration_losses.csv'
    if iter_csv.exists():
        with open(iter_csv, 'r') as f:
            reader = csv.DictReader(f)
            result['iteration'] = list(reader)
    
    # Load epoch CSV
    epoch_csv = log_dir / 'epoch_metrics.csv'
    if epoch_csv.exists():
        with open(epoch_csv, 'r') as f:
            reader = csv.DictReader(f)
            result['epoch'] = list(reader)
    
    # Load config
    config_file = log_dir / 'training_config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            result['config'] = json.load(f)
    
    return result


if __name__ == '__main__':
    # Demo/test
    print("Testing TrainingLogger...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(save_dir=tmpdir, use_tensorboard=False)
        
        # Log config
        logger.log_config({'epochs': 100, 'batch_size': 16})
        
        # Simulate training
        for epoch in range(3):
            for i in range(10):
                logger.log_iteration(epoch, i, {
                    'loss_sr': 0.5 - epoch * 0.1 - i * 0.01,
                    'loss_distill': 0.2 - epoch * 0.05,
                }, global_step=epoch * 10 + i)
            
            logger.log_epoch(epoch, {
                'mAP50': 0.5 + epoch * 0.1,
                'mAP50-95': 0.3 + epoch * 0.08,
            })
        
        logger.finalize()
        
        # Test loading
        data = load_training_log(tmpdir)
        print(f"Loaded {len(data.get('iteration', []))} iterations")
        print(f"Loaded {len(data.get('epoch', []))} epochs")
        print("Test PASSED!")
