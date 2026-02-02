"""
Configuration loader for FusionDA training.
Handles loading, merging, and validating config files.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from argparse import Namespace


@dataclass
class ModelConfig:
    weights: str = "yolov8l.pt"
    imgsz: int = 640


@dataclass
class DataConfig:
    config: str = "data.yaml"
    workers: int = 8
    batch_size: int = 4


@dataclass
class TrainingConfig:
    epochs: int = 200
    warmup_epochs: int = 10
    lr0: float = 0.001
    lrf: float = 0.01
    device: str = "0"


@dataclass
class TeacherConfig:
    alpha: float = 0.999
    ema_warmup_epochs: int = 1  # EMA warmup: only 1 epoch delay


@dataclass
class DistillationConfig:
    conf_thres_min: float = 0.15
    conf_thres_max: float = 0.50
    iou_thres: float = 0.45
    lambda_weight: float = 0.1
    use_progressive_lambda: bool = True
    class_mapping: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 1, 2: 1})


@dataclass
class GRLConfig:
    enabled: bool = True
    warmup_epochs: int = 10
    max_alpha: float = 0.5
    weight: float = 0.05
    hidden_dim: int = 256
    dropout: float = 0.3
    lr: float = 0.0001


@dataclass
class OutputConfig:
    project: str = "runs/fda"
    name: str = "exp"


@dataclass
class LoggingConfig:
    log_interval: int = 100
    val_interval: int = 10
    save_interval: int = 10
    enable_monitoring: bool = True


@dataclass
class PerformanceConfig:
    amp: bool = True
    cache_clear_interval: int = 200
    gradient_clip: float = 10.0


@dataclass
class FDAConfig:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    grl: GRLConfig = field(default_factory=GRLConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


def load_config(config_path: Optional[str] = None) -> FDAConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, use defaults.
        
    Returns:
        FDAConfig object with all settings.
    """
    config = FDAConfig()
    
    if config_path is None:
        return config
    
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return config
    
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    if yaml_config is None:
        return config
    
    # Update each section
    if 'model' in yaml_config:
        for k, v in yaml_config['model'].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)
    
    if 'data' in yaml_config:
        for k, v in yaml_config['data'].items():
            if hasattr(config.data, k):
                setattr(config.data, k, v)
    
    if 'training' in yaml_config:
        for k, v in yaml_config['training'].items():
            if hasattr(config.training, k):
                setattr(config.training, k, v)
    
    if 'teacher' in yaml_config:
        for k, v in yaml_config['teacher'].items():
            if hasattr(config.teacher, k):
                setattr(config.teacher, k, v)
    
    if 'distillation' in yaml_config:
        for k, v in yaml_config['distillation'].items():
            if hasattr(config.distillation, k):
                setattr(config.distillation, k, v)
    
    if 'grl' in yaml_config:
        for k, v in yaml_config['grl'].items():
            if hasattr(config.grl, k):
                setattr(config.grl, k, v)
    
    if 'output' in yaml_config:
        for k, v in yaml_config['output'].items():
            if hasattr(config.output, k):
                setattr(config.output, k, v)
    
    if 'logging' in yaml_config:
        for k, v in yaml_config['logging'].items():
            if hasattr(config.logging, k):
                setattr(config.logging, k, v)
    
    if 'performance' in yaml_config:
        for k, v in yaml_config['performance'].items():
            if hasattr(config.performance, k):
                setattr(config.performance, k, v)
    
    return config


def config_to_namespace(config: FDAConfig) -> Namespace:
    """
    Convert FDAConfig to flat Namespace for backward compatibility.
    
    This allows existing code using opt.epochs, opt.batch, etc. to work.
    """
    ns = Namespace()
    
    # Model
    ns.weights = config.model.weights
    ns.imgsz = config.model.imgsz
    
    # Data
    ns.data = config.data.config
    ns.workers = config.data.workers
    ns.batch = config.data.batch_size
    
    # Training
    ns.epochs = config.training.epochs
    ns.warmup_epochs = config.training.warmup_epochs
    ns.lr0 = config.training.lr0
    ns.lrf = config.training.lrf
    ns.device = config.training.device
    
    # Teacher
    ns.teacher_alpha = config.teacher.alpha
    ns.ema_warmup_epochs = config.teacher.ema_warmup_epochs
    
    # Distillation
    ns.conf_thres = config.distillation.conf_thres_min
    ns.conf_thres_max = config.distillation.conf_thres_max
    ns.iou_thres = config.distillation.iou_thres
    ns.lambda_weight = config.distillation.lambda_weight
    ns.use_progressive_lambda = config.distillation.use_progressive_lambda
    ns.class_mapping = config.distillation.class_mapping
    
    # GRL
    ns.use_grl = config.grl.enabled
    ns.grl_warmup = config.grl.warmup_epochs
    ns.grl_max_alpha = config.grl.max_alpha
    ns.grl_weight = config.grl.weight
    ns.grl_hidden_dim = config.grl.hidden_dim
    ns.grl_dropout = config.grl.dropout
    ns.grl_lr = config.grl.lr
    
    # Output
    ns.project = config.output.project
    ns.name = config.output.name
    
    # Logging
    ns.log_interval = config.logging.log_interval
    ns.val_interval = config.logging.val_interval
    ns.save_interval = config.logging.save_interval
    ns.enable_monitoring = config.logging.enable_monitoring
    
    # Performance
    ns.amp = config.performance.amp
    ns.cache_clear_interval = config.performance.cache_clear_interval
    ns.gradient_clip = config.performance.gradient_clip
    
    return ns


def merge_cli_args(config: FDAConfig, args: Namespace) -> FDAConfig:
    """
    Merge command-line arguments into config.
    CLI args take precedence over config file.
    """
    # Map CLI args to config attributes
    cli_mapping = {
        'weights': ('model', 'weights'),
        'imgsz': ('model', 'imgsz'),
        'data': ('data', 'config'),
        'workers': ('data', 'workers'),
        'batch': ('data', 'batch_size'),
        'epochs': ('training', 'epochs'),
        'device': ('training', 'device'),
        'lr0': ('training', 'lr0'),
        'use_grl': ('grl', 'enabled'),
        'name': ('output', 'name'),
        'project': ('output', 'project'),
        'enable_monitoring': ('logging', 'enable_monitoring'),
    }
    
    for arg_name, (section, attr) in cli_mapping.items():
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                section_obj = getattr(config, section)
                if hasattr(section_obj, attr):
                    setattr(section_obj, attr, arg_value)
    
    return config


def print_config(config: FDAConfig):
    """Print configuration summary."""
    print("=" * 60)
    print("FusionDA Training Configuration")
    print("=" * 60)
    print(f"Model:     {config.model.weights}")
    print(f"Data:      {config.data.config}")
    print(f"Epochs:    {config.training.epochs}")
    print(f"Batch:     {config.data.batch_size}")
    print(f"Device:    {config.training.device}")
    print(f"GRL:       {'Enabled' if config.grl.enabled else 'Disabled'}")
    print(f"Warmup:    {config.training.warmup_epochs} epochs")
    print("=" * 60)
