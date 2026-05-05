from .metrics import MetricTracker
from .logger import TrainingLogger, MultiSeedLogger
from .reproducibility import (
    collect_environment,
    get_rng_state,
    make_generator,
    seed_everything,
    seed_worker,
    set_rng_state,
)

# visualize.py lives at the project root (not inside utils/).
# Import it via sys.path so it works regardless of CWD.
try:
    from visualize import (
        GradCAM,
        plot_gradcam_grid,
        plot_feature_maps,
        plot_feature_maps_detailed,
        plot_confusion_matrix,
        plot_training_curves,
        plot_per_class_metrics,
        plot_attention_weights,
    )
except ImportError:
    import importlib.util as _ilu
    import os as _os
    import sys as _sys

    _vis_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "visualize.py")
    if _os.path.isfile(_vis_path):
        _spec = _ilu.spec_from_file_location("visualize", _vis_path)
        _vis = _ilu.module_from_spec(_spec)
        _sys.modules.setdefault("visualize", _vis)
        _spec.loader.exec_module(_vis)

        GradCAM                    = _vis.GradCAM
        plot_gradcam_grid          = _vis.plot_gradcam_grid
        plot_feature_maps          = _vis.plot_feature_maps
        plot_feature_maps_detailed = _vis.plot_feature_maps_detailed
        plot_confusion_matrix      = _vis.plot_confusion_matrix
        plot_training_curves       = _vis.plot_training_curves
        plot_per_class_metrics     = _vis.plot_per_class_metrics
        plot_attention_weights     = _vis.plot_attention_weights
    else:
        import warnings
        warnings.warn(
            "visualize.py not found — plotting functions unavailable. "
            "Ensure visualize.py is in the project root.",
            ImportWarning,
            stacklevel=2,
        )
