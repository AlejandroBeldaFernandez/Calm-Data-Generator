__all__ = [
    "QualityReporter",
    "BaseReporter",
    "DiscriminatorReporter",
    "ExternalReporter",
    "Visualizer",
    "LocalIndexGenerator",
]

_lazy_map = {
    "QualityReporter": "calm_data_generator.reports.QualityReporter",
    "BaseReporter": "calm_data_generator.reports.base",
    "DiscriminatorReporter": "calm_data_generator.reports.DiscriminatorReporter",
    "ExternalReporter": "calm_data_generator.reports.ExternalReporter",
    "Visualizer": "calm_data_generator.reports.Visualizer",
    "LocalIndexGenerator": "calm_data_generator.reports.LocalIndexGenerator",
}


def __getattr__(name: str):
    if name in _lazy_map:
        import importlib
        mod = importlib.import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'calm_data_generator.reports' has no attribute {name!r}")
