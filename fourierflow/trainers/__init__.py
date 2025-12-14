JAXTrainer = None
try:
    from .jax_trainer import JAXTrainer  # noqa: F401
except Exception:
    JAXTrainer = None
