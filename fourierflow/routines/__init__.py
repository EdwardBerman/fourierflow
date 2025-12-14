from .grid_2d_markov import Grid2DMarkovExperiment
from .grid_2d_rollout import Grid2DRolloutExperiment
LearnedInterpolator = None
try:
    from .learned_interpolator import LearnedInterpolator  # noqa: F401
except Exception:
    # Catch *any* import-time failure from the optional JAX stack:
    # haiku/optax/jax_cfd/gin/etc.
    LearnedInterpolator = None
from .meshgraphnet_jax import MeshGraphNet
from .point_cloud import PointCloudExperiment
from .structured_mesh import StructuredMeshExperiment
