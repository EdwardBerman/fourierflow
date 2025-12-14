from .grid_2d_markov import Grid2DMarkovExperiment
from .grid_2d_rollout import Grid2DRolloutExperiment
try:
    from .learned_interpolator import LearnedInterpolator
except ModuleNotFoundError:
    # Optional JAX stack (haiku, optax, jax)
    LearnedInterpolator = None
from .meshgraphnet_jax import MeshGraphNet
from .point_cloud import PointCloudExperiment
from .structured_mesh import StructuredMeshExperiment
