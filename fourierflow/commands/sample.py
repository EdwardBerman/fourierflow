import pickle
from pathlib import Path
from typing import List, Optional

import debugpy
import hydra
import jax
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typer import Argument, Typer

from fourierflow.utils import import_string

from scipy.spatial import Delaunay
import gpytoolbox as gpt

from tqdm import tqdm

app = Typer()


@app.callback(invoke_without_command=True)
def main(config_path: Path,
         overrides: Optional[List[str]] = Argument(None),
         trial: int = 0,
         map_location: Optional[str] = None,
         debug: bool = False):
    """Test a Pytorch Lightning experiment."""
    config_dir = config_path.parent
    config_name = config_path.stem
    hydra.initialize(config_path=str(Path('../..') / config_dir), version_base='1.2')
    #hydra.initialize(config_path=Path('../..') /
                     #config_dir, version_base='1.2')
    config = hydra.compose(config_name, overrides=overrides)
    OmegaConf.set_struct(config, False)

    # This debug mode is for those who use VS Code's internal debugger.
    if debug:
        debugpy.listen(5678)
        debugpy.wait_for_client()
        # debugger doesn't play well with multiple processes.
        config.builder.num_workers = 0
        jax.config.update('jax_disable_jit', True)

    # Strange bug: We need to check if cuda is availabe first; otherwise,
    # sometimes lightning's CUDAAccelerator.is_available() returns false :-/
    torch.cuda.is_available()

    chkpt_dir = Path(config_dir) / 'checkpoints'
    paths = list(chkpt_dir.glob(f'trial-{trial}-*/epoch*.ckpt'))
    assert len(paths) == 1
    checkpoint_path = paths[0]
    config.trial = trial
    config.wandb.name = f"{config.wandb.group}/{trial}"

    # Set seed for reproducibility.
    rs = np.random.RandomState(7231 + trial)
    seed = config.get('seed', rs.randint(1000, 1000000))
    pl.seed_everything(seed, workers=True)
    config.seed = seed
    if 'seed' in config.trainer:
        config.trainer.seed = seed

    builder = instantiate(config.builder)
    routine = instantiate(config.routine)
    routine.load_lightning_model_state(str(checkpoint_path), map_location)

    # Start the main testing pipeline.
    Trainer = import_string(config.trainer.pop(
        '_target_', 'pytorch_lightning.Trainer'))

    trainer = Trainer(logger=False, enable_checkpointing=False,
                          **OmegaConf.to_container(config.trainer))
    trainer.test(routine, datamodule=builder)

    routine = routine.cuda()
    loader = builder.test_dataloader()
    with torch.no_grad():
        for batch in loader:
            rq = []
            rq_y = []
            try:
                x = batch["x"].to(next(routine.model.parameters()).device)
                # print x shape 
                print(f"Input shape: {x.shape}")
                pred = routine.model.forward(x).cpu().numpy()
            except:
                x = batch["xy"].to(next(routine.model.parameters()).device)
                print(f"Input shape: {x.shape}")
                pred = routine.model.forward(x).cpu().numpy()
            # print pred shape
            print(f"Output shape: {pred.shape}")
            out_path = Path(config_dir) / 'sample.pkl'
            # for each element in batch 
            for i in tqdm(range(x.shape[0]), desc="Computing Rayleigh quotients"):
                if pred.ndim == 4:
                    V = pred[i, :, :, 0]
                    x_coords = x[i, :, :, 0].cpu().numpy().ravel()
                    y_coords = x[i, :, :, 1].cpu().numpy().ravel()
                    points = np.stack([x_coords, y_coords], axis=1)
                    tri = Delaunay(points)
                    F = tri.simplices.astype(np.int64)
                    L = gpt.cotangent_laplacian(points, F)
                    numerator = V.reshape(-1).T @ L @ V.reshape(-1)
                    denominator = np.sum(V.reshape(-1) * V.reshape(-1))  
                    rq.append(numerator / denominator)
                    y = batch["y"]
                    V_gt = y[i, :, :, 0].cpu().numpy()
                    numerator_gt = V_gt.reshape(-1).T @ L @ V_gt.reshape(-1)
                    denominator_gt = np.sum(V_gt.reshape(-1) * V_gt.reshape(-1))
                    rq_gt = numerator_gt / denominator_gt
                    rq_y.append(rq_gt)
                elif pred.ndim == 3:
                    V = pred[i, :, 0]
                    points = x[i, :, :2]
                    points_unique, inverse_indices = np.unique(points.cpu().numpy(), axis=0, return_inverse=True)

                    n_points = len(points)
                    x_unique = np.unique(points[:, 0].cpu().numpy())
                    y_unique = np.unique(points[:, 1].cpu().numpy())
                    nx, ny = len(x_unique), len(y_unique)
                    
                    F = []
                    for i in range(ny - 1):
                        for j in range(nx - 1):
                            # Two triangles per grid cell
                            v0 = i * nx + j
                            v1 = i * nx + (j + 1)
                            v2 = (i + 1) * nx + j
                            v3 = (i + 1) * nx + (j + 1)
                            
                            F.append([v0, v1, v2])
                            F.append([v1, v3, v2])
                    
                    F = np.array(F, dtype=np.int64)
                    print(f"Created {len(F)} triangles from grid")

                    L = gpt.cotangent_laplacian(points.cpu().numpy(), F)
                    numerator = V.reshape(-1).T @ L @ V.reshape(-1)
                    denominator = np.sum(V.reshape(-1) * V.reshape(-1))  
                    #numerator_trace = np.trace(numerator)
                    #rq.append(numerator_trace / denominator)
                    rq.append(numerator / denominator)
                    y = batch["sigma"]
                    V_gt = y[i, :, 0].cpu().numpy()
                    numerator_gt = V_gt.reshape(-1).T @ L @ V_gt.reshape(-1)
                    denominator_gt = np.sum(V_gt.reshape(-1) * V_gt.reshape(-1))
                    rq_gt = numerator_gt / (denominator_gt + 1e-8)
                    print(f"Sample {i}: Predicted RQ = {rq[-1]}, Ground Truth RQ = {rq_gt}")
                    rq_y.append(rq_gt)
                else:
                    raise ValueError(f"Unexpected pred dimensions: {pred.ndim}. Expected 3D or 4D.")


                try: 
                    y = batch["y"]          # <- if this KeyError’s, try batch["u"] / batch["target"]
                except:
                    y = batch["sigma"]
                y = y.cpu().numpy()

                V_gt = y[i, :, :]    
                nx, ny = V_gt.shape

            print(f"Rayleigh quotients mean +/- std: {np.mean(rq)} +/- {np.std(rq)}")
            print(f"Ground truth Rayleigh quotients mean +/- std: {np.mean(rq_gt)} +/- {np.std(rq_gt)}")

            relative_rq_error = np.array(rq) - np.array(rq_y) / (np.array(rq_y) + 1e-8)

            print(f"Relative Rayleigh quotient error mean +/- std: {np.mean(relative_rq_error)} +/- {np.std(relative_rq_error)}")
                
            with open(out_path, 'wb') as f:
                pickle.dump([batch, pred], f)
            break

    print(f"Sampled data saved to {out_path}")


if __name__ == "__main__":
    app()
