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
                V = pred[i, :, :, 0] if pred.ndim == 4 else pred[i, :, 0]
                #print(f"V shape before Delaunay: {V.shape}")
                if V.ndim == 2:
                    x_coords = x[i, :, :, 0].ravel()
                    y_coords = x[i, :, :, 1].ravel()
                    points = np.stack([x_coords, y_coords], axis=1)
                    tri = Delaunay(points)
                    F = tri.simplices.astype(np.int64)
                    L = gpt.cotangent_laplacian(points, F)
                    numerator = V.reshape(-1).T @ L @ V.reshape(-1)
                    denominator = np.sum(V.reshape(-1) * V.reshape(-1))  
                    rq.append(numerator / denominator)
                elif V.ndim == 1:
                    points = x[i, :, :2]
                    tri = Delaunay(points)
                    F = tri.simplices.astype(np.int64)
                    L = gpt.cotangent_laplacian(points, F)
                    numerator = V.reshape(-1).T @ L @ V.reshape(-1)
                    denominator = np.sum(V.reshape(-1) * V.reshape(-1))  
                    #numerator_trace = np.trace(numerator)
                    #rq.append(numerator_trace / denominator)
                    rq.append(numerator / denominator)

                try: 
                    y = batch["y"]          # <- if this KeyError’s, try batch["u"] / batch["target"]
                except:
                    y = batch["sigma"]
                y = y.cpu().numpy()

                V_gt = y[i, :, :]    
                nx, ny = V_gt.shape

                X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
                points = np.stack([X.ravel(), Y.ravel()], axis=1)     # (nx*ny, 2)
                tri = Delaunay(points)
                F = tri.simplices.astype(np.int64)

                L = gpt.cotangent_laplacian(points, F)
                num = V_gt.reshape(-1).T @ L @ V_gt.reshape(-1)
                den = (V_gt.reshape(-1) ** 2).sum()
                rq_gt = num / den
                rq_y.append(rq_gt)
            print(f"Rayleigh quotients mean +/- std: {np.mean(rq)} +/- {np.std(rq)}")
            print(f"Ground truth Rayleigh quotients mean +/- std: {np.mean(rq_gt)} +/- {np.std(rq_gt)}")
                
            with open(out_path, 'wb') as f:
                pickle.dump([batch, pred], f)
            break

    print(f"Sampled data saved to {out_path}")


if __name__ == "__main__":
    app()
