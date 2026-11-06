"""Load a benchmark checkpoint."""
import torch

from cfm.model import CFMTransformer
from cfm.transform import ParamTransform


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CFMTransformer(
        n_params=ckpt["n_params"],
        n_dct=ckpt["n_dct"],
        d_model=ckpt["d_model"],
        n_obs_tokens=ckpt["n_obs_tokens"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
        time_scale=ckpt.get("time_scale", 1.0),
        conditioning=ckpt.get("conditioning", "cross_two"),
        n_enc_layers=ckpt.get("n_enc_layers", 2),
        soft_conditioning=ckpt.get("soft_conditioning", False),
        stmt_embed=ckpt.get("stmt_embed", "joint"),
        n_ctx_features=ckpt.get("n_ctx_features", 0),
    ).to(device)
    if ckpt.get("sigma_max") is not None:
        model.sigma_max = ckpt["sigma_max"]
    model.load_state_dict(ckpt["model_state_dict"])
    if ckpt.get("param_transform") is not None:
        model.param_transform = ParamTransform.from_state_dict(ckpt["param_transform"]).to(device)
    if ckpt.get("x_stats") is not None:
        mean, std = ckpt["x_stats"]
        model.x_stats = (mean.to(device), std.to(device))
    model.eval()
    return model, ckpt
