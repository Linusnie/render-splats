from plyfile import PlyData
import torch
from torch.nn.functional import normalize
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GaussianCloud:
    means: torch.Tensor
    opacities: torch.Tensor
    features: torch.Tensor
    scales: torch.Tensor
    quaternions: torch.Tensor

    @property
    def max_sh_degree(self):
        return int(np.sqrt(self.features.shape[1]) - 1)

    @property
    def n_gaussians(self):
        return len(self.means)


def load_3dgs(load_path, device="cuda"):
    gs_data = PlyData.read(load_path).elements[0]
    means = np.array(
        [
            gs_data["x"],
            gs_data["y"],
            gs_data["z"],
        ]
    ).T
    features_dc = np.array(
        [
            gs_data["f_dc_0"],
            gs_data["f_dc_1"],
            gs_data["f_dc_2"],
        ]
    ).T

    extra_f_names = [p.name for p in gs_data.properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

    features_extra = (
        np.array([gs_data[extra_f_name] for extra_f_name in extra_f_names])
        .T.reshape(len(means), 3, -1)
        .transpose(0, 2, 1)
    )

    features = np.concatenate([features_dc[:, None], features_extra], axis=1)

    scales = np.array(
        [
            gs_data["scale_0"],
            gs_data["scale_1"],
            gs_data["scale_2"],
        ]
    ).T

    quaternions = np.array(
        [
            gs_data["rot_0"],
            gs_data["rot_1"],
            gs_data["rot_2"],
            gs_data["rot_3"],
        ]
    ).T

    return GaussianCloud(
        means=torch.tensor(means, device=device),
        opacities=torch.sigmoid(torch.tensor(gs_data["opacity"], device=device)),
        features=torch.tensor(features, device=device),
        scales=torch.exp(torch.tensor(scales, device=device)),
        quaternions=normalize(torch.tensor(quaternions, device=device)),
    )


def load_splatfacto(load_path):
    checkpoint = torch.load(load_path)["pipeline"]
    s = "_model.gauss_params."
    config_path = Path(load_path).parent.parent / "config.yml"

    # string parsing of yaml file, avoiding dependency on nerfactory
    if config_path.exists():
        with open(config_path) as yaml_file:
            for line in yaml_file.readlines():
                if "scene_scale" in line:
                    scale = float(line.split(" ")[-1])
    else:
        scale = 1

    return GaussianCloud(
        means=checkpoint[s + "means"] / scale,
        opacities=torch.sigmoid(checkpoint[s + "opacities"]),
        features=torch.concatenate(
            [checkpoint[s + "features_dc"][:, None], checkpoint[s + "features_rest"]],
            axis=1,
        ),
        scales=torch.exp(checkpoint[s + "scales"]) / scale,
        quaternions=normalize(checkpoint[s + "quats"]),
    )
