from plyfile import PlyData
import torch
from torch.nn.functional import normalize
import numpy as np
from pathlib import Path
from dataclasses import dataclass

def bmv(A, b):
    return torch.einsum('ij, ...j->...i', A, b)

@dataclass
class GaussianCloud:
    means: torch.Tensor
    opacities: torch.Tensor
    features: torch.Tensor
    scales: torch.Tensor
    quaternions: torch.Tensor
    rotation: torch.Tensor

    @property
    def max_sh_degree(self):
        return int(np.sqrt(self.features.shape[1]) - 1)

    @property
    def n_gaussians(self):
        return len(self.means)

    def __getitem__(self, idx):
        return GaussianCloud(
            self.means[idx],
            self.opacities[idx],
            self.features[idx],
            self.scales[idx],
            self.quaternions[idx],
            self.rotation,
        )

    def __rmatmul__(self, T):
        R, t = T[:3, :3], T[:3, -1]
        return GaussianCloud(
            bmv(R, self.means) + t,
            self.opacities,
            self.features,
            self.scales,
            bmv(left(r_to_q(R)), self.quaternions),
            R @ self.rotation,
        )


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
        rotation=torch.eye(3, device=device),
    )


def load_splatfacto(load_path, device='cuda'):
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
        opacities=torch.sigmoid(checkpoint[s + "opacities"])[:, 0],
        features=torch.concatenate(
            [checkpoint[s + "features_dc"][:, None], checkpoint[s + "features_rest"]],
            axis=1,
        ),
        scales=torch.exp(checkpoint[s + "scales"]) / scale,
        quaternions=normalize(checkpoint[s + "quats"]),
        rotation=torch.eye(3, device=device),
    )

def homogenize(x):
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    x = torch.hstack([x, torch.ones((x.shape[0], 1))])
    x = x.reshape((*shape[:-1], shape[-1] + 1))
    return x

def copysign(a, b):
    a[torch.sign(a) != torch.sign(b)] *= -1

def safe_sqrt(a):
    return torch.sqrt(a + 1e-8)

def r_to_q(R):
    q_out = torch.zeros((*R.shape[:-2], 4)).to(R.device)
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    q_out[..., 0] = torch.sqrt(torch.maximum(1 + R00 + R11 + R22, torch.tensor(0.,))) / 2
    q_out[..., 1] = torch.sqrt(torch.maximum(1 + R00 - R11 - R22, torch.tensor(0.,))) / 2
    q_out[..., 2] = torch.sqrt(torch.maximum(1 - R00 + R11 - R22, torch.tensor(0.,))) / 2
    q_out[..., 3] = torch.sqrt(torch.maximum(1 - R00 - R11 + R22, torch.tensor(0.,))) / 2
    copysign(q_out[..., 1], R21 - R12)
    copysign(q_out[..., 2], R02 - R20)
    copysign(q_out[..., 3], R10 - R01)
    return q_out

def left(q):
    w, x, y, z = q
    L = torch.zeros((4, 4), device=q.device)
    L[0, 0], L[0, 1], L[0, 2], L[0, 3] = w, -x, -y, -z
    L[1, 0], L[1, 1], L[1, 2], L[1, 3] = x, w, -z, y
    L[2, 0], L[2, 1], L[2, 2], L[2, 3] = y, z, w, -x
    L[3, 0], L[3, 1], L[3, 2], L[3, 3] = z, -y, x, w
    return L