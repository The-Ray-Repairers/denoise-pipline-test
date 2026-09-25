import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Dict


def apply_hdr_log(tensor: torch.Tensor) -> torch.Tensor:
    """Applies log(1 + x) compression for HDR dynamic range stabilization."""
    return torch.log1p(torch.clamp(tensor, min=0.0))


def invert_hdr_log(tensor: torch.Tensor) -> torch.Tensor:
    """Inverts log(1 + x) back to linear HDR radiance."""
    return torch.expm1(torch.clamp(tensor, min=0.0))


class DenoisingDataset(Dataset):
    """
    Dataset loader for 1 SPP single-frame Monte Carlo denoising.
    
    Expected packed data dictionary or .npz containing:
      - 'noisy': [3, H, W] 1 SPP HDR Radiance
      - 'target': [3, H, W] 512+ SPP Converged Ground Truth
      - 'normal': [2 or 3, H, W] View-space / Shading normals
      - 'depth': [1, H, W] Linearized depth
      - 'roughness': [1, H, W] Material roughness
      - 'albedo': [3, H, W] (Optional) Diffuse albedo for demodulation
    """
    def __init__(
        self,
        data_dir: str,
        patch_size: Optional[int] = 256,
        is_training: bool = True,
        use_demodulation: bool = False,
        use_log_transform: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.is_training = is_training
        self.use_demodulation = use_demodulation
        self.use_log_transform = use_log_transform

        # Search for .npz, .pt, or .npy data files
        self.files = sorted(
            glob.glob(os.path.join(data_dir, "*.npz"))
            + glob.glob(os.path.join(data_dir, "*.pt"))
            + glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True)
            + glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True)
        )

        if len(self.files) == 0:
            print(f"[Warning] No dataset files found in {data_dir}. Generating on-the-fly or please run download_sample.py")

    def __len__(self) -> int:
        return len(self.files)

    def _load_file(self, path: str) -> Dict[str, torch.Tensor]:
        if path.endswith(".npz"):
            data = np.load(path)
            res = {k: torch.from_numpy(data[k]).float() for k in data.files}
        elif path.endswith(".pt"):
            res = torch.load(path, map_location="cpu")
        else:
            raise ValueError(f"Unsupported format: {path}")

        # Ensure shapes are [C, H, W]
        for k in res:
            if res[k].dim() == 2:
                res[k] = res[k].unsqueeze(0)
            elif res[k].dim() == 3 and res[k].shape[2] in [1, 2, 3]:  # [H, W, C] -> [C, H, W]
                res[k] = res[k].permute(2, 0, 1)

        return res

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._load_file(self.files[idx])

        noisy = sample["noisy"]          # [3, H, W]
        target = sample["target"]        # [3, H, W]
        normal = sample["normal"]        # [2 or 3, H, W]
        depth = sample["depth"]          # [1, H, W]
        roughness = sample["roughness"]  # [1, H, W]
        albedo = sample.get("albedo", None)

        # Random patch extraction during training
        if self.patch_size is not None and self.is_training:
            h, w = noisy.shape[1], noisy.shape[2]
            if h >= self.patch_size and w >= self.patch_size:
                top = np.random.randint(0, h - self.patch_size + 1)
                left = np.random.randint(0, w - self.patch_size + 1)
                ps = self.patch_size

                noisy = noisy[:, top:top+ps, left:left+ps]
                target = target[:, top:top+ps, left:left+ps]
                normal = normal[:, top:top+ps, left:left+ps]
                depth = depth[:, top:top+ps, left:left+ps]
                roughness = roughness[:, top:top+ps, left:left+ps]
                if albedo is not None:
                    albedo = albedo[:, top:top+ps, left:left+ps]

        # Albedo Demodulation (optional)
        if self.use_demodulation and albedo is not None:
            eps = 1e-3
            noisy = noisy / (albedo + eps)
            target = target / (albedo + eps)

        # Log compression for HDR dynamic range
        if self.use_log_transform:
            noisy = apply_hdr_log(noisy)
            target = apply_hdr_log(target)

        # Normalize depth & normal if needed
        depth = torch.clamp(depth, 0.0, 100.0) / 100.0  # Normalized depth heuristic
        if normal.shape[0] == 3:
            # Use 2D projection or first 2 view channels
            normal = normal[:2, :, :]

        # Pack input feature tensor: [3 (radiance) + 2 (normal) + 1 (depth) + 1 (roughness)] = 7 channels
        input_tensor = torch.cat([noisy, normal, depth, roughness], dim=0)

        out_dict = {
            "input": input_tensor,
            "target": target,
        }
        if albedo is not None:
            out_dict["albedo"] = albedo

        return out_dict


class SequenceDenoisingDataset(Dataset):
    """
    Dataset loader for multi-frame animated sequences for Recurrent U-Net training.
    Packs sequences of shape [T, C, H, W].
    """
    def __init__(
        self,
        data_dir: str,
        seq_length: int = 5,
        patch_size: Optional[int] = 256,
        is_training: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.is_training = is_training

        # Each file should contain a sequence array or directory of consecutive frames
        self.files = sorted(
            glob.glob(os.path.join(data_dir, "*.npz"))
            + glob.glob(os.path.join(data_dir, "*.pt"))
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx]) if self.files[idx].endswith(".npz") else torch.load(self.files[idx])
        
        # Expecting sequence shape [T, C, H, W]
        noisy_seq = torch.from_numpy(data["noisy"]).float() if isinstance(data, np.lib.npyio.NpzFile) else data["noisy"]
        target_seq = torch.from_numpy(data["target"]).float() if isinstance(data, np.lib.npyio.NpzFile) else data["target"]
        normal_seq = torch.from_numpy(data["normal"]).float() if isinstance(data, np.lib.npyio.NpzFile) else data["normal"]
        depth_seq = torch.from_numpy(data["depth"]).float() if isinstance(data, np.lib.npyio.NpzFile) else data["depth"]
        roughness_seq = torch.from_numpy(data["roughness"]).float() if isinstance(data, np.lib.npyio.NpzFile) else data["roughness"]

        t, _, h, w = noisy_seq.shape
        # Subsequence slicing
        if t > self.seq_length and self.is_training:
            start_t = np.random.randint(0, t - self.seq_length + 1)
            noisy_seq = noisy_seq[start_t:start_t+self.seq_length]
            target_seq = target_seq[start_t:start_t+self.seq_length]
            normal_seq = normal_seq[start_t:start_t+self.seq_length]
            depth_seq = depth_seq[start_t:start_t+self.seq_length]
            roughness_seq = roughness_seq[start_t:start_t+self.seq_length]

        # Patch crop
        if self.patch_size is not None and self.is_training and h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            ps = self.patch_size

            noisy_seq = noisy_seq[:, :, top:top+ps, left:left+ps]
            target_seq = target_seq[:, :, top:top+ps, left:left+ps]
            normal_seq = normal_seq[:, :, top:top+ps, left:left+ps]
            depth_seq = depth_seq[:, :, top:top+ps, left:left+ps]
            roughness_seq = roughness_seq[:, :, top:top+ps, left:left+ps]

        noisy_seq = apply_hdr_log(noisy_seq)
        target_seq = apply_hdr_log(target_seq)
        if normal_seq.shape[1] == 3:
            normal_seq = normal_seq[:, :2, :, :]

        inputs = torch.cat([noisy_seq, normal_seq, depth_seq, roughness_seq], dim=1)  # [T, 7, H, W]

        return {
            "input": inputs,
            "target": target_seq,
        }
