from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as TV
import importlib
import sys


class DINOv3Encoder:

    def __init__(
        self,
        repo_or_dir="checkpoints/dinov3",
        model_name="dinov3_vitb16",
        weights=None,
        device="cuda",
        image_size=224,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")

        self.transform = TV.Compose([
            TV.Resize((image_size, image_size)),
            TV.ToTensor(),
            TV.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        repo_dir = Path(repo_or_dir).expanduser()
        if not repo_dir.is_absolute():
            repo_dir = (Path(__file__).resolve().parents[2] / repo_dir).resolve()

        if not repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repo not found: {repo_dir}")

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        backbones = importlib.import_module("dinov3.hub.backbones")

        if not hasattr(backbones, model_name):
            raise ValueError(f"Unknown model: {model_name}")

        self.model = getattr(backbones, model_name)(pretrained=False)
        self.model_name = model_name

        if weights is None:
            raise ValueError("missing weights for DINOv3")

        state_dict = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.embed_dim = getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features", None)
        if self.embed_dim is None:
            known_dims = {
                "dinov3_vitb16": 768,
                "dinov3_vitl16": 1024,
                "dinov3_vith16plus": 1280,
            }
            self.embed_dim = known_dims.get(model_name, 0)

    def encode_batch(self, crops, batch_size=32):
        if not crops:
            return np.zeros((0, int(self.embed_dim)), dtype=np.float32)

        out = []

        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch_crops = crops[i:i + batch_size]
                x = torch.stack([self.transform(c) for c in batch_crops]).to(self.device)

                z = self.model(x)
                if z.ndim == 3:
                    z = z[:, 0]

                z = torch.nn.functional.normalize(z, dim=-1)

                out.append(z.cpu())

        return torch.cat(out, dim=0).numpy()
