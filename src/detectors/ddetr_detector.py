import sys
from pathlib import Path

import torch
from PIL import Image


class DDETRDetector:
    def __init__(self, repo_root, checkpoint_path, device="cuda"):
        self.repo_root = Path(repo_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        if not self.repo_root.exists():
            raise FileNotFoundError(f"Repo not found: {self.repo_root}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        repo_str = str(self.repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        import third_party.deformable_detr.datasets.transforms as T
        from third_party.deformable_detr.main import get_args_parser
        from third_party.deformable_detr.models import build_model

        args = get_args_parser().parse_args([])
        args.device = str(self.device)

        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.model, _, self.postprocessors = build_model(args)
        self.model.to(self.device)
        self.model.eval()

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    def predict(self, image_rgb):
        h, w = image_rgb.shape[:2]
        tensor, _ = self.transform(Image.fromarray(image_rgb), None)

        with torch.no_grad():
            outputs = self.model([tensor.to(self.device)])

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.postprocessors["bbox"](outputs, target_sizes)[0]

        return results["boxes"].cpu().numpy(), results["scores"].cpu().numpy()
