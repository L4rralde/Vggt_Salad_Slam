from typing import List, Dict

from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
import numpy as np
from addict import Dict

from .dtypes import ViewPrediction, Prediction


class VggtSaladSplit:
    def __init__(self, vpr_repo: str, device: str='cuda') -> None:
        assert device == 'cuda', "VGGT only works with CUDA"
        self.device = 'cuda'
        self.model: torch.nn.Module = torch.hub.load(
            'L4rralde/Visual_Place_Recognition',
            "vggt_salad",
            vpr_repo
        )
        self.model = self.model.eval().to(device)
        self.last_keyframe_descriptor: np.ndarray|None = None

    @property
    def backbone(self) -> torch.nn.Module:
        return self.model.backbone

    def preprocess_images(self, pil_img_list: List[Image.Image]) -> torch.Tensor:
        return torch.stack([
            to_tensor(self.backbone.preprocess_image(img))
            for img in pil_img_list
        ])

    def views_encoding(self, pil_img_list: List[Image.Image]) -> Dict[str, torch.Tensor]:
        #Image preprocessing and input checking
        images = self.preprocess_images(pil_img_list).to(self.device) #Torch tensor of shape n x C x H x W
        n, C, H, W = images.shape
        assert n == len(pil_img_list), "Something wrong happened with the number of images of the sequence"
        assert C == 3, "Expected a tensor of Images"
        images = images.unsqueeze(0)

        #Inference pass
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=dtype):
                patch_tokens = self.backbone.dino_forward(images) # 1 x ...
                feats, cls = self.backbone.prepare_tokens_for_salad(patch_tokens, images.shape)
                descriptor = self.model.aggregator((feats, cls)) #n x d
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        n_desc, d = descriptor.shape
        assert n_desc == n, "Sequence lenght mismatch"
        #Preparing predictions for future steps.

        view_preds = ViewPrediction(
            images.squeeze(0).cpu(),
            patch_tokens.cpu(),
            descriptor.cpu()
        )

        torch.cuda.empty_cache()
        return view_preds

    def sequence_encoding(self, view_preds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=dtype):
                aggregated_tokens_list, _ = self.backbone.alternate_attention(
                    view_preds.images.unsqueeze(0).to(self.device),
                    view_preds.patch_tokens.to(self.device)
                )
        seq_preds = Dict()
        seq_preds['seq_tokens_list'] = [t.cpu() for t in aggregated_tokens_list]
        seq_preds['images'] = view_preds.images
        torch.cuda.empty_cache()

        return seq_preds

    def heads_prediction(self, seq_preds: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=dtype):
                seq_token_list = [t.to(self.device) for t in seq_preds.seq_tokens_list]
                predictions = self.backbone.heads_forward(
                    seq_preds.images.unsqueeze(0).to(self.device),
                    seq_token_list,
                    self.backbone.vggt.aggregator.patch_start_idx,
                    query_points=None
                )
        extrinsic, intrinsic = self.backbone.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            seq_preds["images"].shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        to_keep = ['depth', 'depth_conf', 'images', 'extrinsic', 'intrinsic']
        filetered_preds = {
            k: predictions[k]
            for k in to_keep
        }

        for key, value in filetered_preds.items():
            if isinstance(value, torch.Tensor):
                filetered_preds[key] = value.cpu().numpy().squeeze(0)

        torch.cuda.empty_cache()
        return Prediction(
            filetered_preds['depth'],
            filetered_preds['depth_conf'],
            filetered_preds['extrinsic'],
            filetered_preds['intrinsic'],
            filetered_preds['images']
        )

    def chunk_prediction(self, view_preds: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=dtype):
                images = view_preds.images.unsqueeze(0).to(self.device)
                patch_tokens = view_preds.patch_tokens.to(self.device)
                aggregated_tokens_list, patch_start_idx = self.backbone.alternate_attention(
                    images,
                    patch_tokens
                )
                predictions = self.backbone.heads_forward(
                    images,
                    aggregated_tokens_list,
                    patch_start_idx,
                    query_points=None
                )
        extrinsic, intrinsic = self.backbone.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        to_keep = ['depth', 'depth_conf', 'images', 'extrinsic', 'intrinsic']
        filetered_preds = {
            k: predictions[k]
            for k in to_keep
        }

        for key, value in filetered_preds.items():
            if isinstance(value, torch.Tensor):
                filetered_preds[key] = value.cpu().numpy().squeeze(0)

        torch.cuda.empty_cache()
        return Prediction(
            filetered_preds['depth'],
            filetered_preds['depth_conf'],
            filetered_preds['extrinsic'],
            filetered_preds['intrinsic'],
            filetered_preds['images']
        )
