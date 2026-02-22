from typing import List

from PIL import Image
import torch
import numpy as np
from addict import Dict


class VggtSalad:
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

    def per_view_encoding(self, pil_img_list: List[Image.Image]) -> Dict[str, torch.Tensor]:
        #Image preprocessing and input checking
        images = self.model.backbone.preprocess_images(pil_img_list)
        images = torch.stack(images).to(self.device) #Torch tensor of shape n x C x H x W
        n, C, H, W = images.shape
        assert n == len(pil_img_list), "Something wrong happened with the number of images of the sequence"
        assert C == 3, "Expected a tensor of Images"

        #Inference pass
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=dtype):
                patch_tokens = self.model.backbone.dino_forward(images.unsqueeze(0)) # 1 x ...
                feats, cls = self.model.backbone.prepare_tokens_for_salad(patch_tokens, images.shape)
                global_descriptor = self.model.aggregator((feats, cls)) #n x d
        assert patch_tokens.size(0) == 1, "The first dimension corresponds to the scene/batch"
        patch_tokens = patch_tokens.squeeze(0)
        n_desc, d = global_descriptor.shape
        assert n_desc == n, "Sequence lenght mismatch"
        #Preparing predictions for future steps.
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        view_preds = Dict()
        view_preds['images'] = images.cpu() # n x c x h x w
        view_preds['patch_tokens'] = patch_tokens.cpu() # n x ...
        view_preds['global_descriptor'] = global_descriptor.cpu() # n x d

        torch.cuda.empty_cache()
        return view_preds

    def per_sequence_encoding(self, view_preds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=dtype):
                aggregated_tokens_list, _ = self.model.backbone.alternate_attention(
                    view_preds.images.unsqueeze(0).to(self.device),
                    view_preds.patch_tokens.unsqueeze(0).to(self.device)
                )
        seq_preds = Dict()
        seq_preds['seq_tokens_list'] = aggregated_tokens_list.squeeze(0).cpu()
        seq_preds['images'] = view_preds.images
        torch.cuda.empty_cache()

        return seq_preds

    def heads_prediction(self, seq_preds: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=dtype):
                predictions = self.model.backbone.heads_forward(
                    seq_preds.images.unsqueeze(0).to(self.device),
                    seq_preds.seq_tokens_list.unsqueeze(0).to(self.device),
                    self.model.backbone.vggt.aggregator.patch_start_idx,
                    query_points=None
                )
        extrinsic, intrinsic = self.model.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            seq_preds["images"].shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        torch.cuda.empty_cache()

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)
        return predictions

    def views_chunk_predicton(self, view_preds: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=dtype):
                images = view_preds.images.unsqueeze(0).to(self.device)
                patch_tokens = view_preds.patch_tokens.unsqueeze(0).to(self.device)
                aggregated_tokens_list, patch_start_idx = self.model.backbone.alternate_attention(
                    images,
                    patch_tokens
                )
                predictions = self.model.backbone.heads_forward(
                    images,
                    aggregated_tokens_list,
                    patch_start_idx,
                    query_points=None
                )
        extrinsic, intrinsic = self.model.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        torch.cuda.empty_cache()

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)
        return predictions

    @staticmethod
    def key_frame_selection(
        ref_descriptor: torch.Tensor,
        descriptors: torch.Tensor,
        th: float,
    ) -> List[int]:
        n, d = descriptors.shape
        _d = ref_descriptor.shape
        assert _d == d, "Dimension mismatch"

        keyframes = []
        current_ref = ref_descriptor
        for i in range(n):
            sim = (current_ref @ descriptors[i]).item()
            if sim < th:
                keyframes.append(i)
                current_ref = descriptors[i].clone()
        return keyframes

    def keyframe_filtering(
        self,
        view_preds: Dict[str, torch.Tensor],
        th_l: float,
    ) -> Dict[str, torch.Tensor]:
        descriptors = view_preds.global_descriptor
        if self.last_keyframe_descriptor is None:
            ref_descriptor = descriptors[0].clone()
        else:
            ref_descriptor = self.last_keyframe_descriptor
        keyframes = VggtSalad.key_frame_selection(
            ref_descriptor,
            descriptors,
            th_l,
        )
        if not keyframes:
            return Dict()
        filtered_view_preds = Dict()
        for k in ['global_descriptor', 'images', 'patch_tokens']:
            filtered_view_preds[k] = view_preds[k][keyframes]
        return filtered_view_preds
