from typing import List
import gc

import numpy as np
import torch
from PIL import Image
from addict import Dict

from .dtypes import ViewPrediction, Prediction


class Da3SaladSplit:
    def __init__(self, vpr_repo: str, da3_config: str='giant', device: str='cuda') -> None:
        self.device = device
        if not da3_config in {'giant', 'large', 'base', 'small'}:
            raise RuntimeError(f"Config {da3_config} not recognized")
        self.model = torch.hub.load(
            'L4rralde/Visual_Place_Recognition',
            f'da3_salad_{da3_config}',
            vpr_repo
        )
        self.model = self.model.eval().to(self.device)

    @property
    def backbone(self) -> torch.nn.Module:
        return self.model.backbone

    def preprocess_images(self, pil_img_list: List[Image.Image]) -> torch.Tensor:
        return self.backbone._preprocess_inputs(pil_img_list)

    def views_encoding(self, img_list: List[Image.Image], **kwargs) -> Dict[str, torch.Tensor]:
        imgs_cpu = self.preprocess_images(img_list)
        imgs = imgs_cpu.to(self.device, non_blocking=True)[None].float()

        feat_layer = self.backbone.dino.alt_start - 1
        feat_layers = [feat_layer]
        
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=autocast_dtype):
                batch_shape = imgs.shape
                x, aux_outputs = self.backbone._dino_attend(
                    imgs,
                    batch_shape,
                    feat_layers,
                    cam_token=None,
                    **kwargs
                )
                aux_feats, cls_token = self.backbone._aux_layers_feats(aux_outputs)
                output = Dict()
                H, W = imgs.shape[-2], imgs.shape[-1]
                output.aux = self.backbone._extract_auxiliary_features(aux_feats, feat_layers, H, W)
                output.aux_cls = self.backbone._extract_cls_token(cls_token, feat_layers)
            feats, cls = self.backbone._format_output_for_salad(output, feat_layer)
            descriptor = self.model.aggregator((feats, cls))

        #torch.cuda.empty_cache()

        return ViewPrediction(
            imgs_cpu,
            x,
            descriptor.cpu()
        )

    def sequence_encoding(self, view_preds: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        imgs_cpu = view_preds.images
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=autocast_dtype):
                outputs, _ = self.backbone._alt_attend(
                    view_preds.patch_tokens,
                    [1, *imgs_cpu.shape],
                    n=self.backbone.da3.model.backbone.out_layers,
                    **kwargs
                )
                feats = self.backbone._alt_attend_feats(outputs)
        output = Dict()
        output['images'] = imgs_cpu
        output['latent_tokens'] = tuple(
            (f, cam)
            for f, cam in feats
        )
        #torch.cuda.empty_cache()
        return output

    def heads_prediction(self, seq_preds: Dict[str, torch.Tensor], **kwargs) -> Dict[str, np.ndarray]:
        imgs_cpu = seq_preds.images

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=autocast_dtype):
                feats = tuple(
                    (f, cam)
                    for f, cam in seq_preds.latent_tokens
                )
                imgs = imgs_cpu.to(self.device, non_blocking=True)[None].float()
                output = self.backbone._heads_forward(imgs, feats, **kwargs)
            
        output = self.backbone.da3._add_processed_images(output, imgs_cpu)
        for k, v in output.items():
            if not isinstance(v, torch.Tensor):
                continue
            output[k] = v.squeeze(0).cpu().numpy()
            #del v
            #gc.collect()
        #torch.cuda.empty_cache()

        return Prediction(
            output['depth'],
            output['depth_conf'],
            output['extrinsics'],
            output['intrinsics'],
            output['processed_images']
        )

    def chunk_prediction(self, view_preds: Dict[str, torch.Tensor], **kwargs) -> Dict[str, np.ndarray]:
        imgs_cpu = view_preds.images
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=autocast_dtype):
                outputs, _ = self.backbone._alt_attend(
                    view_preds.patch_tokens,
                    [1, *imgs_cpu.shape],
                    n=self.backbone.da3.model.backbone.out_layers,
                    **kwargs
                )
                feats = self.backbone._alt_attend_feats(outputs)
                imgs = imgs_cpu.to(self.device, non_blocking=True)[None].float()
                output = self.backbone._heads_forward(imgs, feats, **kwargs)

        output = self.backbone.da3._add_processed_images(output, imgs_cpu)
        for k, v in output.items():
            if not isinstance(v, torch.Tensor):
                continue
            output[k] = v.squeeze(0).cpu().numpy()
            #del v
            #gc.collect()
        #torch.cuda.empty_cache()

        return Prediction(
            output['depth'],
            output['depth_conf'],
            output['extrinsics'],
            output['intrinsics'],
            output['processed_images']
        )
