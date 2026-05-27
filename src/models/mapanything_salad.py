from typing import List, Dict
from collections import defaultdict

import torch
from PIL import Image
import numpy as np

from .dtypes import ViewPrediction, Prediction


#TODO. Is with torch.no_grad required? model is already in eval mode.

def preds_dict_list_to_dict(dict_list: List[Dict]) -> Dict:
    single_dict = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            single_dict[k].append(v)
    return {k: torch.cat(v) for k, v in single_dict.items()}


class MapAnythingSaladSplit:
    def __init__(self, vpr_repo: str, device: str='cuda'):
        assert device == 'cuda', "I think using other value will cause an exception in MapAnythingSalad"
        self.device = 'cuda'
        self.model: torch.nn.Module = torch.hub.load(
            'L4rralde/Visual_Place_Recognition',
            "mapanything_salad",
            vpr_repo
        )
        self.model = self.model.eval().to(device)

    @property
    def backbone(self) -> torch.nn.Module:
        return self.model.backbone

    def preprocess_images(self, pil_img_list: List[Image.Image]) -> torch.Tensor:
        images = self.backbone.preprocess_images(pil_img_list)
        if isinstance(images, list):
            t_imgs = torch.cat([img.unsqueeze(0) for img in images])
        elif isinstance(images, torch.Tensor):
            #t_imgs = images.squeeze(0) #Maybe we want to keep the batch size
            t_imgs = images
        return t_imgs

    def views_encoding(self, pil_img_list: List[Image.Image]) -> Dict[str, torch.Tensor]:
        #Let's use autocast to check if gets faster. It does, like 2x
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        images = self.preprocess_images(pil_img_list).to(self.device)

        # Get input shape of the images, number of views, and batch size per view
        num_views, c, height, width = images.shape
        assert num_views == len(pil_img_list)
        assert c == 3

        #To autocast or to not autocast. Do not autocast.
        with torch.no_grad():
            with torch.autocast('cuda', enabled=True, dtype=amp_dtype):
                patch_tokens = self.backbone.dino_forward(images)
                descriptor = self.model.aggregator(
                    self.backbone.prepare_tokens_for_salad(
                        patch_tokens,
                        height//self.backbone.PATCH_SIZE,
                        width//self.backbone.PATCH_SIZE
                    )
                )

        view_preds = ViewPrediction(
            images,
            patch_tokens,
            descriptor.cpu()
        )

        return view_preds

    def sequence_encoding(self, view_preds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError() #FUTURE

    def heads_prediction(self, seq_preds: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        raise NotImplementedError() #FUTURE

    def chunk_prediction(self, view_preds: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        images = view_preds.images
        patch_tokens = view_preds.patch_tokens

        num_views, c, height, width = images.shape
        img_shape = (int(height), int(width))

        all_encoder_features_across_views, all_encoder_registers_across_views = (
            self.backbone.unpack_dino_outputs(
                patch_tokens,
                height//self.backbone.PATCH_SIZE,
                width//self.backbone.PATCH_SIZE
            )
        )

        views = self.backbone.imgs_tensor_as_views(images)

        with torch.no_grad():
            #with torch.autocast("cuda", enabled=False): As long as this is not inside an enabled autocast, this is not required
            with torch.autocast('cuda', enabled=False):
                all_encoder_features_across_views = (
                    self.backbone._map_anything._encode_and_fuse_optional_geometric_inputs(
                        views, all_encoder_features_across_views
                    )
                )

            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast('cuda', enabled=True, dtype=amp_dtype):
                final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.backbone.alternate_attention(
                    all_encoder_features_across_views,
                    all_encoder_registers_across_views,
                    batch_size_per_view = 1
                )

                res = self.backbone.heads_forward(
                    all_encoder_features_across_views,
                    final_info_sharing_multi_view_feat,
                    intermediate_info_sharing_multi_view_feat,
                    num_views,
                    img_shape
                )

        #Compute intrinsics, extrinsics, mask, and more (?)
        with torch.autocast('cuda', enabled=False):
            res = self.backbone.postprocess_model_outputs_for_inference(res, views)
        preds_dict = preds_dict_list_to_dict(res)
        
        #Now map to Prediction class and we are done.
        prediction = Prediction(
            depth=preds_dict['depth_z'].squeeze(-1).cpu().numpy(),
            depth_conf=preds_dict['conf'].cpu().numpy(),
            extrinsic=torch.linalg.inv(preds_dict['camera_poses']).cpu().numpy()[:, :3], #Cam poses are cam to world. We need world to cam. Fixme we only store 3 rows per matrix
            intrinsic=preds_dict['intrinsics'].cpu().numpy(),
            images=preds_dict['img_no_norm'].permute(0, 3, 1, 2).cpu().numpy(),
            mask=preds_dict['mask'].squeeze(-1).cpu().numpy()
        )

        return prediction
