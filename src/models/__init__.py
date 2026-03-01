from .three_step_vggt_like import VggtLikeSaladSplit
from .dtypes import ViewPrediction, Prediction


def get_model(
    backbone_arch: str,
    vpr_repo: str,
    **kwagrs
) -> VggtLikeSaladSplit:
    if 'da3' in backbone_arch.lower():
        from .da3_salad import Da3SaladSplit
        _, config = backbone_arch.split('-')
        device = kwagrs.get('device', 'cuda')
        model = Da3SaladSplit(vpr_repo, config, device)
    elif 'vggt' in backbone_arch.lower():
        from .vggt_salad import VggtSaladSplit
        model = VggtSaladSplit(vpr_repo)
    else:
        raise ValueError(f"Backbone {backbone_arch} not supported")

    return model
