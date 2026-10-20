"""同次前向输出中自动识别身份描述子，不把 logits 或空间特征图当作身份向量。"""

from collections.abc import Mapping

import torch


def collect_identity_features(outputs, batch_size):
    """约定 feature 为最终描述子，分支使用 <branch>_feature，形状均为 B x D。

    cv_feature 是历史代码中供其他用途使用的兼容输出，不自动重复评估。
    模型必须明确输出分支描述子；框架不能从任意维度的最终向量推断其语义。
    """
    if not isinstance(outputs, Mapping) or 'feature' not in outputs:
        raise ValueError('Evaluation model output must contain a final identity descriptor named feature.')
    names = ['feature'] + sorted(
        name for name in outputs
        if name.endswith('_feature') and name != 'cv_feature' and not name.startswith('vis_')
    )
    features = {}
    for name in names:
        value = outputs[name]
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            # 其他模型可能用 *_feature 命名空间图，只自动接收二维描述子。
            if name == 'feature':
                raise ValueError('feature must be a (B,D) identity descriptor tensor.')
            continue
        if value.shape[0] != batch_size or value.shape[1] == 0:
            raise ValueError(f'{name} must have shape ({batch_size},D), got {tuple(value.shape)}.')
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'{name} must contain finite floating-point descriptors.')
        features[name] = value
    return features
