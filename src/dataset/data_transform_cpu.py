'''
@Project DHGA
@File    data_transform.py
@IDE     PyCharm
@Author  WXL
@Date    2026/4/26
'''

import math
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

PART_HEATMAP_PARTS = 6
KEYPOINT_PART_GROUPS = (
    (0, 1, 5, 9, 13, 17),
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
KEYPOINT_PART_BONES = (
    ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 5), (5, 9), (9, 13), (13, 17)),
    ((1, 2), (2, 3), (3, 4)),
    ((5, 6), (6, 7), (7, 8)),
    ((9, 10), (10, 11), (11, 12)),
    ((13, 14), (14, 15), (15, 16)),
    ((17, 18), (18, 19), (19, 20)),
)
SPATIAL_MODALITIES = ('color_hand', 'hand_mask_RMBG', 'hand_mask', 'mask', 'color_background', 'color', 'depth_hand', 'depth_hand_datV2L')
COLOR_MODALITIES = {'color_hand', 'color_background', 'color'}
MASK_MODALITIES = {'hand_mask_RMBG', 'mask','hand_mask'}
DEPTH_MODALITIES = {'depth_hand', 'depth_hand_datV2L'}


def _resolve_part_heatmap_groups(include_whole_hand=False):
    groups = list(KEYPOINT_PART_GROUPS)
    if include_whole_hand:
        groups.append(tuple(range(21)))
    return tuple(groups)


def _resolve_part_heatmap_bones(include_whole_hand=False):
    bones = list(KEYPOINT_PART_BONES)
    if include_whole_hand:
        whole_hand_bones = []
        for part_bones in KEYPOINT_PART_BONES:
            whole_hand_bones.extend(part_bones)
        bones.append(tuple(whole_hand_bones))
    return tuple(bones)


def _get_resample_mode(name):
    if hasattr(Image, 'Resampling'):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


class VideoCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for transform in self.transforms:
            clip = transform(clip)
        return clip


class VideoResize:
    def __init__(self, size, interpolation=None):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.interpolation = interpolation or _get_resample_mode('BILINEAR')

    def __call__(self, clip):
        height, width = self.size
        return [
            frame.resize((width, height), self.interpolation)
            for frame in clip
        ]


class TensorColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def _get_factor(self, value):
        if isinstance(value, (tuple, list)):
            return random.uniform(float(value[0]), float(value[1]))
        value = float(value)
        return random.uniform(max(0.0, 1.0 - value), 1.0 + value)

    def _adjust_brightness(self, tensor, factor):
        return tensor * factor

    def _adjust_contrast(self, tensor, factor):
        mean = tensor.mean(dim=(0, 2, 3), keepdim=True)
        return (tensor - mean) * factor + mean

    def _adjust_saturation(self, tensor, factor):
        if tensor.shape[0] != 3:
            return tensor
        gray = (
            tensor[0:1] * 0.2989
            + tensor[1:2] * 0.5870
            + tensor[2:3] * 0.1140
        )
        return (tensor - gray) * factor + gray

    def __call__(self, tensor):
        brightness = self._get_factor(self.brightness)
        contrast = self._get_factor(self.contrast)
        saturation = self._get_factor(self.saturation)

        transforms = [
            lambda x: self._adjust_brightness(x, brightness),
            lambda x: self._adjust_contrast(x, contrast),
            lambda x: self._adjust_saturation(x, saturation),
        ]
        random.shuffle(transforms)

        for transform in transforms:
            tensor = transform(tensor)
        return tensor.clamp(0.0, 1.0)


class ClipToTensor:
    def __call__(self, clip):
        frame_arrays = []
        for frame in clip:
            frame_np = np.array(frame, copy=True)
            if frame_np.ndim == 2:
                frame_np = frame_np[..., None]
            frame_arrays.append(frame_np)
        clip_np = np.stack(frame_arrays, axis=0)
        clip_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).contiguous().float()
        if clip_tensor.max() > 1.0:
            clip_tensor = clip_tensor / 255.0
        return clip_tensor  # (C, T, H, W)


class VideoNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1)

    def __call__(self, tensor):
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        return (tensor - mean) / std


class SafeRandomRotation:
    def __init__(self, degrees):
        if isinstance(degrees, (tuple, list)):
            self.min_degrees = float(degrees[0])
            self.max_degrees = float(degrees[1])
        else:
            degrees = float(degrees)
            self.min_degrees = -degrees
            self.max_degrees = degrees

    def sample_angle(self):
        return random.uniform(self.min_degrees, self.max_degrees)

    def _get_reflect_padding(self, width, height):
        # Pad to the frame diagonal so rotation will not expose black corners.
        diagonal = int(math.ceil(math.sqrt(width * width + height * height)))
        pad_w = max(0, diagonal - width) + 4
        pad_h = max(0, diagonal - height) + 4

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        return pad_left, pad_right, pad_top, pad_bottom

    def _reflect_pad_frame(self, frame, pad_left, pad_right, pad_top, pad_bottom):
        frame_np = np.asarray(frame)
        if frame_np.ndim == 2:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        else:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        padded_np = np.pad(
            frame_np,
            pad_width,
            mode='reflect'
        )
        return Image.fromarray(padded_np)

    def transform_keypoints(self, keypoints, frame_size, angle):
        if keypoints is None:
            return None

        width, height = frame_size
        width = max(int(width), 1)
        height = max(int(height), 1)
        pad_left, pad_right, pad_top, pad_bottom = self._get_reflect_padding(width, height)
        padded_width = width + pad_left + pad_right
        padded_height = height + pad_top + pad_bottom
        crop_left = (padded_width - width) // 2
        crop_top = (padded_height - height) // 2

        points = keypoints.detach().clone().float()
        points[..., 0] = points[..., 0] * float(width) + float(pad_left)
        points[..., 1] = points[..., 1] * float(height) + float(pad_top)

        cx = float(padded_width) / 2.0
        cy = float(padded_height) / 2.0
        radians = math.radians(float(angle))
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)

        x = points[..., 0] - cx
        y = points[..., 1] - cy
        rotated_x = cx + cos_a * x + sin_a * y
        rotated_y = cy - sin_a * x + cos_a * y

        points[..., 0] = (rotated_x - float(crop_left)) / float(width)
        points[..., 1] = (rotated_y - float(crop_top)) / float(height)
        return points

    def __call__(self, clip, angle=None):
        if not clip:
            return clip

        if angle is None:
            angle = self.sample_angle()
        width, height = clip[0].size
        pad_left, pad_right, pad_top, pad_bottom = self._get_reflect_padding(width, height)

        rotated_clip = []
        for frame in clip:
            padded_frame = self._reflect_pad_frame(frame, pad_left, pad_right, pad_top, pad_bottom)
            rotated_frame = padded_frame.rotate(angle, resample=_get_resample_mode('BILINEAR'))

            crop_left = (rotated_frame.width - width) // 2
            crop_top = (rotated_frame.height - height) // 2
            cropped_frame = rotated_frame.crop((crop_left, crop_top, crop_left + width, crop_top + height))
            rotated_clip.append(cropped_frame)

        return rotated_clip


train_rotation_transform = SafeRandomRotation(15)
train_color_transform = VideoCompose([
    ClipToTensor(),
    TensorColorJitter(0.3, 0.3, 0.3),
    VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


eval_color_transform = VideoCompose([
    ClipToTensor(),
    VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


mask_transform = ClipToTensor()


def color_clip_to_uint8(clip):
    """Stack decoded RGB frames without running CPU-side augmentation."""
    frame_arrays = [np.asarray(frame, dtype=np.uint8) for frame in clip]
    clip_np = np.stack(frame_arrays, axis=0)
    return torch.from_numpy(clip_np).permute(0, 3, 1, 2).contiguous()


def depth_transform(clip, normalization='sequence'):
    frame_arrays = []
    for frame in clip:
        frame_np = np.array(frame, dtype=np.float32, copy=True)
        if frame_np.ndim == 3:
            frame_np = frame_np[..., 0]
        frame_arrays.append(frame_np)
    clip_tensor = torch.from_numpy(np.stack(frame_arrays, axis=0)).unsqueeze(1)  # (T, 1, H, W)
    if normalization == 'pre_normalized':
        return clip_tensor.clamp(0.0, 1.0).contiguous()
    if normalization != 'sequence':
        raise ValueError(f'Unsupported depth normalization: {normalization}')
    lower = clip_tensor.amin()
    upper = clip_tensor.amax()
    return ((clip_tensor - lower) / (upper - lower).clamp(min=1e-6)).contiguous()


def _split_transformed_clips(transformed_clips, modality_lengths):
    split_clips = {}
    start = 0
    for modality, clip_len in modality_lengths:
        end = start + clip_len
        split_clips[modality] = transformed_clips[start:end]
        start = end
    return split_clips


def _make_point_heatmap(points_norm, xx, yy, output_size, sigma):
    frame_count = points_norm.shape[0]
    point_valid = (
        (points_norm[..., 0] >= 0.0)
        & (points_norm[..., 0] <= 1.0)
        & (points_norm[..., 1] >= 0.0)
        & (points_norm[..., 1] <= 1.0)
    )
    points = points_norm.clamp(0.0, 1.0) * float(output_size)
    dx = xx.view(1, 1, output_size, output_size) - points[:, :, 0].view(frame_count, -1, 1, 1)
    dy = yy.view(1, 1, output_size, output_size) - points[:, :, 1].view(frame_count, -1, 1, 1)
    point_heatmaps = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    point_heatmaps = point_heatmaps * point_valid.to(dtype=point_heatmaps.dtype).view(frame_count, -1, 1, 1)
    return point_heatmaps.amax(dim=1)


def _make_bone_heatmap(keypoints, bone_pairs, xx, yy, output_size, sigma):
    frame_count = keypoints.shape[0]
    if not bone_pairs:
        return torch.zeros(frame_count, output_size, output_size, dtype=torch.float32, device=keypoints.device)

    starts = torch.tensor([pair[0] for pair in bone_pairs], dtype=torch.long, device=keypoints.device)
    ends = torch.tensor([pair[1] for pair in bone_pairs], dtype=torch.long, device=keypoints.device)
    start_norm = keypoints[:, starts, :]
    end_norm = keypoints[:, ends, :]
    start_valid = (
        (start_norm[..., 0] >= 0.0)
        & (start_norm[..., 0] <= 1.0)
        & (start_norm[..., 1] >= 0.0)
        & (start_norm[..., 1] <= 1.0)
    )
    end_valid = (
        (end_norm[..., 0] >= 0.0)
        & (end_norm[..., 0] <= 1.0)
        & (end_norm[..., 1] >= 0.0)
        & (end_norm[..., 1] <= 1.0)
    )
    bone_valid = start_valid & end_valid

    start = start_norm.clamp(0.0, 1.0) * float(output_size)
    end = end_norm.clamp(0.0, 1.0) * float(output_size)
    x = xx.view(1, 1, output_size, output_size)
    y = yy.view(1, 1, output_size, output_size)
    start_x = start[:, :, 0].view(frame_count, -1, 1, 1)
    start_y = start[:, :, 1].view(frame_count, -1, 1, 1)
    vx = (end[:, :, 0] - start[:, :, 0]).view(frame_count, -1, 1, 1)
    vy = (end[:, :, 1] - start[:, :, 1]).view(frame_count, -1, 1, 1)

    length_sq = (vx * vx + vy * vy).clamp(min=1e-6)
    t = (((x - start_x) * vx + (y - start_y) * vy) / length_sq).clamp(0.0, 1.0)
    proj_x = start_x + t * vx
    proj_y = start_y + t * vy
    dist_sq = (x - proj_x) * (x - proj_x) + (y - proj_y) * (y - proj_y)
    bone_heatmaps = torch.exp(-dist_sq / (2.0 * sigma * sigma))
    bone_heatmaps = bone_heatmaps * bone_valid.to(dtype=bone_heatmaps.dtype).view(frame_count, -1, 1, 1)
    return bone_heatmaps.amax(dim=1)


def _build_part_heatmap_tensor(
    keypoints,
    keypoints_valid,
    output_size,
    sigma_ratio,
    include_whole_hand=False,
    use_bones=False,
    bone_sigma_ratio=None,
):
    keypoints = keypoints.detach().float()
    frame_count = keypoints.shape[0]
    output_size = max(int(output_size), 1)
    sigma = max(float(sigma_ratio) * float(output_size), 1e-6)
    bone_sigma_ratio = float(bone_sigma_ratio) if bone_sigma_ratio is not None else float(sigma_ratio) * 0.75
    bone_sigma = max(float(bone_sigma_ratio) * float(output_size), 1e-6)
    part_groups = _resolve_part_heatmap_groups(include_whole_hand)
    part_bones = _resolve_part_heatmap_bones(include_whole_hand)

    x_coords = torch.arange(output_size, dtype=torch.float32, device=keypoints.device) + 0.5
    y_coords = torch.arange(output_size, dtype=torch.float32, device=keypoints.device) + 0.5
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    heatmaps = torch.zeros(
        frame_count,
        len(part_groups),
        output_size,
        output_size,
        dtype=torch.float32,
        device=keypoints.device,
    )

    valid = keypoints_valid.detach().bool().view(frame_count, 1, 1) if keypoints_valid is not None else None
    for part_idx, point_indices in enumerate(part_groups):
        points_norm = keypoints[:, list(point_indices), :]
        part_heatmap = _make_point_heatmap(points_norm, xx, yy, output_size, sigma)
        if use_bones:
            bone_heatmap = _make_bone_heatmap(keypoints, part_bones[part_idx], xx, yy, output_size, bone_sigma)
            part_heatmap = torch.maximum(part_heatmap, bone_heatmap)
        part_heatmap = part_heatmap / part_heatmap.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        if valid is not None:
            part_heatmap = part_heatmap * valid.to(dtype=part_heatmap.dtype)
        heatmaps[:, part_idx] = part_heatmap
    return heatmaps.contiguous()


def _transform_keypoints(
    sample,
    phase,
    rotation_angle,
    part_heatmap_output_size,
    part_heatmap_sigma,
    part_heatmap_include_whole_hand=False,
    part_heatmap_use_bones=False,
    part_heatmap_bone_sigma=None,
    build_part_heatmap=True,
):
    keypoints = sample.get('keypoints', None)
    if keypoints is None:
        return

    keypoints_valid = sample.get('keypoints_valid', None)
    if keypoints_valid is None:
        keypoints_valid = torch.ones(keypoints.shape[0], dtype=torch.bool)

    frame_size = sample.get('_keypoint_frame_size', None)
    transformed_keypoints = keypoints.detach().clone().float()
    if phase == 'train' and rotation_angle is not None and frame_size is not None:
        transformed_keypoints = train_rotation_transform.transform_keypoints(
            transformed_keypoints,
            frame_size,
            rotation_angle,
        )

    # Keep coordinate supervision aligned with the transformed RGB/depth clip.
    # The keypoint model consumes these coordinates directly for its heatmap
    # target and spatial normalization; retaining the pre-rotation values
    # would supervise it with a different geometry than the input image.
    sample['keypoints'] = transformed_keypoints

    if not build_part_heatmap:
        return

    output_size = int(part_heatmap_output_size) if part_heatmap_output_size is not None else 224
    sample['part_heatmap'] = _build_part_heatmap_tensor(
        transformed_keypoints,
        keypoints_valid,
        output_size,
        part_heatmap_sigma,
        include_whole_hand=part_heatmap_include_whole_hand,
        use_bones=part_heatmap_use_bones,
        bone_sigma_ratio=part_heatmap_bone_sigma,
    )


def image_transform(
    sample,
    phase,
    part_heatmap_output_size=None,
    part_heatmap_sigma=0.12,
    part_heatmap_include_whole_hand=False,
    part_heatmap_use_bones=False,
    part_heatmap_bone_sigma=None,
    enable_color_jitter=True,
    depth_normalization='sequence',
    image_size=224,
    build_part_heatmap=True,
):
    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError(f'image_size must be positive, got {image_size}')
    clip_list = []
    modality_lengths = []

    for modality in SPATIAL_MODALITIES:
        if modality in sample:
            clip_list.extend(sample[modality])
            modality_lengths.append((modality, len(sample[modality])))

    if not clip_list:
        _transform_keypoints(
            sample,
            phase,
            None,
            part_heatmap_output_size,
            part_heatmap_sigma,
            part_heatmap_include_whole_hand=part_heatmap_include_whole_hand,
            part_heatmap_use_bones=part_heatmap_use_bones,
            part_heatmap_bone_sigma=part_heatmap_bone_sigma,
            build_part_heatmap=build_part_heatmap,
        )
        return sample

    rotation_angle = None
    if phase == 'train':
        rotation_angle = train_rotation_transform.sample_angle()
        transformed_clips = train_rotation_transform(clip_list, angle=rotation_angle)
        transformed_clips = VideoResize((image_size, image_size))(transformed_clips)
    else:
        transformed_clips = VideoResize((image_size, image_size))(clip_list)

    split_clips = _split_transformed_clips(transformed_clips, modality_lengths)

    for modality, clip in split_clips.items():
        if modality in COLOR_MODALITIES:
            if phase == 'train' and enable_color_jitter:
                frames_tensor = train_color_transform(clip)  # (C, T, H, W)
            else:
                frames_tensor = eval_color_transform(clip)  # (C, T, H, W)
        elif modality in DEPTH_MODALITIES:
            sample[modality] = depth_transform(clip, normalization=depth_normalization)
            continue
        elif modality in MASK_MODALITIES:
            frames_tensor = mask_transform(clip)  # (C, T, H, W)
            frames_tensor = (frames_tensor > 0.5).float()
        else:
            continue

        sample[modality] = frames_tensor.permute(1, 0, 2, 3).contiguous()  # (T, C, H, W)

    _transform_keypoints(
        sample,
        phase,
        rotation_angle,
        part_heatmap_output_size,
        part_heatmap_sigma,
        part_heatmap_include_whole_hand=part_heatmap_include_whole_hand,
        part_heatmap_use_bones=part_heatmap_use_bones,
        part_heatmap_bone_sigma=part_heatmap_bone_sigma,
        build_part_heatmap=build_part_heatmap,
    )
    return sample


def depth_clip_to_float(clip):
    frames = []
    for frame in clip:
        pixels = np.asarray(frame, dtype=np.float32)
        if pixels.ndim == 3:
            if not np.all(pixels == pixels[..., :1]):
                raise ValueError('Depth must be scalar, not a colorized image.')
            pixels = pixels[..., 0]
        if not np.isfinite(pixels).all():
            raise ValueError('Non-finite depth pixels encountered.')
        frames.append(pixels.copy())
    return torch.from_numpy(np.stack(frames)).unsqueeze(1)


def normalize_depth_clip(clip):
    values = depth_clip_to_float(clip)
    lower, upper = values.amin(), values.amax()
    normalized = (values - lower) / (upper - lower).clamp(min=1e-6)
    return [Image.fromarray(frame[0].numpy()) for frame in normalized]


def normalize_rid_depth_clip(clip):
    arrays = [np.asarray(frame, dtype=np.float32) for frame in clip]
    if not arrays or not all(np.isfinite(array).all() for array in arrays):
        raise ValueError('Empty or non-finite RID depth clip.')
    values = np.stack(arrays)
    lower, upper = float(values.min()), float(values.max())
    normalized = (values - lower) / max(upper - lower, 1e-6)
    return [Image.fromarray(np.rint(frame * 255).astype(np.uint8)).convert('RGB')
            for frame in normalized]
