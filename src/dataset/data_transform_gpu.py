"""Batched GPU video transforms for DHGA and RID."""

import random

import torch
import torch.nn.functional as F

from src.dataset.data_transform_cpu import _build_part_heatmap_tensor

class GPUVideoTransform(torch.nn.Module):
    """Apply the DHGA-IR spatial and color pipeline to a complete GPU batch."""

    def __init__(
        self,
        output_size=(224, 224),
        rotation_degrees=15.0,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(int(value) for value in output_size)
        self.rotation_degrees = float(rotation_degrees)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.register_buffer(
            'mean',
            torch.tensor(mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'std',
            torch.tensor(std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _sample_factor(batch_size, amount, device):
        if amount <= 0:
            return torch.ones(batch_size, 1, 1, 1, 1, device=device)
        return torch.empty(batch_size, 1, 1, 1, 1, device=device).uniform_(
            max(0.0, 1.0 - amount),
            1.0 + amount,
        )

    def _rotate_and_resize(self, video):
        batch_size, frame_count, channels, height, width = video.shape
        output_height, output_width = self.output_size
        flat_video = video.reshape(batch_size * frame_count, channels, height, width)

        if self.rotation_degrees <= 0:
            resized = F.interpolate(
                flat_video,
                size=self.output_size,
                mode='bilinear',
                align_corners=False,
            )
            return resized.view(batch_size, frame_count, channels, output_height, output_width)

        # One angle per video, shared by all of its frames, matching the CPU path.
        angles = torch.empty(batch_size, device=video.device).uniform_(
            -self.rotation_degrees,
            self.rotation_degrees,
        )
        radians = torch.deg2rad(angles)
        cosines = torch.cos(radians)
        sines = torch.sin(radians)
        theta = torch.zeros(batch_size, 2, 3, device=video.device, dtype=video.dtype)
        theta[:, 0, 0] = cosines
        theta[:, 0, 1] = sines
        theta[:, 1, 0] = -sines
        theta[:, 1, 1] = cosines
        theta = theta.repeat_interleave(frame_count, dim=0)

        grid = F.affine_grid(
            theta,
            size=(batch_size * frame_count, channels, output_height, output_width),
            align_corners=False,
        )
        transformed = F.grid_sample(
            flat_video,
            grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=False,
        )
        return transformed.view(batch_size, frame_count, channels, output_height, output_width)

    def _color_jitter(self, video):
        batch_size = video.shape[0]
        brightness = self._sample_factor(batch_size, self.brightness, video.device)
        contrast = self._sample_factor(batch_size, self.contrast, video.device)
        saturation = self._sample_factor(batch_size, self.saturation, video.device)

        # Randomize operation order per batch without introducing a GPU-to-CPU sync.
        for operation in torch.randperm(3).tolist():
            if operation == 0:
                video = video * brightness
            elif operation == 1:
                frame_mean = video.mean(dim=(2, 3, 4), keepdim=True)
                video = (video - frame_mean) * contrast + frame_mean
            else:
                gray = (
                    video[:, :, 0:1] * 0.2989
                    + video[:, :, 1:2] * 0.5870
                    + video[:, :, 2:3] * 0.1140
                )
                video = (video - gray) * saturation + gray
        return video.clamp_(0.0, 1.0)

    @torch.no_grad()
    def forward(self, video, phase, enable_color_jitter=True):
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(
                "GPUVideoTransform expects video with shape (B, T, 3, H, W), "
                f"got {tuple(video.shape)}."
            )
        if video.dtype != torch.uint8:
            raise TypeError(
                "GPUVideoTransform expects uint8 input from Dataset_DHGA_RID, "
                f"got {video.dtype}."
            )

        video = video.to(device=self.mean.device, non_blocking=True).float().div_(255.0)
        if phase == 'train':
            video = self._rotate_and_resize(video)
            if enable_color_jitter:
                video = self._color_jitter(video)
        else:
            batch_size, frame_count, channels, height, width = video.shape
            video = F.interpolate(
                video.reshape(batch_size * frame_count, channels, height, width),
                size=self.output_size,
                mode='bilinear',
                align_corners=False,
            ).view(batch_size, frame_count, channels, *self.output_size)

        return ((video - self.mean) / self.std).contiguous()


class _GPUMultimodalVideoTransform(torch.nn.Module):
    """Transform a whole video and its depth/keypoints with one angle per sample."""

    def __init__(self, output_size=224, rotation_degrees=15.0):
        super().__init__()
        self.output_size = (int(output_size), int(output_size))
        self.rotation_degrees = float(rotation_degrees)
        self.register_buffer(
            'rgb_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'rgb_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def _warp(self, video, angles, mode):
        batch, frames, channels, height, width = video.shape
        flat = video.reshape(batch, frames * channels, height, width)
        if angles is None:
            if mode == 'nearest':
                transformed = F.interpolate(flat, size=self.output_size, mode=mode)
            else:
                transformed = F.interpolate(
                    flat, size=self.output_size, mode=mode, align_corners=False,
                )
        else:
            radians = torch.deg2rad(angles)
            cosine, sine = torch.cos(radians), torch.sin(radians)
            theta = video.new_zeros(batch, 2, 3)
            # Inverse map for image sampling; aspect factors express a pixel
            # rotation in the normalized coordinates used by affine_grid.
            theta[:, 0, 0] = cosine
            theta[:, 0, 1] = -sine * (height / width)
            theta[:, 1, 0] = sine * (width / height)
            theta[:, 1, 1] = cosine
            out_height, out_width = self.output_size
            x = (torch.arange(out_width, device=video.device, dtype=video.dtype) + 0.5) * (2.0 / out_width) - 1.0
            y = (torch.arange(out_height, device=video.device, dtype=video.dtype) + 0.5) * (2.0 / out_height) - 1.0
            grid_x = theta[:, 0, 0, None, None] * x[None, None, :] + theta[:, 0, 1, None, None] * y[None, :, None]
            grid_y = theta[:, 1, 0, None, None] * x[None, None, :] + theta[:, 1, 1, None, None] * y[None, :, None]
            grid = torch.stack((grid_x, grid_y), dim=-1)
            transformed = F.grid_sample(
                flat, grid, mode=mode, padding_mode='reflection', align_corners=False,
            )
        return transformed.reshape(batch, frames, channels, *self.output_size)

    @staticmethod
    def _rotate_keypoints(points, angles, height, width):
        radians = torch.deg2rad(angles).view(-1, 1, 1, 1)
        cosine, sine = torch.cos(radians), torch.sin(radians)
        centered_x = (points[..., 0:1] - 0.5) * width
        centered_y = (points[..., 1:2] - 0.5) * height
        rotated_x = (cosine * centered_x + sine * centered_y) / width + 0.5
        rotated_y = (-sine * centered_x + cosine * centered_y) / height + 0.5
        return torch.cat((rotated_x, rotated_y), dim=-1)

    @staticmethod
    def _color_jitter(rgb):
        # One set of factors and one operation order per video, shared by all frames.
        results = []
        for video in rgb.split(1, dim=0):
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            saturation = random.uniform(0.7, 1.3)
            operations = random.sample(('brightness', 'contrast', 'saturation'), 3)
            for operation in operations:
                if operation == 'brightness':
                    video = video * brightness
                elif operation == 'contrast':
                    mean = video.mean(dim=(1, 3, 4), keepdim=True)
                    video = (video - mean) * contrast + mean
                else:
                    gray = (video[:, :, 0:1] * 0.2989
                            + video[:, :, 1:2] * 0.5870
                            + video[:, :, 2:3] * 0.1140)
                    video = (video - gray) * saturation + gray
            results.append(video.clamp(0.0, 1.0))
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def forward(self, data, phase, angles=None, enable_color_jitter=True):
        raw_rgb = data['rgb']
        if raw_rgb.ndim != 5 or raw_rgb.shape[2] != 3 or raw_rgb.dtype != torch.uint8:
            raise ValueError('GPU multimodal transform expects RGB (B,T,3,H,W) uint8.')
        batch, frames, _, height, width = raw_rgb.shape
        device = self.rgb_mean.device
        if phase == 'train':
            if angles is None:
                angles = torch.empty(batch, device=device).uniform_(
                    -self.rotation_degrees, self.rotation_degrees,
                )
            else:
                angles = torch.as_tensor(angles, device=device, dtype=torch.float32)
                if angles.shape != (batch,):
                    raise ValueError('Expected one rotation angle per video.')
        else:
            angles = None

        rgb = raw_rgb.to(device=device, non_blocking=True).float().div_(255.0)
        rgb = self._warp(rgb, angles, mode='bilinear')
        if phase == 'train' and enable_color_jitter:
            rgb = self._color_jitter(rgb)
        data['rgb'] = ((rgb - self.rgb_mean) / self.rgb_std).contiguous()

        for name, mode in (('depth', 'bilinear'), ('mask', 'nearest')):
            value = data.get(name)
            if value is None:
                continue
            if value.ndim != 5 or value.shape[:2] != (batch, frames):
                raise ValueError(f'{name} must have shape (B,T,C,H,W) with the same B,T as RGB.')
            source = value.to(device=device, dtype=torch.float32, non_blocking=True)
            if name == 'mask':
                source = source.div(255.0)
            transformed = self._warp(source, angles, mode=mode)
            if name == 'mask':
                data[name] = (transformed > 0.5).float()
            elif '_gpu_depth_pre_normalized' in data:
                data[name] = transformed.clamp(0.0, 1.0)
            else:
                lower = transformed.amin(dim=(1, 3, 4), keepdim=True)
                upper = transformed.amax(dim=(1, 3, 4), keepdim=True)
                data[name] = ((transformed - lower) / (upper - lower).clamp(min=1e-6)).contiguous()

        if 'keypoints' in data:
            points = data['keypoints'].to(device=device, dtype=torch.float32, non_blocking=True)
            if points.shape[:2] != (batch, frames) or points.shape[-1] != 2:
                raise ValueError('keypoints must have shape (B,T,K,2) matching RGB.')
            data['keypoints'] = (
                self._rotate_keypoints(points, angles, height, width)
                if angles is not None else points
            )
        return data




class GPUMultimodalVideoTransform(_GPUMultimodalVideoTransform):
    def __init__(self, output_size=224, build_part_heatmap=False,
                 part_heatmap_output_size=None, part_heatmap_sigma=0.12,
                 part_heatmap_include_whole_hand=False, part_heatmap_use_bones=False,
                 part_heatmap_bone_sigma=None):
        super().__init__(output_size=output_size)
        self.build_part_heatmap = build_part_heatmap
        self.part_heatmap_output_size = part_heatmap_output_size or output_size
        self.part_heatmap_sigma = part_heatmap_sigma
        self.part_heatmap_include_whole_hand = part_heatmap_include_whole_hand
        self.part_heatmap_use_bones = part_heatmap_use_bones
        self.part_heatmap_bone_sigma = part_heatmap_bone_sigma

    @torch.no_grad()
    def forward(self, data, phase, angles=None, enable_color_jitter=True):
        data = super().forward(data, phase, angles=angles,
                               enable_color_jitter=enable_color_jitter)
        if self.build_part_heatmap and 'keypoints' in data:
            points = data['keypoints']
            batch, frames = points.shape[:2]
            valid = data.get('keypoints_valid')
            if valid is not None:
                valid = valid.to(points.device).reshape(batch * frames)
            maps = _build_part_heatmap_tensor(
                points.reshape(batch * frames, points.shape[2], 2), valid,
                self.part_heatmap_output_size, self.part_heatmap_sigma,
                include_whole_hand=self.part_heatmap_include_whole_hand,
                use_bones=self.part_heatmap_use_bones,
                bone_sigma_ratio=self.part_heatmap_bone_sigma,
            )
            data['part_heatmap'] = maps.reshape(batch, frames, *maps.shape[1:])
        data.pop('_gpu_depth_pre_normalized', None)
        return data


__all__ = ['GPUMultimodalVideoTransform', 'GPUVideoTransform']
