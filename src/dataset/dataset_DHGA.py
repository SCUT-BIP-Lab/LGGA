'''
@Project DHGA
@File    dataset_DHGA.py
@IDE     PyCharm
@Author  WXL
@Date    2026/4/26
'''

import csv
import json
import os

import numpy as np
import torch

from src.dataset.data_transform_cpu import (image_transform, color_clip_to_uint8,
                                            depth_clip_to_float, normalize_depth_clip,
                                            COLOR_MODALITIES, MASK_MODALITIES, DEPTH_MODALITIES)
from src.dataset.dataset_base import DHGABaseDataset
from src.dataset.readers import FrameReader


class Dataset_DHGA(DHGABaseDataset):
    def __init__(self, conf, phase, **kwargs):
        super().__init__(conf, phase)
        self.frame_reader = FrameReader()

        self.data_sources = self._resolve_data_sources(kwargs)
        self.use_mask = self._validate_mask_usage()
        self.use_depth = self._validate_depth_usage()
        self.use_keypoints = self.phase == 'train' and self._resolve_keypoint_usage()
        dataset_cfg = self.conf.get('dataset', {})
        self.image_size = int(dataset_cfg.get('image_size', 224))
        if self.image_size <= 0:
            raise ValueError(f'dataset.image_size must be positive, got {self.image_size}')
        self.keypoint_normalize_size = float(dataset_cfg.get('keypoint_normalize_size', 200.0))
        self.keypoint_coordinate_order = str(dataset_cfg.get('keypoint_coordinate_order', 'xy')).lower()
        if self.keypoint_coordinate_order not in {'xy', 'yx'}:
            raise ValueError('dataset.keypoint_coordinate_order must be xy or yx.')
        self.part_heatmap_sigma = float(dataset_cfg.get('part_heatmap_sigma', 0.12))
        self.part_heatmap_output_size = dataset_cfg.get('part_heatmap_output_size', None)
        self.build_part_heatmap = self._as_bool(dataset_cfg.get('build_part_heatmap', True))
        self.part_heatmap_include_whole_hand = self._as_bool(
            dataset_cfg.get('part_heatmap_include_whole_hand', False)
        )
        self.part_heatmap_use_bones = self._as_bool(dataset_cfg.get('part_heatmap_use_bones', False))
        self.part_heatmap_bone_sigma = dataset_cfg.get('part_heatmap_bone_sigma', None)
        self.contrastive_mode = self._resolve_contrastive_mode()
        self._cache_from_file(self.phase)

    def _as_bool(self, value):
        return self.as_bool(value)

    def _resolve_contrastive_mode(self):
        contrastive_cfg = self.conf.get('contrastive', {})
        if isinstance(contrastive_cfg, dict) and 'enabled' in contrastive_cfg:
            return self._as_bool(contrastive_cfg['enabled'])
        return self._as_bool(self.conf.get('dataset', {}).get('contrastive_mode', False))

    def _normalize_subsets(self, subsets):
        return self.normalize_subsets(subsets)

    def _resolve_data_sources(self, kwargs):
        input_sources = kwargs.get('data_sources', None)
        if input_sources is None:
            raise ValueError('Dataset_DHGA requires explicit data_sources in multi-source mode.')

        resolved_sources = []
        dataset_cfg = self.conf.get('dataset', {})
        default_keypoint_root = dataset_cfg.get('keypoint_root', None)
        default_frames_per_sample = self.conf['dataset'].get('frames_per_sample', None)
        for idx, source in enumerate(input_sources):
            root = source.get('root', None)
            metadata_root = source.get('metadata_root', None)
            label_file = source.get('label_file', None)
            if root is None or metadata_root is None or label_file is None:
                raise ValueError(
                    f"Each data source must provide 'root', 'metadata_root', and 'label_file'. "
                    f"Invalid source #{idx}: {source}"
                )

            label_path = label_file if os.path.isabs(label_file) else os.path.join(metadata_root, label_file)
            source_name = source.get('name', f'source_{idx}')
            subsets = self._normalize_subsets(source.get('subsets', None))
            if not subsets:
                raise ValueError(f"Each data source must provide non-empty 'subsets'. Invalid source #{idx}: {source}")

            unsupported_subsets = [
                subset for subset in subsets
                if subset not in COLOR_MODALITIES and subset not in MASK_MODALITIES and subset not in DEPTH_MODALITIES
            ]
            if unsupported_subsets:
                raise ValueError(
                    f"Unsupported subsets in source #{idx}: {unsupported_subsets}. "
                    f"Supported color: {sorted(COLOR_MODALITIES)}, mask: {sorted(MASK_MODALITIES)}, depth: {sorted(DEPTH_MODALITIES)}"
                )

            subset_overrides = source.get('subset_roots', {})
            if not isinstance(subset_overrides, dict) or set(subset_overrides) - set(subsets):
                raise ValueError(f'subset_roots must map selected modalities to directories: {source}')
            if any(not isinstance(path, str) or not path.strip() for path in subset_overrides.values()):
                raise ValueError(f'subset_roots paths must be non-empty strings: {source}')
            subset_roots = {
                subset: subset_overrides.get(subset, os.path.join(root, subset)) for subset in subsets
            }

            rgb_candidates = [subset for subset in subsets if subset in COLOR_MODALITIES]
            if len(rgb_candidates) != 1:
                raise ValueError(
                    f"Each data source must contain exactly one color modality in subsets. "
                    f"Got {rgb_candidates} from subsets={subsets} for source #{idx}."
                )
            rgb_modality = rgb_candidates[0]

            mask_candidates = [subset for subset in subsets if subset in MASK_MODALITIES]
            if len(mask_candidates) > 1:
                raise ValueError(f"Only one mask subset is supported per source. Invalid source #{idx}: {source}")
            mask_modality = mask_candidates[0] if mask_candidates else None

            depth_candidates = [subset for subset in subsets if subset in DEPTH_MODALITIES]
            if len(depth_candidates) > 1:
                raise ValueError(f"Only one depth subset is supported per source. Invalid source #{idx}: {source}")
            depth_modality = depth_candidates[0] if depth_candidates else None

            frames_per_sample = source.get('frames_per_sample', default_frames_per_sample)
            if frames_per_sample is None:
                raise ValueError(
                    f"Each data source must provide 'frames_per_sample' in source config. Invalid source #{idx}: {source}"
                )
            frames_per_sample = int(frames_per_sample)
            if frames_per_sample <= 0:
                raise ValueError(
                    f"'frames_per_sample' must be positive. Invalid source #{idx}: {source}"
                )

            resolved_sources.append({
                'source_index': idx,
                'name': source_name,
                'root': root,
                'metadata_root': metadata_root,
                'label_file': label_file,
                'label_path': label_path,
                'subsets': subsets,
                'subset_roots': subset_roots,
                'rgb_modality': rgb_modality,
                'mask_modality': mask_modality,
                'depth_modality': depth_modality,
                'keypoint_root': source.get('keypoint_root', default_keypoint_root),
                'frames_per_sample': frames_per_sample,
            })
        return resolved_sources

    def _validate_mask_usage(self):
        has_mask_flags = {source['mask_modality'] is not None for source in self.data_sources}
        if len(has_mask_flags) > 1:
            raise ValueError(
                'All data sources within the same dataset split must either all provide mask supervision or all omit it.'
            )
        return has_mask_flags.pop() if has_mask_flags else False

    def _validate_depth_usage(self):
        # Depth supervision can be available only for part of an evaluation
        # split (for example DHG-Auth has GT depth while Real-DHGA does not).
        # Keep the dataset-level flag for compatibility, but decide whether to
        # attach ``depth`` per sample below.
        return any(source['depth_modality'] is not None for source in self.data_sources)

    def _resolve_keypoint_usage(self):
        dataset_cfg = self.conf.get('dataset', {})
        return self._as_bool(dataset_cfg.get('use_keypoints', False))

    def _cache_from_file(self, phase):
        self.samples = []

        if phase == 'train': # TODO: 验证是否支持多数据集多类自动统计
            self.id_list = []
            self.gesture_list = []
            self.id_name_list = []
            self.gesture_name_list = []

            for source in self.data_sources:
                subset_roots = source['subset_roots']
                with open(source['label_path']) as fin:
                    fin_csv = csv.reader(fin)
                    for i, row in enumerate(fin_csv):
                        if i == 0:
                            continue
                        if row[2] == 'False':
                            continue

                        vid_name = row[0]
                        sample_paths = {
                            subset: os.path.join(subset_roots[subset], vid_name) for subset in source['subsets']
                        }
                        self.samples.append({
                            'vid_name': vid_name,
                            'source_name': source['name'],
                            'source_index': source['source_index'],
                            'paths': sample_paths,
                            'rgb_modality': source['rgb_modality'],
                            'mask_modality': source['mask_modality'],
                            'depth_modality': source['depth_modality'],
                            'keypoint_path': (
                                os.path.join(source['keypoint_root'], vid_name)
                                if source.get('keypoint_root', None) else None
                            ),
                            'frames_per_sample': source['frames_per_sample'],
                        })

                        id_name = vid_name.split('_')[2]
                        if id_name not in self.id_name_list:
                            self.id_name_list.append(id_name)
                        self.id_list.append(id_name)

                        gesture_name = vid_name.split('_')[3]
                        if gesture_name not in self.gesture_name_list:
                            self.gesture_name_list.append(gesture_name)
                        self.gesture_list.append(gesture_name)

            self.total_num = len(self.id_list)

            id_name_class_map = {name: i for i, name in enumerate(self.id_name_list)}
            gesture_name_class_map = {name: i for i, name in enumerate(self.gesture_name_list)}

            for i, name in enumerate(self.id_list):
                self.id_list[i] = id_name_class_map[name]

            for i, name in enumerate(self.gesture_list):
                self.gesture_list[i] = gesture_name_class_map[name]

            print(
                f"dataset {self.phase} cached, total num: {self.total_num}, "
                f"id num: {len(self.id_name_list)}, gesture num: {len(self.gesture_name_list)}"
            )

            self.conf['classes_num'] = len(self.id_name_list)
            self.conf['gesture_classes_num'] = len(self.gesture_name_list)

        else:
            for source in self.data_sources:
                subset_roots = source['subset_roots']
                vid_names = set()
                with open(source['label_path']) as fin:
                    fin_csv = csv.reader(fin)
                    for i, row in enumerate(fin_csv):
                        if i == 0:
                            continue
                        vid_names.update(row[:2])

                for vid_name in sorted(vid_names):
                    sample_paths = {
                        subset: os.path.join(subset_roots[subset], vid_name) for subset in source['subsets']
                    }
                    self.samples.append({
                        'vid_name': vid_name,
                        'source_name': source['name'],
                        'source_index': source['source_index'],
                        'paths': sample_paths,
                        'rgb_modality': source['rgb_modality'],
                        'mask_modality': source['mask_modality'],
                        'depth_modality': source['depth_modality'],
                        'keypoint_path': (
                            os.path.join(source['keypoint_root'], vid_name)
                            if source.get('keypoint_root', None) else None
                        ),
                        'frames_per_sample': source['frames_per_sample'],
                    })

            self.total_num = len(self.samples)
            print(f"dataset {self.phase} cached, total num: {self.total_num}")

    def _transform_raw_sample(self, raw_sample):
        transform_input = {}
        for modality, value in raw_sample.items():
            if isinstance(value, (list, tuple)):
                transform_input[modality] = list(value)
            else:
                transform_input[modality] = value
        return image_transform(
            transform_input,
            phase=self.phase,
            part_heatmap_output_size=self.part_heatmap_output_size,
            part_heatmap_sigma=self.part_heatmap_sigma,
            part_heatmap_include_whole_hand=self.part_heatmap_include_whole_hand,
            part_heatmap_use_bones=self.part_heatmap_use_bones,
            part_heatmap_bone_sigma=self.part_heatmap_bone_sigma,
            build_part_heatmap=self.build_part_heatmap,
            depth_normalization='pre_normalized',
            image_size=self.image_size,
        )

    def _build_single_view_sample(self, transformed_sample, sample_info):
        sample = {
            'rgb': transformed_sample[sample_info['rgb_modality']],
            'source_name': sample_info['source_name'],
        }
        if self.use_mask:
            sample['mask'] = transformed_sample[sample_info['mask_modality']]
        if self.use_depth and sample_info['depth_modality'] is not None:
            sample['depth'] = transformed_sample[sample_info['depth_modality']]
        if self.use_keypoints and 'part_heatmap' in transformed_sample:
            sample['part_heatmap'] = transformed_sample['part_heatmap']
        if self.use_keypoints:
            sample['keypoints'] = transformed_sample['keypoints']
        return sample

    def _load_subset_frames(self, subset, slice_list, sample_path, sample_info):
        return self.frame_reader.load(sample_path, slice_list, subset)

    def _keypoint_frame_name(self, slice_index, sample_info):
        """Keep the original annotation naming convention for legacy datasets."""
        return str(int(slice_index) + 1).zfill(2) + '.jpg'

    def _keypoint_frame_names(self, slice_index, sample_info):
        source = self.data_sources[sample_info['source_index']]
        modality = sample_info['rgb_modality']
        names = None
        names = self.frame_reader.names(sample_info['paths'][modality])
        legacy = self._keypoint_frame_name(slice_index, sample_info)
        if names is not None and 0 <= int(slice_index) < len(names):
            actual = names[int(slice_index)]
            return (actual,) if actual == legacy else (actual, legacy)
        return (legacy,)

    def _gpu_raw_sample(self, raw_sample, info):
        sample = {
            'rgb': color_clip_to_uint8(raw_sample[info['rgb_modality']]),
            'source_name': info['source_name'],
        }
        if info['mask_modality'] is not None:
            sample['mask'] = color_clip_to_uint8(raw_sample[info['mask_modality']])
        if info['depth_modality'] is not None:
            sample['depth'] = depth_clip_to_float(raw_sample[info['depth_modality']])
            sample['_gpu_depth_pre_normalized'] = True
        if self.use_keypoints:
            sample['keypoints'] = raw_sample['keypoints']
            sample['keypoints_valid'] = raw_sample['keypoints_valid']
        return sample

    def _add_contrastive_views(self, sample, view1, view2, sample_info):
        sample['rgb_view1'] = view1[sample_info['rgb_modality']]
        sample['rgb_view2'] = view2[sample_info['rgb_modality']]
        if self.use_mask:
            sample['mask_view1'] = view1[sample_info['mask_modality']]
            sample['mask_view2'] = view2[sample_info['mask_modality']]
        if self.use_depth and sample_info['depth_modality'] is not None:
            sample['depth_view1'] = view1[sample_info['depth_modality']]
            sample['depth_view2'] = view2[sample_info['depth_modality']]
        if self.use_keypoints and 'part_heatmap' in view1 and 'part_heatmap' in view2:
            sample['part_heatmap'] = view1['part_heatmap']
            sample['part_heatmap_view1'] = view1['part_heatmap']
            sample['part_heatmap_view2'] = view2['part_heatmap']

    def _parse_keypoint_array(self, raw_keypoints):
        keypoints = torch.as_tensor(raw_keypoints, dtype=torch.float32)
        if keypoints.numel() < 42:
            return None
        if keypoints.ndim == 1:
            keypoints = keypoints.view(-1, 2)
        elif keypoints.ndim >= 2:
            keypoints = keypoints.reshape(keypoints.shape[0], -1)
            keypoints = keypoints[:, :2]
        if keypoints.shape[0] < 21:
            return None
        keypoints = keypoints[:21, :2]
        if not torch.isfinite(keypoints).all():
            return None
        if getattr(self, 'keypoint_coordinate_order', 'xy') == 'yx':
            keypoints = keypoints[:, [1, 0]]
        limit = max(self.keypoint_normalize_size, 1e-6)
        return keypoints.clamp(0.0, limit) / limit

    def _load_keypoints(self, slice_list, sample_info):
        frame_count = len(slice_list)
        keypoints = torch.zeros(frame_count, 21, 2, dtype=torch.float32)
        valid = torch.zeros(frame_count, dtype=torch.bool)
        keypoint_path = sample_info.get('keypoint_path', None)
        if not keypoint_path:
            return keypoints, valid

        json_path = keypoint_path if str(keypoint_path).endswith('.json') else keypoint_path + '.json'
        if not os.path.exists(json_path):
            return keypoints, valid

        with open(json_path, 'r') as fin:
            payload = json.load(fin)
        frame_payload = payload.get('info', payload)

        for out_idx, slice_index in enumerate(slice_list):
            frame_info = next(
                (frame_payload[name] for name in self._keypoint_frame_names(slice_index, sample_info)
                 if name in frame_payload),
                None,
            )
            if frame_info is None:
                continue
            raw_keypoints = frame_info.get('keypoints', frame_info) if isinstance(frame_info, dict) else frame_info
            parsed = self._parse_keypoint_array(raw_keypoints)
            if parsed is None:
                continue
            keypoints[out_idx] = parsed
            valid[out_idx] = True

        return keypoints, valid

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        slice_list = self.get_slice_list(
            conf=self.conf,
            phase=self.phase,
            video_len=sample_info['frames_per_sample'],
        )
        raw_sample = {}
        video_name = sample_info['vid_name']

        for subset, sample_path in sample_info['paths'].items():
            frames = self._load_subset_frames(subset, slice_list, sample_path, sample_info)
            if subset == sample_info['depth_modality']:
                frames = normalize_depth_clip(frames)
            raw_sample[subset] = frames
        if self.use_keypoints:
            reference_frames = raw_sample[sample_info['rgb_modality']]
            reference_size = reference_frames[0].size if reference_frames else (self.image_size, self.image_size)
            keypoints, keypoints_valid = self._load_keypoints(slice_list, sample_info)
            raw_sample['keypoints'] = keypoints
            raw_sample['keypoints_valid'] = keypoints_valid
            raw_sample['_keypoint_frame_size'] = reference_size

        if self.gpu_video_transform:
            sample = self._gpu_raw_sample(raw_sample, sample_info)
            if self.phase == 'train' and self.contrastive_mode:
                for name in ('rgb', 'mask', 'depth'):
                    if name in sample:
                        sample[name + '_view1'] = sample[name].clone()
                        sample[name + '_view2'] = sample[name].clone()
        elif self.phase == 'train' and self.contrastive_mode:
            view1 = self._transform_raw_sample(raw_sample)
            view2 = self._transform_raw_sample(raw_sample)
            sample = self._build_single_view_sample(view1, sample_info)
            self._add_contrastive_views(sample, view1, view2, sample_info)
        else:
            transformed_sample = self._transform_raw_sample(raw_sample)
            sample = self._build_single_view_sample(transformed_sample, sample_info)

        if self.phase == 'train':
            sample['id_label'] = torch.tensor(self.id_list[idx]).squeeze()
            sample['gesture_label'] = torch.tensor(self.gesture_list[idx]).squeeze()
        if self.use_keypoints:
            sample['keypoints_valid'] = keypoints_valid
        sample['frame_indices'] = torch.as_tensor(np.asarray(slice_list).copy(), dtype=torch.long)
        sample['sequence_length'] = int(sample_info['frames_per_sample'])
        sample['vid_name'] = video_name

        return sample
