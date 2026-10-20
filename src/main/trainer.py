"""Original DHGA trainer, restricted to this model's train/test protocol."""
import csv
import json
import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib import tenumerate

from src.utils.util import set_random_seed
from src.dataset.data_transform_gpu import GPUMultimodalVideoTransform
from src.dataset.dataset_utils import make_dataset
from src.model.model_utils import make_model, load_pretrained_params
from src.model.loss.eer import calculate_eer
from src.main.trainer_components.evaluation_features import collect_identity_features


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    value = str(value).strip().lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off', 'none'):
        return False
    return bool(value)


class Trainer:

    def __init__(self, conf):
        self.conf = conf
        # tools
        self.logger = conf["logger"]
        self.tb_writer = conf["tb_writer"]

        # framework
        self.device = conf["device"]
        self.mode = conf["mode"]
        self.epochs = conf.get("train", {}).get("num_epochs", 0)

        self.eval_internal = conf.get("evaluate", {}).get("eval_interval", 1)
        self.model_save_dir = conf["model_save_dir"]

        self.resume = conf["resume"]

        self.global_step = 0
        self.eval_step = 0
        self.current_epoch = 0

        self.seed = int(conf.get("seed", conf.get("random_seed", 1314)))
        set_random_seed(self.seed)

        dataset_cfg = self.conf.get('dataset', {})
        self.gpu_multimodal_transform_enabled = (
            as_bool(dataset_cfg.get('gpu_video_transform', True))
            and dataset_cfg.get('dataset_class_name') in {
                'Dataset_DHGA',
            }
            and self.device.type == 'cuda'
        )
        self.gpu_multimodal_transform = None
        if self.gpu_multimodal_transform_enabled:
            self.gpu_multimodal_transform = GPUMultimodalVideoTransform(
                output_size=int(dataset_cfg.get('image_size', 224)),
                build_part_heatmap=as_bool(dataset_cfg.get('build_part_heatmap', True)),
                part_heatmap_output_size=dataset_cfg.get('part_heatmap_output_size'),
                part_heatmap_sigma=float(dataset_cfg.get('part_heatmap_sigma', 0.12)),
                part_heatmap_include_whole_hand=as_bool(dataset_cfg.get('part_heatmap_include_whole_hand', False)),
                part_heatmap_use_bones=as_bool(dataset_cfg.get('part_heatmap_use_bones', False)),
                part_heatmap_bone_sigma=dataset_cfg.get('part_heatmap_bone_sigma'),
            ).to(self.device)

        # ============ 1. dataset ============
        self._init_dataloaders()

        # ============ 2. model ============
        self._init_model()

        # ============ 3. optimizer ============
        self._init_optimizer()


    def _make_dataloader_generator(self, offset=0):
        generator = torch.Generator()
        generator.manual_seed(self.seed + offset)
        return generator

    def _build_dataloader_options(self, section_name):
        loader_conf = self.conf.get(section_name, {})
        num_workers = int(loader_conf.get('num_workers', 0))
        if num_workers < 0:
            raise ValueError(f"{section_name}.num_workers must be non-negative, got {num_workers}.")

        pin_memory_requested = as_bool(
            loader_conf.get('pin_memory', self.device.type == 'cuda')
        )
        options = {
            'num_workers': num_workers,
            'pin_memory': pin_memory_requested and self.device.type == 'cuda',
        }

        if num_workers > 0:
            prefetch_factor = int(loader_conf.get('prefetch_factor', 2))
            if prefetch_factor <= 0:
                raise ValueError(
                    f"{section_name}.prefetch_factor must be positive when num_workers > 0, "
                    f"got {prefetch_factor}."
                )
            options['prefetch_factor'] = prefetch_factor
            # Preserve historical worker seeding and shuffle order across epochs.
            options['persistent_workers'] = as_bool(
                loader_conf.get('persistent_workers', False)
            )

        return options

    def _build_train_sources(self):
        dataset_cfg = self.conf['dataset']
        train_sources = dataset_cfg.get('train_sources', None)
        if not train_sources:
            raise ValueError("dataset.train_sources must be provided in multi-source mode.")
        return train_sources

    def _prepare_model_batch(self, data, phase):
        if self.gpu_multimodal_transform_enabled:
            with torch.no_grad():
                return self.gpu_multimodal_transform(data, phase=phase)
        return data

    def _build_eval_sources(self):
        dataset_cfg = self.conf['dataset']
        eval_sources = dataset_cfg.get('eval_sources', None)
        if not eval_sources:
            raise ValueError("dataset.eval_sources must be provided in multi-source mode.")

        default_frames_per_sample = dataset_cfg.get('frames_per_sample', None)
        default_storage = str(dataset_cfg.get('storage', 'frames')).strip().lower()
        default_pack_root = dataset_cfg.get('pack_root', None)
        default_pack_extension = str(dataset_cfg.get('pack_extension', '.dhgpack')).strip()
        default_pack_verify_crc = as_bool(dataset_cfg.get('pack_verify_crc', False))
        normalized_sources = []
        for idx, source in enumerate(eval_sources):
            root = source.get('root', None)
            metadata_root = source.get('metadata_root', None)
            subsets = source.get('subsets', None)
            if root is None or metadata_root is None or subsets is None:
                raise ValueError(
                    f"Each eval source must provide 'root', 'metadata_root', and 'subsets'. Invalid source #{idx}: {source}"
                )

            label_files = source.get('label_files', [])
            if isinstance(label_files, str):
                label_files = [label.strip() for label in label_files.split(',') if label.strip()]
            if not label_files:
                raise ValueError(
                    f"Each eval source must provide non-empty 'label_files'. Invalid source #{idx}: {source}"
                )

            frames_per_sample = source.get('frames_per_sample', default_frames_per_sample)
            if frames_per_sample is None:
                raise ValueError(
                    f"Each eval source must provide 'frames_per_sample'. Invalid source #{idx}: {source}"
                )

            normalized_sources.append({
                'name': source.get('name', f'eval_{idx}'),
                'root': root,
                'metadata_root': metadata_root,
                'subsets': subsets,
                'label_files': label_files,
                'frames_per_sample': int(frames_per_sample),
                'storage': str(source.get('storage', default_storage)).strip().lower(),
                'pack_root': source.get('pack_root', default_pack_root),
                'pack_extension': str(
                    source.get('pack_extension', default_pack_extension)
                ).strip(),
                'pack_verify_crc': as_bool(
                    source.get('pack_verify_crc', default_pack_verify_crc)
                ),
                'subset_roots': source.get('subset_roots', {}),
            })
        return normalized_sources

    def _split_eval_key(self, eval_key):
        if "::" in eval_key:
            return eval_key.split("::", 1)
        return "default_eval", eval_key

    def _get_eval_tb_tag_parts(self, metric_name):
        metric_stem = os.path.splitext(metric_name)[0]
        keywords = metric_stem.split("_")
        session = keywords[1] if len(keywords) > 1 else "eval"
        metric_suffix = "_".join(keywords[3:]) if len(keywords) > 3 else metric_stem
        if not metric_suffix:
            metric_suffix = metric_stem
        return session, metric_suffix

    def _log_best_eer_by_source(self):
        # 最终特征仍使用历史兼容字段 best_eer；分支特征使用各自的最佳表。
        best_by_feature = {'feature': self.best_eer}
        best_by_feature.update(getattr(self, 'best_eer_by_feature', {}))
        best_epoch_by_feature = {'feature': self.best_epoch}
        best_epoch_by_feature.update(getattr(self, 'best_epoch_by_feature', {}))

        for feature_name, feature_best_eer in best_by_feature.items():
            feature_best_epoch = best_epoch_by_feature.get(feature_name, {})
            source_best_eers = {}
            for key, value in feature_best_eer.items():
                source_name, _ = self._split_eval_key(key)
                source_best_eers.setdefault(source_name, []).append(value)

                log_str = (
                    'feature : {}, task : {}, best eer : {}, epoch : {}'
                    .format(feature_name, key, value, feature_best_epoch.get(key, 0))
                )
                print(log_str)
                self.logger.info(log_str)

            for source_name, eer_list in source_best_eers.items():
                best_eer_avg = sum(eer_list) / len(eer_list)
                log_str = (
                    'feature : {}, source : {}, best_eer_avg : {}'
                    .format(feature_name, source_name, best_eer_avg)
                )
                print(log_str)
                self.logger.info(log_str)

    def _init_dataloaders(self):
        print('#' * 40)
        print('Building Dataset ...')
        print('#' * 40)

        if self.mode == 'train':
            self.train_dataset = make_dataset(
                self.conf,
                phase='train',
                data_sources=self._build_train_sources(),
            )
            train_loader_options = self._build_dataloader_options('train')
            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                               batch_size=self.conf["train"]["batch_size"],
                                               shuffle=True,
                                               drop_last=True,
                                               generator=self._make_dataloader_generator(0),
                                               worker_init_fn=seed_dataloader_worker,
                                               **train_loader_options)


        if self.mode in ('train', 'test'):
            self.eval_dataLoaders={}
            self.best_eer = {}
            self.best_epoch = {}
            self.best_eer_by_feature = {}
            self.best_epoch_by_feature = {}
            self.best_eer_avg = {}
            self.eval_source_lookup = {}
            for source in self._build_eval_sources():
                for label_file in source['label_files']:
                    eval_key = f"{source['name']}::{label_file}"
                    self.eval_source_lookup[eval_key] = {
                        'source_name': source['name'],
                        'root': source['root'],
                        'metadata_root': source['metadata_root'],
                        'subsets': source['subsets'],
                        'label_file': label_file,
                        'frames_per_sample': source['frames_per_sample'],
                        'storage': source['storage'],
                        'pack_root': source['pack_root'],
                        'pack_extension': source['pack_extension'],
                        'pack_verify_crc': source['pack_verify_crc'],
                        'subset_roots': source['subset_roots'],
                    }
                    eval_dataset = make_dataset(
                        self.conf,
                        phase='eval',
                        data_sources=[{
                            'name': source['name'],
                            'root': source['root'],
                            'metadata_root': source['metadata_root'],
                            'subsets': source['subsets'],
                            'label_file': label_file,
                            'frames_per_sample': source['frames_per_sample'],
                            'storage': source['storage'],
                            'pack_root': source['pack_root'],
                            'pack_extension': source['pack_extension'],
                            'pack_verify_crc': source['pack_verify_crc'],
                            'subset_roots': source['subset_roots'],
                        }]
                    )
                    eval_loader_options = self._build_dataloader_options('evaluate')
                    self.eval_dataLoaders[eval_key] = DataLoader(dataset=eval_dataset,
                                                                 batch_size=self.conf["evaluate"]["batch_size"],
                                                                 shuffle=False,
                                                                 drop_last=False,
                                                                 generator=self._make_dataloader_generator(10000 + len(self.eval_dataLoaders)),
                                                                 worker_init_fn=seed_dataloader_worker,
                                                                 **eval_loader_options)
                    self.best_eer[eval_key] = 1.0
                    self.best_epoch[eval_key] = 0

    def _init_model(self):
        print('#' * 40)
        print(f"Model is {self.conf['model']['model_file_name']}.{self.conf['model']['model_class_name']}")
        print('#' * 40)
        print(f"selfdevice is {self.device}")
        self.model = make_model(self.conf).to(self.device)

    def _init_optimizer(self):
        """初始化优化器"""
        optimizer_type = self.conf['train']['optimizer']
        learning_rate = self.conf["train"]['lr'].split(",")

        print(f"Optimizer is {optimizer_type}")
        self.logger.info(f'optimizer is {optimizer_type}')

        if optimizer_type == 'adam':
            if len(learning_rate) > 1:
                # 双学习率设置（预训练参数 + 新参数）
                self._init_dual_lr_optimizer(learning_rate)
            else:
                # 单一学习率
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=float(learning_rate[0]),
                    capturable=False,
                    foreach=True
                )

    def _init_dual_lr_optimizer(self, learning_rate):
        """初始化双学习率优化器（预训练参数 + 新参数）"""
        pretrained_params = []
        pretrained_param_names = []
        new_params = []
        new_params_names = []
        model_for_grouping = self.model.module if hasattr(self.model, 'module') else self.model
        declared_prefixes = tuple(
            str(prefix) for prefix in getattr(
                model_for_grouping,
                'pretrained_parameter_prefixes',
                (),
            )
        )
        declared_exclude_prefixes = tuple(
            str(prefix) for prefix in getattr(
                model_for_grouping,
                'pretrained_parameter_exclude_prefixes',
                (),
            )
        )

        def is_declared_pretrained(name):
            if any(name.startswith(prefix) for prefix in declared_exclude_prefixes):
                return False
            return any(name.startswith(prefix) for prefix in declared_prefixes)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                group_name = name[7:] if name.startswith('module.') else name
                is_pbnet_pretrained_branch = (
                    group_name.startswith('P_Branch.') or group_name.startswith('B_Branch.')
                ) and '.fc.' not in group_name
                # Models that split a torchvision backbone into stem/layerN
                # attributes declare these prefixes explicitly.  The fallback
                # keeps the historical backbone/P_Branch behavior unchanged.
                is_torchvision_backbone = (
                    ('backbone' in group_name and '.fc.' not in group_name
                     and not group_name.endswith('.fc.weight')
                     and not group_name.endswith('.fc.bias'))
                    or is_declared_pretrained(group_name)
                )
                if is_pbnet_pretrained_branch or is_torchvision_backbone:
                    pretrained_params.append(param)
                    pretrained_param_names.append(group_name)
                elif 'fc' in group_name:
                    new_params.append(param)
                    new_params_names.append(group_name)
                else:
                    new_params.append(param)
                    new_params_names.append(group_name)

        self.optimizer = torch.optim.Adam(
            [
                {'params': pretrained_params, 'lr': float(learning_rate[0])},
                {'params': new_params, 'lr': float(learning_rate[1])}
            ],
            capturable=False,
            foreach=True
        )
        pretrained_count = sum(parameter.numel() for parameter in pretrained_params)
        new_count = sum(parameter.numel() for parameter in new_params)
        print(
            'Optimizer groups: '
            f'pretrained={len(pretrained_params)} tensors/{pretrained_count} params '
            f'lr={float(learning_rate[0])}; '
            f'new={len(new_params)} tensors/{new_count} params '
            f'lr={float(learning_rate[1])}'
        )
        self.logger.info(
            'Optimizer groups: pretrained=%s tensors/%s params lr=%s; '
            'new=%s tensors/%s params lr=%s',
            len(pretrained_params), pretrained_count, float(learning_rate[0]),
            len(new_params), new_count, float(learning_rate[1]),
        )

    def _set_model_epoch(self, epoch):
        """Notify models that use epoch-based auxiliary-task warm-up."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        setter = getattr(model, 'set_epoch', None)
        if callable(setter):
            setter(int(epoch))

    def _load_checkpoint(self, ckpt_path: str, load_optimizer=True, restore_epoch=True):
        """加载检查点"""
        print('#' * 40)
        print(f"Loading pretrained parameters at '{ckpt_path}'")

        ckpt_path = os.path.expanduser(os.path.expandvars(str(ckpt_path)))
        if self.device.type == "cpu":
            checkpoint = torch.load(ckpt_path, map_location='cpu')
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

        loaded_epoch = checkpoint.get('epoch', 0)
        if restore_epoch:
            self.current_epoch = loaded_epoch
        print(f"{loaded_epoch} epoch model params is loading...")

        model_state_load = checkpoint.get("model", checkpoint)
        model_state = self.model.state_dict()
        state_cur = load_pretrained_params(model_state_load, model_state)
        self.model.load_state_dict(state_cur)

        # 兼容旧 checkpoint：分支 EER 统计字段不存在时沿用当前初始化值。
        if self.mode == 'train':
            if 'best_eer' in checkpoint and hasattr(self, 'best_eer'):
                self.best_eer.update(checkpoint['best_eer'])
            if 'best_epoch' in checkpoint and hasattr(self, 'best_epoch'):
                self.best_epoch.update(checkpoint['best_epoch'])
            for attr in ('best_eer_by_feature', 'best_epoch_by_feature', 'best_eer_avg'):
                if attr in checkpoint:
                    setattr(self, attr, checkpoint[attr])

        if load_optimizer and self.mode == "train" and "optimizer" in checkpoint:
            print("Loading optimizer parameters...")
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        elif self.mode == "train" and not load_optimizer:
            print("Model weights loaded for finetuning; optimizer and epoch schedule are reinitialized.")

        print('Reload successfully')
        print('#' * 40)

    def _resolve_checkpoint_paths(self, resume_from):
        if isinstance(resume_from, (list, tuple)):
            raw_paths = list(resume_from)
        else:
            raw_paths = [path.strip() for path in str(resume_from).split(',') if path.strip()]

        checkpoint_paths = []
        for raw_path in raw_paths:
            raw_path = os.path.expanduser(os.path.expandvars(str(raw_path).strip()))
            if raw_path in ('None', ''):
                continue
            if os.path.isdir(raw_path):
                best_paths = [
                    os.path.join(raw_path, name)
                    for name in sorted(os.listdir(raw_path))
                    if name.endswith('_best.pth')
                ]
                checkpoint_paths.extend(best_paths)
            elif os.path.isfile(raw_path):
                checkpoint_paths.append(raw_path)
            else:
                raise FileNotFoundError(f'Checkpoint path not found: {raw_path}')

        if not checkpoint_paths:
            raise FileNotFoundError(f'No checkpoint files found from resume_from={resume_from}')
        return checkpoint_paths

    def _load_finetune_weights(self, finetune_from):
        checkpoint_paths = self._resolve_checkpoint_paths(finetune_from)
        if len(checkpoint_paths) > 1:
            print(
                "finetune_from resolved to multiple checkpoints; "
                f"using the first one: {checkpoint_paths[0]}"
            )
        self._load_checkpoint(
            checkpoint_paths[0],
            load_optimizer=False,
            restore_epoch=False,
        )
        self.current_epoch = 0

    def _flatten_eer_results(self, result_sum):
        feature_results = result_sum.get('feature', result_sum)
        return {
            key: float(value)
            for key, value in feature_results.items()
            if isinstance(value, (int, float, np.floating))
        }

    def evaluate_checkpoints(self):
        resume_from = self.conf.get('model', {}).get('resume_from', None)
        checkpoint_paths = self._resolve_checkpoint_paths(resume_from)
        print(f'Testing {len(checkpoint_paths)} checkpoint(s).')
        self.logger.info(f'Testing {len(checkpoint_paths)} checkpoint(s).')

        rows = []
        best_overall = None
        best_by_metric = {}
        for ckpt_idx, ckpt_path in enumerate(checkpoint_paths):
            self._load_checkpoint(ckpt_path)
            result_sum = self.evaluate(epoch=self.current_epoch)
            metrics = self._flatten_eer_results(result_sum)
            avg_eer = sum(metrics.values()) / len(metrics) if metrics else float('inf')

            row = {
                'checkpoint': ckpt_path,
                'epoch': self.current_epoch,
                'avg_eer': avg_eer,
                **metrics,
            }
            rows.append(row)

            if best_overall is None or avg_eer < best_overall['avg_eer']:
                best_overall = row
            for metric_name, eer in metrics.items():
                if metric_name not in best_by_metric or eer < best_by_metric[metric_name]['eer']:
                    best_by_metric[metric_name] = {
                        'eer': eer,
                        'checkpoint': ckpt_path,
                        'epoch': self.current_epoch,
                    }

            log_str = f'Test checkpoint {ckpt_idx + 1}/{len(checkpoint_paths)} avg_eer={avg_eer:.6f}: {ckpt_path}'
            print(log_str)
            self.logger.info(log_str)

        result_dir = os.path.join(self.conf['task']['work_dir'], 'test_results', self.conf['task']['name'])
        os.makedirs(result_dir, exist_ok=True)
        result_csv = os.path.join(result_dir, 'checkpoint_results.csv')
        fieldnames = ['checkpoint', 'epoch', 'avg_eer']
        metric_names = sorted({key for row in rows for key in row.keys() if key not in fieldnames})
        with open(result_csv, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames + metric_names)
            writer.writeheader()
            writer.writerows(rows)

        if best_overall is not None:
            log_str = (
                f"Best checkpoint by avg_eer={best_overall['avg_eer']:.6f}: "
                f"{best_overall['checkpoint']}"
            )
            print(log_str)
            self.logger.info(log_str)

        for metric_name, info in best_by_metric.items():
            log_str = (
                f"Best {metric_name}: eer={info['eer']:.6f}, "
                f"epoch={info['epoch']}, checkpoint={info['checkpoint']}"
            )
            print(log_str)
            self.logger.info(log_str)

        print(f'Test results saved to {result_csv}')
        self.logger.info(f'Test results saved to {result_csv}')
        return {
            'rows': rows,
            'best_overall': best_overall,
            'best_by_metric': best_by_metric,
            'result_csv': result_csv,
        }

    def _update_learning_rate(self, epoch: int):
        lr_gamma = float(self.conf['train'].get('lr_gamma', 1.0))
        min_new_lr = float(self.conf['train'].get('min_new_lr', 0.0))
        min_pretrained_lr = float(self.conf['train'].get('min_pretrained_lr', 0.0))
        train_cfg = self.conf['train']
        if len(self.optimizer.param_groups) > 1:
            # 双学习率模式
            new_lr_step = train_cfg.get('new_lr_step', None)
            pretrained_lr_step = train_cfg.get('pretrained_lr_step', None)
            if new_lr_step is not None and (epoch + 1) % int(new_lr_step) == 0:
                self.optimizer.param_groups[1]['lr'] = max(
                    self.optimizer.param_groups[1]['lr'] * lr_gamma,
                    min_new_lr,
                )
            if pretrained_lr_step is not None and (epoch + 1) % int(pretrained_lr_step) == 0:
                self.optimizer.param_groups[0]['lr'] = max(
                    self.optimizer.param_groups[0]['lr'] * lr_gamma,
                    min_pretrained_lr,
                )

            print('\n lr: {}'.format(' '.join(str(group['lr']) for group in self.optimizer.param_groups)))
        else:
            # 单学习率仍按配置中的 new_lr_step 衰减。
            lr_step = train_cfg.get('new_lr_step', train_cfg.get('lr_step', None))
            if lr_step is not None and (epoch + 1) % int(lr_step) == 0:
                for group in self.optimizer.param_groups:
                    group['lr'] = max(
                        group['lr'] * lr_gamma,
                        min_new_lr,
                    )

            print('lr: {}'.format(' '.join(str(group['lr']) for group in self.optimizer.param_groups)))

    def train(self):
        """训练主循环"""
        print("Begin training ...")
        self.logger.info("Begin training ...")

        if self.resume and 'resume_from' in self.conf['model']:
            self._load_checkpoint(self.conf['model']['resume_from'])
        else:
            finetune_from = self.conf.get('model', {}).get('finetune_from', None)
            if finetune_from not in (None, 'None', ''):
                self._load_finetune_weights(finetune_from)

        log_interval = int(self.conf.get('train', {}).get('log_interval', 20))
        if log_interval <= 0:
            raise ValueError(f"train.log_interval must be positive, got {log_interval}.")

        for epoch in range(self.current_epoch, int(self.epochs)):

            # set_random_seed(epoch)

            # lr
            self._update_learning_rate(epoch)

            for group_index, group in enumerate(self.optimizer.param_groups):
                self.tb_writer.add_scalar(
                    f'train/learning_rate/group_{group_index}',
                    float(group['lr']),
                    global_step=epoch,
                )
            if len(self.optimizer.param_groups) > 1:
                self.tb_writer.add_scalar(
                    'train/learning_rate/new_to_pretrained_ratio',
                    float(self.optimizer.param_groups[1]['lr'] / max(self.optimizer.param_groups[0]['lr'], 1e-30)),
                    global_step=epoch,
                )

            lr_text = ' '.join(str(group['lr']) for group in self.optimizer.param_groups)
            log_str = f'{"#" * 20} {self.conf["task"]["name"]} Epoch: {epoch} lr: {lr_text} {"#" * 20}'
            print('\n', log_str)
            self.logger.info(log_str)

            total_loss = None
            num_batches = 0
            # #
            # if epoch ==0 :
            #     self.evaluate(epoch)

            self._set_model_epoch(epoch)
            self.model.train()

            for batch, data in enumerate(tqdm(self.train_dataloader)):
                data = self._prepare_model_batch(data, phase='train')
                # ============ 前向+反向传播 ============
                self.optimizer.zero_grad(set_to_none=True)
                fis = self.model(data)
                loss_ = fis["loss"].mean()
                loss_.backward()
                self.optimizer.step()
                del loss_

                batch_loss = fis["loss"].detach().mean()
                total_loss = batch_loss if total_loss is None else total_loss + batch_loss
                num_batches += 1

                should_log = (
                    self.global_step % log_interval == 0
                    or (batch + 1) == len(self.train_dataloader)
                )
                if should_log:
                    log_lst = ['Train => step: {0:>5}'.format(self.global_step)]
                    for key, value in fis.items():
                        if isinstance(value, torch.Tensor) and value.ndim == 0:
                            new_value = value.detach().mean().item()
                            self.tb_writer.add_scalar("train/{}".format(key), new_value, self.global_step)
                            log_lst.append('{} : {:5f}'.format(key, new_value))
                    log_str = ',   '.join(log_lst)
                    self.logger.info(log_str)

                self.global_step += 1
                del fis, data

            if num_batches == 0:
                raise RuntimeError(
                    "Training DataLoader produced no batches. Check dataset size, batch_size, and drop_last."
                )
            epoch_loss = (total_loss / num_batches).item()
            self.tb_writer.add_scalar("train/epoch_loss", epoch_loss, global_step=epoch)

            if epoch % int(self.eval_internal) == 0 or (epoch + 1) == int(self.epochs):
                self.evaluate(epoch)

                model_state_dict = self.model.state_dict()
                checkpoint = {
                    "model": model_state_dict,
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_eer": self.best_eer,
                    "best_epoch": self.best_epoch,
                    "best_eer_by_feature": getattr(self, 'best_eer_by_feature', {}),
                    "best_epoch_by_feature": getattr(self, 'best_epoch_by_feature', {}),
                    "best_eer_avg": getattr(self, 'best_eer_avg', {}),
                }

                model_save_path = os.path.join(self.model_save_dir, 'last.pth')
                torch.save(checkpoint, model_save_path)
                print('save model : {}'.format(model_save_path.split("/")[-1]))

                self._log_best_eer_by_source()

    def evaluate(self, epoch):
        print('Evaluating ...')
        self._set_model_epoch(epoch)
        self.model.eval()

        # Text prompts are input-independent. Models that expose this hook
        # precompute their trained post-refinement text representations once.
        prepare_text_cache = getattr(self.model, 'prepare_text_inference_cache', None)
        if callable(prepare_text_cache):
            prepare_text_cache()

        result_sum = {}

        # 第一批自动发现最终及分支身份描述子，其后所有批次保持相同输出约定。
        eval_feature_dims = None

        for eval_file_name, eval_dataloader in self.eval_dataLoaders.items():
            vid_names = []
            all_features = {}

            with torch.no_grad():
                for batch, data in tenumerate(eval_dataloader, desc=eval_file_name):
                    data = self._prepare_model_batch(data, phase='eval')
                    fis = self.model(data)

                    descriptors = collect_identity_features(fis, len(data['vid_name']))
                    dimensions = {name: value.shape[1] for name, value in descriptors.items()}
                    if eval_feature_dims is None:
                        eval_feature_dims = dimensions
                        result_sum = {name: {} for name in descriptors}
                    elif dimensions != eval_feature_dims:
                        raise ValueError('Identity descriptor names/dimensions changed between evaluation batches.')
                    for feature_name, features in descriptors.items():
                        all_features.setdefault(feature_name, []).extend(features.detach().cpu().numpy())
                    vid_names.extend([str(vid_name) for vid_name in data["vid_name"]])
                    del fis, data, descriptors, features

            for feature_name in all_features:
                # 所有特征都提取完毕，开始计算eer,传入all_features和vid_names
                self.logger.info(f'Evaluating descriptor: {feature_name}, dataset: {eval_file_name}')
                result_sum[feature_name].update(self.get_eer(all_features[feature_name], vid_names, epoch,eval_file_name))

        # 显示结果，并分别维护每个描述子的逐任务最佳 EER。
        if not hasattr(self, 'best_eer_by_feature'):
            self.best_eer_by_feature = {}
        if not hasattr(self, 'best_epoch_by_feature'):
            self.best_epoch_by_feature = {}
        if not hasattr(self, 'best_eer_avg'):
            self.best_eer_avg = {}

        for feature_name, result_ in result_sum.items():
            source_values = {}
            feature_best_eer = self.best_eer if feature_name == 'feature' else self.best_eer_by_feature.setdefault(feature_name, {})
            feature_best_epoch = self.best_epoch if feature_name == 'feature' else self.best_epoch_by_feature.setdefault(feature_name, {})
            for key in result_:
                feature_best_eer.setdefault(key, 1.0)
                feature_best_epoch.setdefault(key, 0)

            print(f'{feature_name} EER results:')
            for key, value in result_.items():
                print('{}: {}'.format(key, value))
                source_name, metric_name = self._split_eval_key(key)
                session, type_gesture_name = self._get_eval_tb_tag_parts(metric_name)
                # 保留最终特征的历史 tag；分支各有命名空间，最后一个 epoch 也记录。
                prefix = source_name if feature_name == 'feature' else f'{source_name}/{feature_name}'
                self.tb_writer.add_scalar(f'{prefix}/{session}_EER/{type_gesture_name}', value, epoch)
                source_values.setdefault(source_name, []).append(value)
                eer_log = (
                    f'Eval EER: feature={feature_name}, task={key}, '
                    f'epoch={epoch}, eer={value:.6f}'
                )
                print(eer_log)
                self.logger.info(eer_log)
                if value < feature_best_eer[key]:
                    feature_best_eer[key] = value
                    feature_best_epoch[key] = epoch

                    if feature_name == 'feature':

                        if self.mode != 'train':
                            continue

                        safe_key = key.replace('::', '__')
                        model_save_path = os.path.join(self.model_save_dir, safe_key + '_best.pth')
                        if not os.path.isdir(self.model_save_dir):
                            os.makedirs(self.model_save_dir)
                        if self.conf["distributed"]:
                            model_state_dict = self.model.module.state_dict()
                        else:
                            model_state_dict = self.model.state_dict()
                        checkpoint = {
                            "model": model_state_dict,
                            "best_eer": self.best_eer[key],
                            "best_epoch": self.best_epoch[key],
                        }
                        torch.save(checkpoint, model_save_path)
                        print('save model : {}'.format(model_save_path.split("/")[-1]))
            for source_name, values in source_values.items():
                eer_avg = sum(values) / len(values)
                best_values = [
                    feature_best_eer[key]
                    for key in result_
                    if self._split_eval_key(key)[0] == source_name
                ]
                best_eer_avg = sum(best_values) / len(best_values)
                self.best_eer_avg.setdefault(feature_name, {})[source_name] = best_eer_avg
                self.tb_writer.add_scalar(f'{source_name}/{feature_name}/eer_avg', eer_avg, epoch)
                self.tb_writer.add_scalar(f'{source_name}/{feature_name}/best_eer_avg', best_eer_avg, epoch)
                avg_log = (
                    f'Eval EER average: feature={feature_name}, source={source_name}, '
                    f'epoch={epoch}, eer_avg={eer_avg:.6f}, best_eer_avg={best_eer_avg:.6f}'
                )
                print(avg_log)
                self.logger.info(avg_log)
        self.model.train()

        print("Eval Done!")
        if self.mode == 'train':
            time.sleep(3)
        return result_sum

    def get_eer(self, all_features, vid_names, epoch,eval_file_name):
        # t1 = time.time()
        assert (len(all_features) == len(vid_names))
        #把vid_names和all_features合并成字典
        name_feature_dic = dict(zip(vid_names, all_features))

        eval_source = self.eval_source_lookup[eval_file_name]
        eval_label_path = os.path.join(eval_source['metadata_root'], eval_source['label_file'])
        fin = open(eval_label_path)
        fin_csv = csv.reader(fin)
        features1 = []
        features2 = []
        pair_labels = []
        for i, row in enumerate(fin_csv): # row[0]是vid1，row[1]是vid2
            if i > 0:

                features1.append(name_feature_dic[row[0]])
                features2.append(name_feature_dic[row[1]])
                pair_labels.append(row[2] == '1')
        features1 = np.asarray(features1)
        features2 = np.asarray(features2)
        pair_labels = np.asarray(pair_labels)
        result_lst = calculate_eer(features1, features2, pair_labels)
        FARs, FRRs, eer, threshold, distances = result_lst

        evaluate_result = {eval_file_name: eer}
        log_lst = ['Eval => epoch: {0:>5}'.format(epoch)]
        # t_e2 = time.time()  # 记录时间
        for key, value in evaluate_result.items():
            # new_value = value
            # self.tb_writer.add_scalar("{}_EER/{}".format(self.session, self.type_gesture_name), value, epoch)
            log_lst.append('{} : {:5f}'.format(key, value))
        log_str = ',   '.join(log_lst)
        self.logger.info(log_str)

        return evaluate_result
