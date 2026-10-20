'''
@Project ：DHGA 
@File    ：dataset_utils.py
@IDE     ：PyCharm 
@Author  ：WXL
@Date    ：2026/4/26
'''

import importlib
import random
import numpy as np

def make_dataset(conf, phase, **kwargs):
    dataset_file_name=conf['dataset']['dataset_file_name']
    dataset_class_name=conf['dataset']['dataset_class_name']
    dataset_file = importlib.import_module(f'.dataset.{dataset_file_name}', 'src')
    MyDataset = getattr(dataset_file, dataset_class_name)
    dataset = MyDataset(conf, phase, **kwargs)
    return dataset


def sampling_tsn(num_frames, num_segments, phase, strip_num=0):
    """
    Split the complete video into temporal windows and sample one frame per window.

    Training uses an independent random frame from every window. Evaluation uses
    the deterministic center frame of every window.
    """
    if num_frames < num_segments:
        raise ValueError(
            f"TSN requires num_frames >= num_segments, got "
            f"num_frames={num_frames}, num_segments={num_segments}."
        )

    # linspace covers the complete video when num_frames is not divisible by
    # num_segments. For 240 frames and 60 segments, every window has 4 frames.
    segment_boundaries = np.linspace(
        0,
        num_frames,
        num_segments + 1,
        dtype=np.int64,
    )
    segment_starts = segment_boundaries[:-1]
    segment_ends = segment_boundaries[1:]

    if phase == 'train':
        frame_idx = np.fromiter(
            (
                np.random.randint(low=start, high=end)
                for start, end in zip(segment_starts, segment_ends)
            ),
            dtype=np.int64,
            count=num_segments,
        )
    else:
        frame_idx = (segment_starts + segment_ends) // 2

    assert frame_idx.size == num_segments
    return frame_idx + strip_num


# get sample slice list for each sample, this slice can be used for sequence data sampling
def get_slice_list(conf, phase, video_len=None):
    task_type = conf["task"]['type']
    if video_len is None:
        video_len = conf["dataset"]['frames_per_sample']
    sample_len = int(conf["dataset"]['frames_per_video'])
    if phase == 'eval' and conf['dataset'].get('eval_frames_per_video', None) is not None:
        sample_len = int(conf['dataset']['eval_frames_per_video'])

    if task_type == "FG":
        if phase == "train":
            if "is_sample" in conf and conf["is_sample"] is True:
                blockStartPos = random.randrange(4, 40, 5)
            else:
                blockStartPos = random.randint(0, video_len - sample_len)  # 随机选择一个位置作为切片的起点，用于训练情况
        else:
            blockStartPos = (video_len + 1) // 2 - 10 - 1  # -1 to zero，中间截取20帧，用于测试情况
        slicelist = list(range(blockStartPos, blockStartPos + sample_len))  # 生成各切片的索引
    else:
        # 用于MG等情况，按照tsn的形式读取数据，抽取64帧
        slicelist = sampling_tsn(video_len, num_segments=sample_len, phase=phase)

    return slicelist
