'''
@Project ：DHGA 
@File    ：eer.py
@IDE     ：PyCharm 
@Author  ：WXL
@Date    ：2026/4/27
'''


import time

import torch

import numpy as np


def calculate_eer(features1, features2, pair_labels, test_fusion_method="mean", length_rate=1, model=None, merge_rate=None):
    assert (features1.shape == features2.shape)
    eps = 1e-12
    t1 = time.time()
    # linear
    if test_fusion_method in ["mean", "concatenate", "vap", "None"]:
        feature_norm_1 = np.linalg.norm(features1, axis=1)
        feature_norm_2 = np.linalg.norm(features2, axis=1)
        norm_product = np.clip(feature_norm_1 * feature_norm_2, a_min=eps, a_max=None)
        distances = 1 - np.sum(features1 * features2, axis=1) / norm_product

    elif test_fusion_method in ["mean_phy", ]:
        distances = 1 - np.sum(features1 * features2, axis=1)

    # # dtw
    # elif test_fusion_method in ["dtw"]:
    #     distances = []
    #     for i in range(features1.shape[0]):
    #         d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(features1[i], features2[i], 'cosine', warp=1)
    #         cost_lst = cost_matrix[path[0], path[1]]
    #         cost_lst.sort()  # 距离从小到大排序,相似度从最像到最不像
    #         d = np.mean(cost_lst[-int(cost_lst.shape[0] / 2):])  # 动态 1/2 最不像的
    #         # d = np.mean(cost_matrix[path[0], path[1]])
    #         distances.append(d)
    #     distances = np.array(distances)
    #
    # # cross_attention
    # elif test_fusion_method in ["crossAttention"]:
    #     assert model is not None
    #     features1_mean = np.mean(features1, axis=1)
    #     features2_mean = np.mean(features2, axis=1)
    #     distances_1 = 1 - np.sum(features1_mean * features2_mean, axis=1) / (
    #             np.linalg.norm(features1_mean, axis=1) * np.linalg.norm(features2_mean, axis=1))
    #     features1 = torch.tensor(features1).cuda()
    #     features2 = torch.tensor(features2).cuda()
    #     cross_features1 = model.crossAttention.cross_attention(features1, features2).detach().cpu().numpy()
    #     cross_features2 = model.crossAttention.cross_attention(features2, features1).detach().cpu().numpy()
    #     distances_2 = 1 - np.sum(cross_features1 * cross_features2, axis=1) / (
    #             np.linalg.norm(cross_features1, axis=1) * np.linalg.norm(cross_features2, axis=1))
    #
    #     distances = 1.0 * distances_1 + 0.0 * distances_2
    #
    # elif test_fusion_method in ["score_fusion", "temp_concat"]:
    #     feature_dim = features1.shape[1] // (length_rate + 1) * length_rate
    #     features11 = features1[:, 0:feature_dim]
    #     features12 = features1[:, feature_dim:]
    #     features21 = features2[:, 0:feature_dim]
    #     features22 = features2[:, feature_dim:]
    #     distances_1 = 1 - np.sum(features11 * features21, axis=1) / (
    #             np.linalg.norm(features11, axis=1) * np.linalg.norm(features21, axis=1))
    #     distances_2 = 1 - np.sum(features12 * features22, axis=1) / (
    #             np.linalg.norm(features12, axis=1) * np.linalg.norm(features22, axis=1))
    #     if merge_rate is None:
    #         distances_list = [0.1 * proportion * distances_1 + (1 - 0.1 * proportion) * distances_2 for proportion in range(11)]
    #         EER_list = []
    #         for i, distances in enumerate(distances_list):
    #             min_dis = np.min(distances)
    #             max_dis = np.max(distances)
    #             accept_distances = distances[pair_labels == True]
    #             reject_distances = distances[pair_labels == False]
    #
    #             FARs = []
    #             FRRs = []
    #             thresholds = []
    #             errors = []
    #             for threshold in np.linspace(min_dis, max_dis, num=1000):  # threshold linspace
    #                 thresholds.append(threshold)
    #                 FRR = np.sum(accept_distances >= threshold) / accept_distances.shape[0]
    #                 FAR = np.sum(reject_distances < threshold) / reject_distances.shape[0]
    #                 FRRs.append(FRR)
    #                 FARs.append(FAR)
    #                 errors.append(abs(FAR - FRR))
    #             min_errors_idx = np.argmin(np.asarray(errors))
    #             EER = (FRRs[min_errors_idx] + FARs[min_errors_idx]) / 2
    #             EER_list.append(EER)
    #         return EER_list
    #     else:
    #         distances = merge_rate * distances_1 + (1 - merge_rate) * distances_2
    #
    # else:
    #     print("test_fusion_method : {} is not implemented".format(test_fusion_method))
    #     exit(0)

    # t2 = time.time()
    # print("In calculate_eer, calculate distance time is {}".format(t2-t1))
    min_dis = np.min(distances)
    max_dis = np.max(distances)

    accept_distances = distances[pair_labels == True]
    reject_distances = distances[pair_labels == False]
    if accept_distances.shape[0] == 0 or reject_distances.shape[0] == 0:
        raise ValueError(
            f"EER requires both positive and negative pairs, got "
            f"{accept_distances.shape[0]} positive and {reject_distances.shape[0]} negative pairs."
        )

    FARs = []
    FRRs = []
    thresholds = []
    errors = []
    for threshold in np.linspace(min_dis, max_dis, num=1000):  # threshold linspace
        thresholds.append(threshold)
        FRR = np.sum(accept_distances >= threshold) / accept_distances.shape[0]
        FAR = np.sum(reject_distances < threshold) / reject_distances.shape[0]
        FRRs.append(FRR)
        FARs.append(FAR)
        errors.append(abs(FAR - FRR))
    # t3 = time.localtimetime.time()
    # print("In calculate_eer, calculate FAR&FRR time is {}".format(t3-t2))
    min_errors_idx = np.argmin(np.asarray(errors))
    #
    # # 画出ROC曲线
    # plot_and_save_roc(FARs, FRRs)

    EER = (FRRs[min_errors_idx] + FARs[min_errors_idx]) / 2
    best_thres = thresholds[min_errors_idx]
    return FARs, FRRs, EER, best_thres, distances

