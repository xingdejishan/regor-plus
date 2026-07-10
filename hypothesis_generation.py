import torch


def sample_correspondences(src_corr, tgt_corr, sample_count):
    if src_corr.shape[1] != tgt_corr.shape[1]:
        raise ValueError("Source and target correspondences must have equal length.")
    count = min(int(sample_count), src_corr.shape[1])
    if count < 3:
        raise ValueError("At least three seed correspondences are required.")
    indices = torch.randperm(src_corr.shape[1], device=src_corr.device)[:count]
    return src_corr[:, indices], tgt_corr[:, indices]
