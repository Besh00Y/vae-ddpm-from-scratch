import torch_fidelity


def compute_metrics(gen_dir, reference="cifar10-train", cuda=True, batch_size=100):
    """FID and Inception Score of the images in gen_dir against a reference set."""
    m = torch_fidelity.calculate_metrics(
        input1=gen_dir,
        input2=reference,
        input2_cache_name=reference,   # cache reference statistics, computed once
        cuda=cuda,
        isc=True,
        fid=True,
        batch_size=batch_size,
        verbose=False,
    )
    return {
        "fid": float(m["frechet_inception_distance"]),
        "is_mean": float(m["inception_score_mean"]),
        "is_std": float(m["inception_score_std"]),
    }