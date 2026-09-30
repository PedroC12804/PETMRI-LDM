"""
Image-quality metric functions for PET reconstruction evaluation.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import linalg


def compute_body_mask(gt_slice, threshold=0.01):
    return gt_slice > threshold


def compute_metrics(generated, ground_truth, mask):
    """
    generated, ground_truth: 2D numpy arrays (single slice, decoded PET)
    mask: boolean 2D array, True = inside body/tracer region

    Returns a dict of scalar metrics for this one slice.
    """
    data_range = float(ground_truth.max() - ground_truth.min())
    if data_range <= 0:
        data_range = 1.0  # avoid div-by-zero on empty/degenerate slices

    whole_psnr = peak_signal_noise_ratio(ground_truth, generated, data_range=data_range)
    whole_ssim = structural_similarity(ground_truth, generated, data_range=data_range)

    if mask.sum() > 0:
        gt_masked = ground_truth[mask]
        gen_masked = generated[mask]
        masked_mse = float(np.mean((gt_masked - gen_masked) ** 2))
        masked_nrmse = float(np.sqrt(masked_mse) / (gt_masked.max() - gt_masked.min() + 1e-8))
    else:
        masked_mse = float("nan")
        masked_nrmse = float("nan")

    return {
        "psnr": float(whole_psnr),
        "ssim": float(whole_ssim),
        "masked_mse": masked_mse,
        "masked_nrmse": masked_nrmse,
    }


def to_lpips_tensor(img, device):
    """
    Converts a single 2D grayscale numpy array into the (1,3,H,W),
    [-1,1]-normalized tensor LPIPS expects. Normalization is per-image
    (based on that image's own min/max), consistent with the
    per-image data_range used for PSNR/SSIM above.
    """
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        norm = np.zeros_like(img)
    else:
        norm = 2 * (img - mn) / (mx - mn) - 1

    t = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = t.repeat(1, 3, 1, 1)  # LPIPS expects 3 channels
    return t.to(device)


@torch.no_grad()
def compute_lpips(generated, ground_truth, lpips_model, device):
    gen_t = to_lpips_tensor(generated, device)
    gt_t = to_lpips_tensor(ground_truth, device)
    return float(lpips_model(gen_t, gt_t).item())


def compute_all_metrics(generated, ground_truth, lpips_model=None, device=None):
    """
    Convenience wrapper: computes PSNR/SSIM/masked-MSE/masked-NRMSE,
    plus LPIPS if a lpips_model is provided.
    """
    mask = compute_body_mask(ground_truth)
    scores = compute_metrics(generated, ground_truth, mask)
    if lpips_model is not None:
        scores["lpips"] = compute_lpips(generated, ground_truth, lpips_model, device)
    return scores


def aggregate(metrics_list):
    """
    Averages a list of per-sample metric dicts into {metric: {mean, std}}.
    NaNs (e.g. from degenerate empty-mask slices) are excluded.
    """
    if len(metrics_list) == 0:
        return {}
    keys = metrics_list[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        agg[k] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) else float("nan"),
        }
    return agg

_FID_MODEL_CACHE = {}


def get_fid_model(device):
    from pytorch_fid.inception import InceptionV3

    key = str(device)
    if key not in _FID_MODEL_CACHE:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _FID_MODEL_CACHE[key] = model
    return _FID_MODEL_CACHE[key]


def to_fid_tensor(img):
    """
    img: 2D numpy array (grayscale PET slice). Returns a (3,299,299)
    tensor in [0,1], the format pytorch-fid's InceptionV3 wrapper
    expects (grayscale replicated to 3 channels, resized to Inception's
    native input size).
    """
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        norm = np.zeros_like(img)
    else:
        norm = (img - mn) / (mx - mn)

    t = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = t.repeat(1, 3, 1, 1)
    t = torch.nn.functional.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
    return t[0]  # (3,299,299)


@torch.no_grad()
def get_fid_features(images, device, batch_size=32):
    """
    images: list of 2D numpy arrays.
    Returns: (N, 2048) numpy array of Inception pool3 features.
    """
    model = get_fid_model(device)
    tensors = torch.stack([to_fid_tensor(img) for img in images])

    feats = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size].to(device)
        out = model(batch)[0]              # (B, 2048, 1, 1)
        out = out.squeeze(-1).squeeze(-1)  # (B, 2048)
        feats.append(out.cpu().numpy())

    return np.concatenate(feats, axis=0)


def compute_fid_from_features(real_feats, gen_feats):
    """
    Same Frechet distance computation as compute_fid, but takes
    already-extracted feature vectors directly -- lets you discard
    raw images right after extraction instead of holding them all
    in memory until a final compute_fid(...) call.
    """
    real_feats = np.stack(real_feats)
    gen_feats = np.stack(gen_feats)

    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(real_images, generated_images, device, batch_size=32):
    """
    real_images, generated_images: lists of 2D numpy arrays (grayscale
    PET slices).
    """
    real_feats = get_fid_features(real_images, device, batch_size)
    gen_feats = get_fid_features(generated_images, device, batch_size)
    return compute_fid_from_features(real_feats, gen_feats)