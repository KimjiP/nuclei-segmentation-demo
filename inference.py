"""Inference helpers shared by the notebook, the Gradio app and serve.py.

Kept in one place so the web app and the API run exactly the code that was
evaluated in the notebook.
"""
import numpy as np
from scipy import ndimage as ndi

# ImageNet statistics, because the encoder was pretrained on ImageNet.
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_rgb(image):
    """Accept gray, RGB or RGBA uint8 images and return RGB uint8."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def for_display(rgb):
    """Brighten dark (fluorescent) images for viewing. The model always sees the original."""
    rgb = to_rgb(rgb)
    if rgb.mean() > 60:
        return rgb
    top = max(float(np.percentile(rgb, 99.5)), 1.0)
    return np.clip(rgb.astype(np.float32) * (255.0 / top), 0, 255).astype(np.uint8)


def normalize(rgb):
    """uint8 HxWx3 -> float32 3xHxW, scaled the way the encoder expects."""
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def preprocess(rgb):
    """Pad to a multiple of 32 (the U-Net needs it) and normalize.

    Returns a 1x3xHxW float32 batch and the original (height, width),
    so the output can be cropped back to the input size.
    """
    h, w = rgb.shape[:2]
    padded = np.pad(rgb, ((0, (-h) % 32), (0, (-w) % 32), (0, 0)), mode="reflect")
    return np.ascontiguousarray(normalize(padded)[None]), (h, w)


def predict_prob_onnx(session, rgb):
    """Probability map (HxW, 0..1) from an ONNX Runtime session."""
    x, (h, w) = preprocess(to_rgb(rgb))
    logits = session.run(None, {session.get_inputs()[0].name: x})[0]
    return 1.0 / (1.0 + np.exp(-logits[0, 0, :h, :w]))


def split_nuclei(mask, min_size=15):
    """Split a binary mask into separate nuclei and count them.

    Touching nuclei form one blob in the mask. The distance to the blob edge
    peaks near each nucleus center, so the peaks become seeds and a watershed
    grows each seed back out to the blob edge.
    Returns an int32 label image (0 = background) and the number of nuclei.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(mask.shape, np.int32), 0
    dist = ndi.distance_transform_edt(mask)
    smooth = ndi.gaussian_filter(dist, 1.0)
    peaks = (smooth == ndi.maximum_filter(smooth, size=7)) & (smooth > 2.0) & mask
    markers, n = ndi.label(peaks, structure=np.ones((3, 3)))
    # Small blobs can be too thin to have a peak; each still counts as one nucleus.
    blobs, n_blobs = ndi.label(mask)
    boxes = ndi.find_objects(blobs)
    seeded = np.unique(blobs[markers > 0])
    for blob in np.setdiff1d(np.arange(1, n_blobs + 1), seeded):
        n += 1
        box = boxes[blob - 1]
        markers[box][blobs[box] == blob] = n
    # Flood from the seeds over the inverted distance. The background costs the most,
    # so it is reached last and never takes pixels away from a nucleus.
    relief = np.full(mask.shape, 65535, np.uint16)
    relief[mask] = (65534 * (1 - dist[mask] / dist.max())).astype(np.uint16)
    labels = ndi.watershed_ift(relief, markers.astype(np.int32))
    labels[~mask] = 0
    # Drop specks left by the flooding.
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    labels[sizes[labels] < min_size] = 0
    kept = np.unique(labels[labels > 0])
    relabel = np.zeros(n + 1, np.int32)
    relabel[kept] = np.arange(1, len(kept) + 1)
    return relabel[labels], len(kept)


def outline(labels, thick=False):
    """Boolean image of the edge pixels of every labeled region."""
    lab = np.asarray(labels)
    edge = np.zeros(lab.shape, bool)
    edge[:-1, :] |= lab[:-1, :] != lab[1:, :]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge &= lab > 0
    if thick:
        edge = ndi.binary_dilation(edge)
    return edge


def draw_nuclei(rgb, labels, color=(235, 104, 52), min_side=640):
    """App view: enlarge small images, tint each nucleus and outline it, so it reads on screen."""
    view, labels = for_display(rgb), np.asarray(labels)
    scale = max(1, int(np.ceil(min_side / max(view.shape[:2]))))
    if scale > 1:
        view = np.repeat(np.repeat(view, scale, axis=0), scale, axis=1)
        labels = np.repeat(np.repeat(labels, scale, axis=0), scale, axis=1)
    out = view.astype(np.float32)
    inside = labels > 0
    out[inside] = 0.65 * out[inside] + 0.35 * np.array(color, np.float32)
    out[outline(labels, thick=True)] = color
    return out.astype(np.uint8)


def draw_outlines(rgb, labels, color=(235, 104, 52), thick=None):
    """Copy of the image with each nucleus outlined in one color."""
    out = to_rgb(rgb).copy()
    if thick is None:
        thick = max(out.shape[:2]) > 600
    out[outline(labels, thick)] = color
    return out
