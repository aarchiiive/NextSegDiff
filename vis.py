import cv2
import os
import glob
import numpy as np
from tqdm import tqdm


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == "__main__":
    path_name = "dataset/AMOS2D/preds/diff_50_ens_5_epoch_13100"
    heatmap_path = os.path.join(path_name, "heatmap")
    if not os.path.isfile(heatmap_path): os.makedirs(heatmap_path, exist_ok=True)
    masks = glob.glob(os.path.join(path_name, "*.jpg"))
    
    image_path = "dataset/AMOS2D/imagesVa"
    
    for m in tqdm(masks):
        if "compared" in m: continue
        # img_name = os.path.basename(m).
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cv2.imread(m, cv2.IMREAD_GRAYSCALE)), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) 
        # heatmap = 0.5 * heatmap / np.max(heatmap) * 255
        # heatmap = np.float32(heatmap) / 255
        cv2.imwrite(os.path.join(heatmap_path, os.path.basename(m)), heatmap)
