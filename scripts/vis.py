
import os
import cv2
import sys
import time
import random
import argparse
import numpy as np

import torch
from torchvision import transforms

sys.path.append("../")
sys.path.append("./")
from dataloader.bratsloader import BRATSDataset, BRATSDataset3D
from dataloader.amosloader import AMOSDataset, AMOSDataset3D
from dataloader.isicloader import ISICDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus
)

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


class SemanticSegmentationTarget:
    def __init__(self, category, mask, device):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.device = device
        if torch.cuda.is_available():
            self.mask = self.mask.to(self.device)
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

def main():
    args = create_argparser().parse_args()
    
    if "ema" in args.model_path:
        args.out_dir = os.path.join(args.out_dir, "cam" + "_epoch_ema_" \
            + str(int(os.path.basename(args.model_path).split('_')[2].rstrip('.pt'))))
    elif "opt" in args.model_path:
        args.out_dir = os.path.join(args.out_dir, "cam" + "_epoch_opt_" \
            + str(int(os.path.basename(args.model_path).lstrip("savedmodel").split('.')[0])))
    else:
        args.out_dir = os.path.join(args.out_dir, "cam" + "_epoch_" \
            + str(int(os.path.basename(args.model_path).lstrip("savedmodel").split('.')[0])))
    
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)
    
    debug_path = os.path.join(args.out_dir, "debug")
    if not os.path.isdir(debug_path): os.mkdir(debug_path)

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir,transform_test)
        args.in_ch = 5
    
    elif args.data_name == 'AMOS2D':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = AMOSDataset(args.data_dir, transform_test, phase="Va")
        args.in_ch = 4
        
    elif args.data_name == 'AMOS3D':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        ds = AMOSDataset3D(args.data_dir, transform_train, args.crop_size, phase="Va")
        args.in_ch = args.crop_size * 2
        
    datal = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    random.seed(int(time.time() % 9999))
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    device = torch.device("cuda:0")
    
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    # print(state_dict.keys())
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    target_layers = [model.output_blocks[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True, device=device)
    
    for img, mask, path in data:
        rgb_img = img.repeat(1, 3, 1, 1).squeeze(0).numpy()
        img = img.to(device)
        mask = mask.to(device)
        noise = torch.randn_like(img[:, :1, ...]).to(device)
        x_noisy = torch.cat((img, noise), 1)
        batch = {"noisy" : x_noisy,
                 "step" : int(args.diffusion_steps)}
        
        targets = [SemanticSegmentationTarget(-1, mask.squeeze(0).cpu().numpy(), device)]
        grayscale_cam = cam(input_tensor=batch, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        cv2.imwrite(os.path.join(args.out_dir, path), cam_image)
        print(target_layers)
        print(grayscale_cam.shape)
        print(batch.shape)

def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        crop_size=48,
        use_ddim=False,
        model_path="",         #path to pretrain model
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/',
        gpu_num = '0', #"0" # single gpu
        multi_gpu = None, #"0,1,2"
        debug = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


