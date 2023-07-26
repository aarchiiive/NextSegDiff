

import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import glob
import random
import warnings
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from dataloader.bratsloader import BRATSDataset, BRATSDataset3D
from dataloader.amosloader import AMOSDataset, AMOSDataset3D
from dataloader.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple, dice_score
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
warnings.filterwarnings("ignore")

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def enhance(image, brightness_factor=2.0, contrast_factor=0.8):
    # 밝은 부분은 밝게, 어두운 부분은 어둡게 만드는 가중치를 적용합니다.
    enhanced_image = image * brightness_factor
    enhanced_image = th.clamp(enhanced_image, 0, 1)  # 픽셀값이 0과 1 사이를 벗어나지 않도록 조정

    # 어두운 부분을 어둡게 만들기 위해 이미지를 그레이스케일로 변환합니다.
    gray_image = th.mean(enhanced_image, dim=0, keepdim=True)

    # 어두운 부분에 대해 가중치를 적용하여 합칩니다.
    darken_image = gray_image * contrast_factor

    # 어두운 부분과 밝은 부분을 합하여 최종 결과 이미지를 생성합니다.
    result_image = th.where(gray_image > 0.5, enhanced_image, darken_image)

    return result_image



def main():
    args = create_argparser().parse_args()
    
    if "ema" in args.model_path:
        args.out_dir = os.path.join(args.out_dir, "diff_"+str(args.diffusion_steps) \
            + "_ens_"+str(args.num_ensemble) + "_epoch_ema_"+str(int(os.path.basename(args.model_path).split('_')[2].rstrip('.pt'))))
    elif "opt" in args.model_path:
        args.out_dir = os.path.join(args.out_dir, "diff_"+str(args.diffusion_steps) \
            + "_ens_"+str(args.num_ensemble) + "_epoch_opt_"+str(int(os.path.basename(args.model_path).lstrip("savedmodel").split('.')[0])))
    else:
        args.out_dir = os.path.join(args.out_dir, "diff_"+str(args.diffusion_steps) \
            + "_ens_"+str(args.num_ensemble) + "_epoch_"+str(int(os.path.basename(args.model_path).lstrip("savedmodel").split('.')[0])))
    
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
        
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    random.seed(int(time.time() % 9999))
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []
    
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
            
    # if args.multi_gpu:
    #     state_dict = dist_util.load_state_dict(args.model_path, map_location={f"cuda:{i}" : f"cuda:{i}" for i in args.multi_gpu.split(',')})
    #     for k, v in state_dict.items():
    #         # name = k[7:] # remove `module.`
    #         if 'module.' in k:
    #             new_state_dict[k] = v
    #             # load params
    #         else:
    #             new_state_dict = state_dict
    #         model.to(dist_util.dev())
    #         # state_dict = dist_util.load_state_dict(args.model_path, map_location="cuda:0")

    #     model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
    #     model.to(device = th.device('cuda', int(args.gpu_dev)))
    #     # state_dict = dist_util.load_state_dict(args.model_path, map_location={f"cuda:{i}" : f"cuda:{i}" for i in args.multi_gpu.split(',')})
    # else:
    #     state_dict = dist_util.load_state_dict(args.model_path, map_location="cuda:0")
    #     for k, v in state_dict.items():
    #         # name = k[7:] # remove `module.`
    #         if 'module.' in k:
    #             new_state_dict[k[7:]] = v
    #             # load params
    #         else:
    #             new_state_dict = state_dict
    #         model.to(dist_util.dev())
    #         # state_dict = dist_util.load_state_dict(args.model_path, map_location="cuda:0")

    # model.load_state_dict(new_state_dict)
    
    device = th.device(f"cuda:{args.gpu_num}")
    
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
    for _ in range(len(data)):
        mask_ch = args.crop_size if args.data_name == 'AMOS3D' else 1
        b, m, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :mask_ch, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        if args.data_name == 'ISIC':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
        elif args.data_name == 'AMOS2D':
            print(path)
            slice_ID = path[0].split('.')[0]
        elif args.data_name == 'AMOS3D':
            print(path)
            slice_ID = path[0].split('.')[0]
            
        # print(path)
        # if not "0207_slice_259" in path[0]: continue
        if len(glob.glob(os.path.join(args.out_dir, str(slice_ID)+"*.jpg"))): continue
                          
        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            print(f'Ensemble step {i} started...')
            t0 = time.time()
            ch = args.crop_size if args.data_name == 'AMOS3D' else 3
            model_kwargs = {}
            start.record()
            # diffusion.dpm_solver = True
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, ch, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                # device=device,
            )

            # tensor2pil = transforms.ToPILImage()
            # s = tensor2pil(sample.squeeze()) 
            # o = tensor2pil(org[:,-1,:,:].squeeze() / org[:,-1,:,:].max())
            # c = tensor2pil(cal.squeeze())
            # c_out = tensor2pil(cal_out.squeeze())
            
            # s.save(f"{slice_ID}_s.png")
            # for i in range(org.size()[1]):
            #     tensor2pil(org[:,i,:,:].squeeze() / org[:,i,:,:].max()).save(f"{slice_ID}_o_{i}.png")
            # c.save(f"{slice_ID}_c.png")
            # c_out.save(f"{slice_ID}_c_out.png")
            
            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            co = th.tensor(cal_out) # input : [1, 48, 256, 256] / predicted mask? : [1, 1, 256, 256]
            
            enslist.append(co)
            
            if args.debug:
                if args.data_name == 'ISIC' or 'AMOS2D':
                    # s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[:,:-1,:,:]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    m = th.tensor(m[:,0,:,:].to(device = f"cuda:{args.gpu_num}")).unsqueeze(1).repeat(1, 3, 1, 1)
                    co = co.repeat(1, 3, 1, 1)

                    s = sample[:,-1,:,:]
                    b,h,w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                    tup = (ss,o,m,c,co)
                elif args.data_name == 'BRATS':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = f"cuda:{args.gpu_num}"))[:,0,:,:].unsqueeze(1)
                    # input 4개라서 o1 o2 o3 o4 48개니까 
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)
                
                elif args.data_name == 'AMOS':
                    pass

                compose = th.cat(tup,0)
                vutils.save_image(compose, fp = os.path.join(debug_path, str(slice_ID)+'_output'+str(i)+".jpg"), nrow = 1, padding = 10)
                
            print(f'Time : {time.time() - t0:.2f}s')
            th.cuda.empty_cache()
        
        ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        ensres = enhance(ensres)
        dice = dice_score(ensres, m)
        vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+f'_output_ens_{dice:.4f}'+".jpg"), nrow = 1, padding = 10)
        
        compared = th.cat((o, ensres.repeat(1, 3, 1, 1), m), 3)
        vutils.save_image(compared, fp = os.path.join(args.out_dir, str(slice_ID)+f'_compared_{dice:.4f}'+".jpg"), nrow = 1, padding = 10)
        
        
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
