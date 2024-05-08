import os
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format
from elevation_estimate.estimate_wild_imgs import estimate_elev #, estimate_elev_jinbin


import numpy as np
from contextlib import nullcontext
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from rich import print
from torch import autocast
from torchvision import transforms
import itertools

def predict_stage1_gradio_xy(model, raw_im, save_path="", adjust_set=[], device="cuda", ddim_steps=75, scale=3.0, delta_x_list=[0], delta_y_list=[0]):
    input_im_init = np.asarray(raw_im, dtype=np.float32) / 255.0
    input_im = transforms.ToTensor()(input_im_init).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1

    ret_imgs = []
    sampler = DDIMSampler(model)


    delta_combinations = list(itertools.product(delta_x_list, delta_y_list))
    delta_x_values, delta_y_values = zip(*delta_combinations)

    x_samples_ddims_8 = sample_model_batch_xy(
        model, sampler, input_im, xs=delta_x_values, ys=delta_y_values,
        n_samples=len(delta_x_values), ddim_steps=ddim_steps, scale=scale)

    sample_idx = 0
    for stage1_idx in range(len(x_samples_ddims_8)):
        x_sample = 255.0 * rearrange(x_samples_ddims_8[sample_idx].numpy(), 'c h w -> h w c')
        out_image = Image.fromarray(x_sample.astype(np.uint8))
        ret_imgs.append(out_image)
        if save_path:
            delta_x, delta_y = delta_combinations[sample_idx]
            out_image.save(os.path.join(save_path, f'{delta_x}_{delta_y}.png'))
        sample_idx += 1
    del x_samples_ddims_8
    del sampler
    torch.cuda.empty_cache()
    return ret_imgs

def sample_model_batch_xy(model, sampler, input_im, xs, ys, n_samples=4, precision='autocast', ddim_eta=1.0, ddim_steps=75, scale=3.0, h=256, w=256):
    precision_scope = autocast if precision == 'autocast' else nullcontext

    ret_imgs = []
    batch_size = 64  # 每批次处理的数量
    for i in range(0, len(xs), batch_size):
        xs_batch = xs[i:i + batch_size]
        ys_batch = ys[i:i + batch_size]
        n_samples = len(xs_batch)

        with precision_scope("cuda"):
            with model.ema_scope():
                c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
                T = []
                for x, y in zip(xs_batch, ys_batch):
                    T.append([np.radians(x), np.sin(np.radians(y)), np.cos(np.radians(y)), 0])
                T = torch.tensor(np.array(T))[:, None, :].float().to(c.device)
                c = torch.cat([c, T], dim=-1)
                c = model.cc_projection(c)
                cond = {}
                cond['c_crossattn'] = [c]
                cond['c_concat'] = [model.encode_first_stage(input_im).mode().detach()
                                    .repeat(n_samples, 1, 1, 1)]
                if scale != 1.0:
                    uc = {}
                    uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                    uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
                else:
                    uc = None

                shape = [4, h // 8, w // 8]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=cond,
                                                batch_size=n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                x_T=None)
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                ret_imgs_batch = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                ret_imgs.extend(ret_imgs_batch)

    del cond, c, x_samples_ddim, samples_ddim, uc, input_im
    torch.cuda.empty_cache()
            
    return ret_imgs


def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256

def stage1_run(model, device, exp_dir,
               input_im, scale, ddim_steps):
    # folder to save the stage 1 images
    stage1_dir = os.path.join(exp_dir, "stage1_8")
    os.makedirs(stage1_dir, exist_ok=True)

    # stage 1: generate 4 views at the same elevation as the input
    output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)
    
    # stage 2 for the first image
    # infer 4 nearby views for an image to estimate the polar angle of the input
    stage2_steps = 50 # ddim_steps
    zero123_infer(model, exp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)
    # estimate the camera pose (elevation) of the input image.
    try:
        polar_angle = estimate_elev(exp_dir)
    except:
        print("Failed to estimate polar angle")
        polar_angle = 90
    print("Estimated polar angle:", polar_angle)
    gen_poses(exp_dir, polar_angle)
    # print(polar_angle) # Estimated polar angle: 78
    # stage 1: generate another 4 views at a different elevation
    if polar_angle <= 75:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4,8)), device=device, ddim_steps=ddim_steps, scale=scale)
    else:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(8,12)), device=device, ddim_steps=ddim_steps, scale=scale)
    torch.cuda.empty_cache()
    return 90-polar_angle, output_ims+output_ims_2
    
def stage2_run(model, device, exp_dir,
               elev, scale, stage2_steps=50):
    # stage 2 for the remaining 7 images, generate 7*4=28 views
    if 90-elev <= 75:
        zero123_infer(model, exp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        zero123_infer(model, exp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)

def reconstruct(exp_dir, output_format=".ply", device_idx=0, resolution=256):
    exp_dir = os.path.abspath(exp_dir)
    main_dir_path = os.path.abspath(os.path.dirname("./"))
    os.chdir('reconstruction/')

    bash_script = f'CUDA_VISIBLE_DEVICES={device_idx} python exp_runner_generic_blender_val.py \
                    --specific_dataset_name {exp_dir} \
                    --mode export_mesh \
                    --conf confs/one2345_lod0_val_demo.conf \
                    --resolution {resolution}'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(exp_dir, f"mesh.ply")
    if output_format == ".ply":
        return ply_path
    if output_format not in [".obj", ".glb"]:
        print("Invalid output format, must be one of .ply, .obj, .glb")
        return ply_path
    return convert_mesh_format(exp_dir, output_format=output_format)


def predict_multiview(shape_dir, args):
    device = f"cuda:{args.gpu_idx}"

    # initialize the zero123 model
    models = init_model(device, 'zero123-xl.ckpt', half_precision=args.half_precision)
    model_zero123 = models["turncam"]

    # initialize the Segment Anything model
    predictor = sam_init(args.gpu_idx)
    input_raw = Image.open(args.img_path)

    # preprocess the input image
    input_256 = preprocess(predictor, input_raw)

    # # generate multi-view images in two stages with Zero123.
    # # first stage: generate N=8 views cover 360 degree of the input shape.
    # elev, stage1_imgs = stage1_run(model_zero123, device, shape_dir, input_256, scale=3, ddim_steps=75)
    # # second stage: 4 local views for each of the first-stage view, resulting in N*4=32 source view images.
    # stage2_run(model_zero123, device, shape_dir, elev, scale=3, stage2_steps=50)


    ######################################################################################################
    # generate the specifical view image
    model, device, exp_dir, input_im, scale, ddim_steps = model_zero123, device, shape_dir, input_256, 3, 75

    # folder to save the stage 1 images
    stage1_dir = os.path.join(exp_dir, "stage1_8")
    os.makedirs(stage1_dir, exist_ok=True)

    # stage 1: generate 4 views at the same elevation as the input
    # output_ims = predict_stage1_gradio_xy(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale,delta_x = args.x,delta_y=args.y)

    #prepare for the dataset
    output_ims = predict_stage1_gradio_xy(model, input_im, save_path=args.output_path, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale, delta_x_list = args.x, delta_y_list=args.y)

    return output_ims


# python pose_synthesis.py --img_path ./imgs/img.png --half_precision
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # Positive x means the camera angle from bottom to top. Negative x means the camera angle from top to bottom.
    # Positive y means the camera angle from right to left. Negative y means the camera angle from left to right.

    import argparse
    import ast

    def parse_int_list(s):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [int(s)]

    parser.add_argument('--x', type=parse_int_list, default=[0], help='List of x values for the generated images')
    parser.add_argument('--y', type=parse_int_list, default=[45], help='List of y values for the generated images')

    parser.add_argument('--img_path', type=str, default="./demo/demo_examples/01_wild_hydrant.png", help='Path to the input image')
    parser.add_argument('--output_path', type=str, default="./demo/output/", help='Path to the output image')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision')
    parser.add_argument('--mesh_resolution', type=int, default=256, help='Mesh resolution')
    parser.add_argument('--output_format', type=str, default=".ply", help='Output format: .ply, .obj, .glb')

    args = parser.parse_args()

    assert(torch.cuda.is_available())

    shape_id = args.img_path.split('/')[-1].split('.')[0]
    shape_dir = f"./exp/{shape_id}"
    os.makedirs(shape_dir, exist_ok=True)

    output_ims = predict_multiview(shape_dir, args)