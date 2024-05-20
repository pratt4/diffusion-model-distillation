import torch
import matplotlib.pyplot as plt
import numpy as np
from imageio import imwrite
from tqdm import tqdm
import os
from utils import show_images, slerp, make_batch_of_images, float_to_image, load_models_from_gdrive
from models import OnestepModel

def interpolation_experiment(model, device, n_images=11, savepath=None):
    tset = [i/(n_images-1) for i in range(n_images)]
    assert min(tset) == 0. and max(tset) == 1.
    z1, z2 = torch.split(torch.randn(2, 3, model.spatialres, model.spatialres), 1)
    z_in = torch.cat([slerp(z1, z2, tset[i]) for i in range(n_images)], dim=0)

    with torch.no_grad():
        model.to(device)
        z_in = z_in.to(device)
        images = model(z_in)

    show_images(images.cpu(), dims=[1, n_images], savepath=savepath)

def get_model_images(model, device, bs):
    z = torch.randn(bs, 3, model.spatialres, model.spatialres).to(device)
    with torch.no_grad():
        images = model(z)
    return images.cpu()

def get_model(data_to_use, denoising_student_dir):
    model = OnestepModel(data_to_use, None)
    model(torch.randn(1, 3, model.spatialres, model.spatialres))
    model.load_state_dict(torch.load(os.path.join(denoising_student_dir, f'{data_to_use}_ema_model.pth')))
    return model

def write_images_to_folder(model, device, write_dir, batch_size=20, n_images=20):
    if write_dir is not None:
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)

    n_batches = n_images // batch_size
    remaining_samples = n_images - batch_size * n_batches
    n_batches += 1
    n_written = 0

    for i in tqdm(range(n_batches)):
        if i == n_batches - 1:
            bs = remaining_samples
        else:
            bs = batch_size
        if bs == 0:
            continue
        images = get_model_images(model, device, bs)
        images = float_to_image(images)

        if write_dir is not None:
            for img in images:
                imgpath = os.path.join(write_dir, f'images{str(n_written)}.png')
                imwrite(imgpath, img)
                n_written += 1

    return n_written == n_images

def get_uncurated_samples(data_to_use, model_dir, savedir, device, n_images):
    model = get_model(data_to_use, model_dir)
    images = get_model_images(model, device, n_images)
    savepath = os.path.join(savedir, f'{data_to_use}_figure_{len(os.listdir(savedir))}.png')
    dims = [int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images)))]
    scale = min(model.spatialres, 192) // 32

    if os.path.isfile(savepath):
        print("There is a file here already. It will be overwritten.")
    show_images(images, scale=scale, savepath=savepath, dims=dims)
    return True

def main(action, savedir, data_to_use, n_images, model_dir, batch_size):
    if model_dir == "download_from_web":
        model_dir = './denoising_student_models'
        if not os.path.exists(model_dir):
            load_models_from_gdrive("./", False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device {device}")

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if action == 'figures':
        status = get_uncurated_samples(data_to_use, model_dir, savedir, device, n_images)
    elif action == 'tofolder':
        model = get_model(data_to_use, model_dir)
        status = write_images_to_folder(model, device, savedir, batch_size, n_images)
    else:
        raise NotImplementedError("Action must be 'figures' or 'tofolder'.")

    if status:
        print("Finished execution properly.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default='figures', help="What action to do. Should be either 'figures' or 'tofolder'. 'figures' option will create a square figure of images. 'tofolder' option will write each image to a file.")
    parser.add_argument("savedir", type=str, help="The directory to save outputs to.")
    parser.add_argument("data_to_use", type=str, help="Which dataset's images to write. Should be one of ['cifar10', 'celeba', 'lsun_bedroom', 'lsun_church']")
    parser.add_argument("--n_images", type=int, default=20, help="How many images to write.")
    parser.add_argument("--model_dir", type=str, default="download_from_web", help="The directory where the denoising_student_models are located. By default it will get them from the web.")
    parser.add_argument("--batch_size", type=int, default=20, help="When using 'tofolder', batch size to run examples on.")
    
    args = parser.parse_args()
    
    main(args.action, args.savedir, args.data_to_use, args.n_images, args.model_dir, args.batch_size)
