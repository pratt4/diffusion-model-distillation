import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import zipfile
from tqdm import tqdm

def get_settings(data_to_use):
    if 'lsun' in data_to_use:
        is_bedroom = 'bedroom' in data_to_use
        data_to_use = 'lsun'
    settings = {
        "cifar10":
        {
         "adambeta1": 0.9,
         "adambeta2": 0.98,
         "adameps": 1e-8,
         "average_decay": 0.995,
         "lr": [2e-4, 5000],
         "epochs": 25,
         "shardsize": 102400,
         "n_test_examples": 51200,
         "batch_size": 512
        }, 
        "celeba":
        {
         "adambeta1": 0.9,
         "adambeta2": 0.98,
         "adameps": 1e-8,
         "average_decay": 0.995,
         "lr": [5e-5, 5000],
         "epochs": 35,
         "shardsize": 25600,
         "n_test_examples": 10240,
         "batch_size": 512
        }, 
        "lsun":
        {
         "adambeta1": 0.98,
         "adambeta2": 0.999,
         "adameps": 1e-8,
         "average_decay": 0.9995,
         "lr": [5e-6, 1000],
         "shardsize": 2560,
         "n_test_examples": 2560,
         "batch_size": 32
        }
    }
    if 'lsun' in data_to_use:
        if is_bedroom:
            settings['lsun']['epochs'] = 50
        else:
            settings['lsun']['epochs'] = 40
    return settings[data_to_use]

def show_images(images, scale=5, savepath=None, dims=None):
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    
    if not isinstance(images[0], np.ndarray):
        images = [float_to_image(image) for image in images]
    
    if dims is None:
        m = len(images) // 10 + 1
        n = 10
    else:
        m, n = dims

    plt.figure(figsize=(scale * n, scale * m))

    for i, image in enumerate(images):
        plt.subplot(m, n, i + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        
    plt.show()

def float_to_image(x):
    x = np.clip(x, -1.0, 1.0)
    x = (x * 127.5 + 127.5).astype(np.uint8)
    return x

def slerp(z1, z2, t):
    omega = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    a = torch.sin((1 - t) * omega) / torch.sin(omega)
    b = torch.sin(t * omega) / torch.sin(omega)
    return a * z1 + b * z2

class WarmupSchedule(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, learnrate, warmup_steps, last_epoch=-1):
        self.learnrate = learnrate
        self.warmup_steps = warmup_steps
        super(WarmupSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        arg1 = self.learnrate
        arg2 = step * self.learnrate / float(self.warmup_steps)
        return [min(arg1, arg2) for _ in self.base_lrs]

def get_strategy(devices_to_use=None):
    if devices_to_use is None:
        devices_to_use = 'cuda' if torch.cuda.is_available() else 'cpu'
    return devices_to_use

def make_runfn(model, run_ddim_process=False):
    if not run_ddim_process:
        def runmodel(z):
            return model(z)
    else:
        def runmodel(xt, index, alpha, alpha_next):
            return model.run_ddim_step(xt, index, alpha, alpha_next)
    
    return runmodel

def make_batch_of_images(model, runfn, z=None, n_samples=None):
    if n_samples is None and z is None:
        raise RuntimeError("Specify input or number of samples.")
    elif z is None:
        z = torch.randn(n_samples, model.spatialres, model.spatialres, 3)
    else:
        n_samples = z.shape[0]
    
    with torch.no_grad():
        images = runfn(z)
    images = [float_to_image(image) for image in images]
    return images

def load_models_from_gdrive(targetdir, get_original_models):
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    
    if get_original_models:
        zipped_loc = os.path.join(targetdir, "Original_Models.zip")
        drive_id = "1KlUuwAbWqHI0u9FXTb5xqSOIUnOMI7EV"
    else:
        zipped_loc = os.path.join(targetdir, "denoising_student_models.zip")
        drive_id = "1tW5t3W4wqE5f0NXaaiYuFK_2JOBrf9cY"

    download_file_from_google_drive(drive_id, zipped_loc)
    with zipfile.ZipFile(zipped_loc, "r") as zip_ref:
        zip_ref.extractall(targetdir)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: 
                f.write(chunk)
