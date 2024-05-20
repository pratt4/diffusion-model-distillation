import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os
from utils import get_strategy, make_runfn, float_to_image, show_images
from models import OnestepModel
from tqdm import tqdm

'''
    Note: when we ran our experiment for the CelebA dataset, we used a quadratic progression.
    This is different from what was used in [1]. [1] reported an FID of 6.53 when using a linear progression
    However, we found that using the quadratic progression led to an FID score closer to what was reported in [1].
    In our experiments, quadratic was 6.3 and linear was 7.1, so we decided to use the quadratic progression. 

    [1]: https://arxiv.org/abs/2010.02502
'''

def write_numpy_images(data_to_use: str, strategy, datadir: str, original_model_dir: str, batch_size: int, shardsize, num_test_examples, n_examples=1024000):
    
    if 'lsun' in data_to_use:
        use_quadratic = False
        num_timesteps = 50
    else:
        use_quadratic = True
        num_timesteps = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with strategy.scope(): 
        model = OnestepModel(data_to_use, os.path.join(original_model_dir, 'model_pytorch_%s.pth' % data_to_use)).to(device)
    
    get_xtm1 = make_runfn(model, device, run_ddim_process=True)

    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    res = model.spatialres

    def pyfunc(xtr):
        xtr = torch.randn(batch_size, 3, res, res).to(device)
        return xtr

    if not use_quadratic:
        seq = range(0, 1000,  1000//num_timesteps)
    else:
        seq = (np.linspace(0, np.sqrt(800), num_timesteps)**2).astype(int).tolist()

    seq_next = [-1] + seq[:-1]
    nshards = n_examples // shardsize + 1
    print(f"Creating {nshards} shards of {model.spatialres}^2 images on {len(seq)} steps")
    beta_set = torch.linspace(1e-4, 0.02, 1000).to(device)
    alpha_set = torch.cumprod(1 - beta_set, dim=0).to(device)
    starter = 0

    for shardnum in tqdm(range(nshards)):
        if shardnum == 0:
            if num_test_examples == 0:
                continue
            else:
                assert num_test_examples % batch_size == 0

            xtr = torch.arange(num_test_examples).to(device)
        else:
            xtr = torch.arange(shardsize).to(device)

        ds = torch.utils.data.TensorDataset(xtr)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

        X_TR = np.zeros((0, 3, res, res), dtype='float16')
        Y_TR = np.zeros((0, 3, res, res), dtype='uint8')
        for x in dl:
            inputs = pyfunc(x[0])
            bs = inputs.size(0)
            for i, j in zip(reversed(seq), reversed(seq_next)): 
                index = torch.full((bs,), i, dtype=torch.float32, device=device)
                alpha = torch.full((bs, 1, 1, 1), alpha_set[i].item(), device=device)
                alpha_next = torch.full((bs, 1, 1, 1), alpha_set[j].item() if j >= 0 else 1.0, device=device)
                beta = torch.full((bs, 1, 1, 1), beta_set[i].item(), device=device)

                inputs = get_xtm1(inputs, index, alpha, alpha_next)

            outputs = inputs
            if starter == 0:
                show_images(outputs[:6].cpu(), 5, savepath="./example_teacher_imgs.png")
                starter += 1

            inputs = inputs.cpu().numpy().astype('float16')
            outputs = float_to_image(outputs.cpu())
            X_TR = np.concatenate((X_TR, inputs), axis=0)
            Y_TR = np.concatenate((Y_TR, outputs), axis=0)
        
        if shardnum == 0:
            np.save(os.path.join(datadir, 'x_test.npy'), X_TR)
            np.save(os.path.join(datadir, 'y_test.npy'), Y_TR)
        else:
            np.save(os.path.join(datadir, f'x_train_{shardnum-1}.npy'), X_TR)
            np.save(os.path.join(datadir, f'y_train_{shardnum-1}.npy'), Y_TR)
        
        del dl
        del ds
