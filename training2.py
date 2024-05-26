import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, update_bn

# Assuming Onestep_Model and other utilities are defined in models and utils modules
from models import Onestep_Model
from utils import WarmupSchedule, show_images, get_strategy, float_to_image, load_models_from_gdrive, get_settings

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_dataset(datadir, shardnum, batch_size):
    if shardnum == -1:
        X = np.load(os.path.join(datadir, 'x_test.npy'))
        Y = np.load(os.path.join(datadir, 'y_test.npy'))
    else:
        X = np.load(os.path.join(datadir, f'x_train_{shardnum}.npy'))
        Y = np.load(os.path.join(datadir, f'y_train_{shardnum}.npy'))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CustomDataset(X, Y, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def make_training_objects(model, optimizer, device, use_l2):
    if use_l2:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.L1Loss()

    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        pred_y = model(x)
        loss = loss_fn(pred_y, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def test_step(x, y):
        model.eval()
        with torch.no_grad():
            pred_y = model(x)
            loss = loss_fn(pred_y, y)
        return loss.item()

    return train_step, test_step

def get_test_loss(test_loader, test_step, device):
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            loss = test_step(x, y)
            test_loss += loss * x.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}")
    return test_loss

def train(data_to_use, savedir, datadir, original_model_dir, devices_to_use, batch_size, use_xla, use_fewer_lsun_examples=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_to_use = data_to_use.lower()
    assert data_to_use in ["cifar10", "celeba", "lsun_church", "lsun_bedroom"]

    os.makedirs(savedir, exist_ok=True)

    if original_model_dir == "download_from_web":
        original_model_dir = './Original_Models'
        load_models_from_gdrive("./", True)

    s = get_settings(data_to_use)
    if batch_size == 'default':
        batch_size = s["batch_size"]
    else:
        batch_size = int(batch_size)
    adambeta1 = s["adambeta1"]
    adambeta2 = s["adambeta2"]
    adameps = s["adameps"]
    average_decay = s["average_decay"]
    learnrate, warmupsteps = s["lr"]
    epochs = s["epochs"]

    use_l2 = data_to_use != "cifar10"

    if 'lsun' in data_to_use and use_fewer_lsun_examples:
        n_examples = 204800
    else:
        n_examples = 1024000

    if datadir == "create":
        datadir = "./datasets"
        from create_dataset import write_numpy_images
        n_test_examples = s["n_test_examples"]
        shardsize = s["shardsize"]
        write_numpy_images(data_to_use=data_to_use, datadir=datadir, 
                           original_model_dir=original_model_dir, shardsize=shardsize,
                           batch_size=2*batch_size, num_test_examples=n_test_examples, n_examples=n_examples)
    elif not os.path.isdir(datadir) or len(os.listdir(datadir)) == 0:
        raise RuntimeError("Data directory not found or empty")
    else:
        print(f"Using the dataset in {datadir}")

    os.makedirs("./model_samples", exist_ok=True)

    test_loader = load_dataset(datadir, -1, batch_size)

    save_path = os.path.join(savedir, f'{data_to_use}_nonema_model.pth')
    opt_path = os.path.join(savedir, f'{data_to_use}_optimizer.p')

    model = Onestep_Model(data_to_use, os.path.join(original_model_dir, f'model_tf2_{data_to_use}.h5')).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learnrate, betas=(adambeta1, adambeta2), eps=adameps)

    # Set up SWA from the start of SWA application
    swa_model = AveragedModel(model)
    swa_start = 10  # Start averaging after SWA_START epochs

    train_step, test_step = make_training_objects(model, optimizer, device, use_l2)

    if os.path.isfile(save_path) and os.path.isfile(opt_path):
        continue_training = input("There is an existing model and optimizer. Continue training from here? y/n")
        if continue_training.lower() == 'y':
            model.load_state_dict(torch.load(save_path))

            with open(opt_path, 'rb') as f:
                opt_weights = pickle.load(f)
            
            optimizer.load_state_dict(opt_weights)

    print(f"Current Optimizer iterations: {optimizer.state_dict()['state'][next(iter(optimizer.state_dict()['state']))]['step']}")

    lowest_ema_loss = float('inf')
    nshards = len(os.listdir(datadir)) // 2 - 1
    print(f"Total number of shards is {nshards} shards")

    start_time = time()

    for ep in range(epochs):
        for i, shard in enumerate(tqdm(range(nshards))):
            train_loader = load_dataset(datadir, shard, batch_size)
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                train_step(x, y)

                # Update the SWA model parameters after SWA_START
                if ep >= swa_start:
                    swa_model.update_parameters(model)

            if (i + 1) % 80 == 0 or i == nshards - 1:
                print(f"Optimizer iterations {optimizer.state_dict()['state'][next(iter(optimizer.state_dict()['state']))]['step']},  Time {time() - start_time:.2f}s")

                torch.save(model.state_dict(), save_path)
                with open(opt_path, 'wb') as f:
                    pickle.dump(optimizer.state_dict(), f, pickle.HIGHEST_PROTOCOL)

                # Update batch normalization statistics for the SWA model
                update_bn(train_loader, swa_model)

                with torch.no_grad():
                    model.eval()
                    images = model(torch.randn(6, model.spatialres, model.spatialres, 3).to(device))
                
                show_images(images, 5, savepath=f"./model_samples/samples_at_it_{optimizer.state_dict()['state'][next(iter(optimizer.state_dict()['state']))]['step']}.png")

                ema_tl = get_test_loss(test_loader, test_step, device)
                if ema_tl < lowest_ema_loss:
                    print("Overwriting EMA model...")
                    torch.save(swa_model.state_dict(), ema_save_path)
                    lowest_ema_loss = ema_tl

                model.load_state_dict(torch.load(save_path))

    print(f"Training is completed. The name of the final trained model is {ema_save_path}")
    print(f"The model was trained for a maximum of {optimizer.state_dict()['state'][next(iter(optimizer.state_dict()['state']))]['step']} iterations")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_to_use", type=str, help="which model to retrain. should be one of ['cifar10', 'celeba', 'lsun_bedroom', 'lsun_church']")
    parser.add_argument("savedir", type=str, help="the directory to save model to.")
    parser.add_argument("--datadir", type=str, default="create", help="the directory where the data is located. by default it will be created in ./datasets")
    parser.add_argument("--original_model_dir", type=str, default="download_from_web", help="the directory where the original models are located. by default it will get them from the web")
    parser.add_argument("--devices", nargs="*", default=[], help="which devices to train on.")
    parser.add_argument("--batch_size", default='default', help="batch size to train model on. recommend keeping at default, or else reproducibility may be affected.")
    parser.add_argument("--xla", action='store_false', help="whether to use XLA, True/False.")
    parser.add_argument("--use_fewer_lsun_examples", action='store_true', help="when training LSUN, whether to use all 1.024M examples. If you are short on disk space, set to True. This may affect reproducibility if set to True.")
    args = parser.parse_args()
    
    train(args.data_to_use, args.savedir, args.datadir, args.original_model_dir, args.devices, args.batch_size, args.xla)
