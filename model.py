import torch
import torch.nn as nn
import torch.nn.functional as F
from nn import get_timestep_embedding, Downsample, Upsample, ResnetBlock, AttnBlock

class DDPMCheckpointModel(nn.Module):
    def __init__(self, data_to_use, c=128):
        super(DDPMCheckpointModel, self).__init__()
        self.data_to_use = data_to_use
        if data_to_use == 'cifar10':
            self.spatialres = 32
            chmul = [1, 2, 2, 2]
        elif 'lsun' in self.data_to_use:
            self.spatialres = 256
            chmul = [1, 1, 2, 2, 4, 4]
        else:
            raise NotImplementedError("dataset must be either cifar10, lsun_church. To load the Celeba model, instantiate a DDIMCheckpointModel object.")

        self.c = c

        self.conv_in = nn.Conv2d(3, c, 3, padding=1)
        self.conv_out = nn.Conv2d(c, 3, 3, padding=1)

        self.down_0 = nn.ModuleList([ResnetBlock(c*chmul[0], was_pytorch=False), ResnetBlock(c*chmul[0], was_pytorch=False), Downsample(c*chmul[0], with_conv=True)])
        if self.spatialres == 32:
            self.attnsdown = nn.ModuleList([AttnBlock(c*chmul[1], was_pytorch=False), AttnBlock(c*chmul[1], was_pytorch=False)])
        self.down_1 = nn.ModuleList([ResnetBlock(c*chmul[1], use_nin_shortcut=(self.spatialres==32), was_pytorch=False), ResnetBlock(c*chmul[1], was_pytorch=False), Downsample(c*chmul[1], with_conv=True)])
        self.down_2 = nn.ModuleList([ResnetBlock(c*chmul[2], use_nin_shortcut=(self.spatialres==256), was_pytorch=False), ResnetBlock(c*chmul[2], was_pytorch=False), Downsample(c*chmul[2], with_conv=True)])
        self.down_3 = nn.ModuleList([ResnetBlock(c*chmul[3], was_pytorch=False), ResnetBlock(c*chmul[3], was_pytorch=False)])
        if self.spatialres == 256:
            self.down_3.append(Downsample(c*chmul[3], with_conv=True))
            self.attnsdown = nn.ModuleList([AttnBlock(c*chmul[4], was_pytorch=False), AttnBlock(c*chmul[4], was_pytorch=False)])
            self.down_4 = nn.ModuleList([ResnetBlock(c*chmul[4], use_nin_shortcut=True, was_pytorch=False), ResnetBlock(c*chmul[4], was_pytorch=False), Downsample(c*chmul[4], with_conv=True)])
            self.down_5 = nn.ModuleList([ResnetBlock(c*chmul[5], was_pytorch=False), ResnetBlock(c*chmul[5], was_pytorch=False)])

        self.mid1 = AttnBlock(c*chmul[-1], was_pytorch=False)
        self.mid0 = ResnetBlock(c*chmul[-1], was_pytorch=False)
        self.mid2 = ResnetBlock(c*chmul[-1], was_pytorch=False)

        self.norm_out = nn.GroupNorm(32, c*chmul[0])
        self.temb = nn.ModuleList([nn.Linear(c, c*4), nn.Linear(c*4, c*4)])

        self.up_0 = nn.ModuleList([ResnetBlock(c*chmul[0], use_nin_shortcut=True, was_pytorch=False) for _ in range(3)])
        if self.spatialres == 32:
            self.attnsup = nn.ModuleList([AttnBlock(c*chmul[1], was_pytorch=False) for _ in range(3)])
        self.up_1 = nn.ModuleList([ResnetBlock(c*chmul[1], use_nin_shortcut=True, was_pytorch=False) for _ in range(3)] + [Upsample(c*chmul[1], with_conv=True)])
        self.up_2 = nn.ModuleList([ResnetBlock(c*chmul[2], use_nin_shortcut=True, was_pytorch=False) for _ in range(3)] + [Upsample(c*chmul[2], with_conv=True)])
        self.up_3 = nn.ModuleList([ResnetBlock(c*chmul[3], use_nin_shortcut=True, was_pytorch=False) for _ in range(3)] + [Upsample(c*chmul[3], with_conv=True)])
        if self.spatialres == 256:
            self.attnsup = nn.ModuleList([AttnBlock(c*chmul[4], was_pytorch=False) for _ in range(3)])
            self.up_4 = nn.ModuleList([ResnetBlock(c*chmul[4], use_nin_shortcut=True, was_pytorch=False) for _ in range(3)] + [Upsample(c*chmul[4], with_conv=True)])
            self.up_5 = nn.ModuleList([ResnetBlock(c*chmul[5], use_nin_shortcut=True, was_pytorch=False) for _ in range(3)] + [Upsample(c*chmul[5], with_conv=True)])

    def get_pretrained_weights(self, weights_path):
        # Note: model name must be specific. The model name must be model_tf2_{DATASET}.h5 e.g. model_tf2_lsun_church.h5
        self(torch.randn(1, 3, self.spatialres, self.spatialres), torch.ones(1))  # builds model.
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x, index):
        index = get_timestep_embedding(index, self.c)
        index = F.silu(self.temb[0](index))
        index = self.temb[1](index)

        x = self.conv_in(x)
        residuals = [x]

        for block in self.down_0:
            x = block(x, index)
            residuals.append(x)

        if self.spatialres == 32:
            for i, block in enumerate(self.down_1):
                x = block(x, index)
                if i < 2: x = self.attnsdown[i](x, index)
                residuals.append(x)
        else:
            for block in self.down_1:
                x = block(x, index)
                residuals.append(x)

        for block in self.down_2:
            x = block(x, index)
            residuals.append(x)

        for block in self.down_3:
            x = block(x, index)
            residuals.append(x)

        if self.spatialres == 256:
            for i, block in enumerate(self.down_4):
                x = block(x, index)
                if i < 2: x = self.attnsdown[i](x, index)
                residuals.append(x)

            for block in self.down_5:
                x = block(x, index)
                residuals.append(x)

        x = self.mid0(x, index)
        x = self.mid1(x, index)
        x = self.mid2(x, index)

        if self.spatialres == 256:
            for i, block in enumerate(self.up_5):
                if i < 3:
                    x = torch.cat([x, residuals.pop()], dim=1)
                x = block(x, index)

            for i, block in enumerate(self.up_4):
                if i < 3:
                    x = torch.cat([x, residuals.pop()], dim=1)
                x = block(x, index)
                if i < 3:
                    x = self.attnsup[i](x, index)

        for i, block in enumerate(self.up_3):
            if i < 3:
                x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, index)

        for i, block in enumerate(self.up_2):
            if i < 3:
                x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, index)

        if self.spatialres == 32:
            for i, block in enumerate(self.up_1):
                if i < 3:
                    x = torch.cat([x, residuals.pop()], dim=1)
                x = block(x, index)
                if i < 3:
                    x = self.attnsup[i](x, index)
        else:
            for i, block in enumerate(self.up_1):
                if i < 3:
                    x = torch.cat([x, residuals.pop()], dim=1)
                x = block(x, index)

        for i, block in enumerate(self.up_0):
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, index)

        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)

        return x

class DDIMCheckpointModel(nn.Module):
    def __init__(self, c=128):
        super(DDIMCheckpointModel, self).__init__()
        chmul = [1, 2, 2, 2, 4]
        self.c = c

        self.temb = nn.ModuleList([nn.Linear(c, c*4), nn.Linear(c*4, c*4)])
        self.conv_in = nn.Conv2d(3, c, 3, padding=1)

        self.down_0 = nn.ModuleList([ResnetBlock(c*chmul[0], was_pytorch=True), ResnetBlock(c*chmul[0], was_pytorch=True), Downsample(c*chmul[0], with_conv=True)])
        self.down_1 = nn.ModuleList([ResnetBlock(c*chmul[1], use_nin_shortcut=True, was_pytorch=True), ResnetBlock(c*chmul[1], was_pytorch=True), Downsample(c*chmul[1], with_conv=True)])
        self.down_2 = nn.ModuleList([ResnetBlock(c*chmul[2], was_pytorch=True), ResnetBlock(c*chmul[2], was_pytorch=True)])
        self.attnsdown = nn.ModuleList([AttnBlock(c*chmul[1], was_pytorch=True), AttnBlock(c*chmul[1], was_pytorch=True)])
        self.downsample2 = Downsample(c*chmul[2], with_conv=True)
        self.down_3 = nn.ModuleList([ResnetBlock(c*chmul[3], was_pytorch=True), ResnetBlock(c*chmul[3], was_pytorch=True), Downsample(c*chmul[3], with_conv=True)])
        self.down_4 = nn.ModuleList([ResnetBlock(c*chmul[4], use_nin_shortcut=True, was_pytorch=True), ResnetBlock(c*chmul[4], was_pytorch=True)])

        self.mids = nn.ModuleList([ResnetBlock(c*chmul[-1], was_pytorch=True), AttnBlock(c*chmul[-1], was_pytorch=True), ResnetBlock(c*chmul[-1], was_pytorch=True)])

        self.up_0 = nn.ModuleList([ResnetBlock(c*chmul[0], use_nin_shortcut=True, was_pytorch=True) for _ in range(3)])
        self.up_1 = nn.ModuleList([ResnetBlock(c*chmul[1], use_nin_shortcut=True, was_pytorch=True) for _ in range(3)] + [Upsample(c*chmul[1], with_conv=True)])
        self.up_2 = nn.ModuleList([ResnetBlock(c*chmul[2], use_nin_shortcut=True, was_pytorch=True) for _ in range(3)])
        self.attnsup = nn.ModuleList([AttnBlock(c*chmul[2], was_pytorch=True) for _ in range(3)])
        self.upsample2 = Upsample(c*chmul[2], with_conv=True)
        self.up_3 = nn.ModuleList([ResnetBlock(c*chmul[3], use_nin_shortcut=True, was_pytorch=True) for _ in range(3)] + [Upsample(c*chmul[3], with_conv=True)])
        self.up_4 = nn.ModuleList([ResnetBlock(c*chmul[4], use_nin_shortcut=True, was_pytorch=True) for _ in range(3)] + [Upsample(c*chmul[4], with_conv=True)])

        self.norm_out = nn.GroupNorm(32, c*chmul[0])
        self.conv_out = nn.Conv2d(c, 3, 3, padding=1)

    def get_pretrained_weights(self, weights_path):
        # Note: model name must be specific. The model name must be model_tf2_{DATASET}.h5 e.g. model_tf2_lsun_church.h5
        self(torch.randn(1, 3, 64, 64), torch.ones(1))  # builds model.
        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x, index):
        index = get_timestep_embedding(index, self.c)
        index = F.silu(self.temb[0](index))
        index = self.temb[1](index)

        x = self.conv_in(x)
        residuals = [x]

        for block in self.down_0:
            x = block(x, index)
            residuals.append(x)

        for block in self.down_1:
            x = block(x, index)
            residuals.append(x)

        for i, block in enumerate(self.down_2):
            x = block(x, index)
            if i < 2: x = self.attnsdown[i](x, index)
            residuals.append(x)

        x = self.downsample2(x, index)
        residuals.append(x)

        for block in self.down_3:
            x = block(x, index)
            residuals.append(x)

        for block in self.down_4:
            x = block(x, index)
            residuals.append(x)

        for block in self.mids:
            x = block(x, index)

        for i, block in enumerate(self.up_4):
            if i < 3:
                x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, index)

        for i, block in enumerate(self.up_3):
            if i < 3:
                x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, index)

        for i, block in enumerate(self.up_2):
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, index)
            x = self.attnsup[i](x, index)

        x = self.upsample2(x, index)

        for i, block in enumerate(self.up_1):
            if i < 3:
                x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, index)

        for i, block in enumerate(self.up_0):
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, index)

        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)

        return x

class OnestepModel(nn.Module):
    def __init__(self, data_to_use, model_path):
        super(OnestepModel, self).__init__()
        self.data_to_use = data_to_use
        if data_to_use == 'cifar10':
            self.pretrained_model = DDPMCheckpointModel(data_to_use)
            self.spatialres = 32
        elif 'lsun' in data_to_use:
            self.pretrained_model = DDPMCheckpointModel(data_to_use)
            self.spatialres = 256
        elif data_to_use == 'celeba':
            self.pretrained_model = DDIMCheckpointModel()
            self.spatialres = 64
        else:
            raise NotImplementedError

        self.pretrained_model.get_pretrained_weights(model_path)

    def forward(self, z):
        # uses the highest index seen by the pretrained model.
        inp = z.clone()
        index = torch.ones_like(z[:, 0, 0, 0]) * 999.
        x = self.pretrained_model(z, index)
        # this layers output can be thought of as returning a prediction for epsilon.
        pred_y = inp - x
        return pred_y
    
    def run_ddim_step(self, xt, index, alpha, alpha_next):
        eps = self.pretrained_model(xt, index)
        x_t_minus1 = torch.sqrt(alpha_next) * (xt - torch.sqrt(1 - alpha) * eps) / torch.sqrt(alpha)
        x_t_minus1 += torch.sqrt(1 - alpha_next) * eps
        return x_t_minus1
