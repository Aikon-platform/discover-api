import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from itertools import chain

from .transformer import PrototypeTransformationNetwork
from .tools import copy_with_noise, create_gaussian_weights, generate_data
from ..utils.logger import print_warning

from .u_net import UNet

NOISE_SCALE = 0.0001
EMPTY_CLUSTER_THRESHOLD = 0.2
LATENT_SIZE = 128


class DTIKmeans(nn.Module):
    name = "dtikmeans"

    def __init__(self, dataset=None, n_prototypes=10, **kwargs):
        super().__init__()
        if dataset is None:
            raise NotImplementedError
        self.n_prototypes = n_prototypes
        self.img_size = dataset.img_size
        # Prototypes
        self.color_channels = kwargs.get("color_channels", 3)
        assert kwargs.get("prototype")
        proto_args = kwargs.get("prototype")
        proto_source = proto_args.get("source", "data")
        assert proto_source in ["data", "generator"]
        self.proto_source = proto_source
        if proto_source == "data":
            data_args = proto_args.get("data")
            init_type = data_args.get("init", "sample")
            std = data_args.get("gaussian_weights_std", 25)
            self.prototype_params = nn.Parameter(
                torch.stack(generate_data(dataset, n_prototypes, init_type, std=std))
            )
        else:
            gen_name = proto_args.get("generator", "mlp")
            print_warning("Sprites will be generated from latent variables.")
            assert gen_name in ["mlp", "unet"]
            latent_dims = (
                (LATENT_SIZE,)
                if gen_name == "mlp"
                else (1, self.img_size[0], self.img_size[1])
            )
            self.latent_params = nn.Parameter(
                torch.stack(
                    [
                        torch.normal(mean=0.0, std=1.0, size=latent_dims)
                        for k in range(n_prototypes)
                    ],
                    dim=0,
                )
            )
            self.generator = self.init_generator(
                gen_name,
                LATENT_SIZE,
                self.color_channels,
                self.color_channels * self.img_size[0] * self.img_size[1],
            )

        self.transformer = PrototypeTransformationNetwork(
            dataset.n_channels, dataset.img_size, n_prototypes, **kwargs
        )
        self.empty_cluster_threshold = kwargs.get(
            "empty_cluster_threshold", EMPTY_CLUSTER_THRESHOLD / n_prototypes
        )
        self._reassign_cluster = kwargs.get("reassign_cluster", True)
        use_gaussian_weights = kwargs.get("gaussian_weights", False)
        if use_gaussian_weights:
            std = kwargs.get("gaussian_weights_std")
            self.register_buffer(
                "loss_weights",
                create_gaussian_weights(dataset.img_size, dataset.n_channels, std),
            )
        else:
            self.loss_weights = None

    def cluster_parameters(self):
        if hasattr(self, "generator"):
            return list(chain(*[self.generator.parameters()])) + [self.latent_params]
        return [self.prototype_params]

    def transformer_parameters(self):
        return self.transformer.parameters()

    @staticmethod
    def init_generator(name, latent_dim, color_channel, out_channel):
        if name == "unet":
            return UNet(1, color_channel)
        elif name == "mlp":
            return nn.Sequential(
                nn.Linear(latent_dim, 8 * latent_dim),
                nn.GroupNorm(8, 8 * latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(8 * latent_dim, out_channel),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError("Generator not implemented.")

    @property
    def prototypes(self):
        if self.proto_source == "generator":
            with torch.no_grad():
                params = self.generator(self.latent_params)
                if len(params.size()) != 4:
                    params = params.reshape(
                        -1, self.color_channels, self.img_size[0], self.img_size[1]
                    )
                return params
        else:
            return self.prototype_params

    def forward(self, x):
        if self.proto_source == "generator":
            params = self.generator(self.latent_params)
            if len(params.size()) != 4:
                params = params.reshape(
                    -1, self.color_channels, self.img_size[0], self.img_size[1]
                )
        else:
            params = self.prototype_params
        prototypes = params.unsqueeze(1).expand(-1, x.size(0), x.size(1), -1, -1)
        inp, target = self.transformer(x, prototypes)
        distances = (inp - target) ** 2
        if self.loss_weights is not None:
            distances = distances * self.loss_weights
        distances = distances.flatten(2).mean(2)
        dist_min = distances.min(1)[0]
        return dist_min.mean(), distances

    @torch.no_grad()
    def transform(self, x, inverse=False):
        if inverse:
            return self.transformer.inverse_transform(x)
        else:
            prototypes = self.prototypes.unsqueeze(1).expand(
                -1, x.size(0), x.size(1), -1, -1
            )
            return self.transformer(x, prototypes)[1]

    def step(self):
        self.transformer.step()

    def set_optimizer(self, opt):
        self.optimizer = opt
        self.transformer.set_optimizer(opt)

    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in state_dict.items():
            if name in state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                state[name].copy_(param)
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f"load_state_dict: {unloaded_params} not found")

    def reassign_empty_clusters(self, proportions):
        if not self._reassign_cluster:
            return [], 0

        idx = np.argmax(proportions)
        reassigned = []
        for i in range(self.n_prototypes):
            if proportions[i] < self.empty_cluster_threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)
        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)
        return reassigned, idx

    def restart_branch_from(self, i, j):
        if hasattr(self, "generator"):
            self.latent_params[i].data.copy_(
                copy_with_noise(self.latent_params[j], NOISE_SCALE)
            )
            param = self.latent_params
        else:
            self.prototype_params[i].data.copy_(
                copy_with_noise(self.prototype_params[j], NOISE_SCALE)
            )
            param = self.prototype_params
        self.transformer.restart_branch_from(i, j, noise_scale=0)

        if hasattr(self, "optimizer"):
            opt = self.optimizer
            if isinstance(opt, (Adam,)):
                opt.state[param]["exp_avg"][i] = opt.state[param]["exp_avg"][j]
                opt.state[param]["exp_avg_sq"][i] = opt.state[param]["exp_avg_sq"][j]
            else:
                raise NotImplementedError(
                    "unknown optimizer: you should define how to reinstanciate statistics if any"
                )
