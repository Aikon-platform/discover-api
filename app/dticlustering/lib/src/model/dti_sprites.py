from copy import deepcopy
from itertools import chain
import math

import torch
from torch.optim import Adam, RMSprop
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from .transformer import (
    PrototypeTransformationNetwork as Transformer,
    N_HIDDEN_UNITS,
    N_LAYERS,
)
from .tools import (
    copy_with_noise,
    generate_data,
    create_gaussian_weights,
    get_clamp_func,
    create_mlp,
    get_bbox_from_mask,
)
from ..utils.logger import print_warning
from .u_net import UNet

NOISE_SCALE = 0.0001
EMPTY_CLUSTER_THRESHOLD = 0.2
LATENT_SIZE = 128


def init_linear(hidden, out, init, n_channels=3, std=5, freeze_frg=False, dataset=None):
    if init == "random":
        return nn.Linear(hidden, out)
    elif init == "gaussian":
        linear = nn.Linear(hidden, out)
        if freeze_frg:
            h = int(math.sqrt(out))
            size = [h, h]
            mask = create_gaussian_weights(size, 1, std)
            sample = mask.flatten()

        else:
            h = int(math.sqrt(out / (n_channels + 1)))
            size = [h, h]
            mask = create_gaussian_weights(size, 1, std)
            sample = torch.cat(
                (
                    torch.full(
                        size=(3 * h * h,),
                        fill_value=0.9,
                    ),
                    mask.flatten(),
                ),
            )
        nn.init.constant_(
            linear.weight, 1e-10
        )  # a small value to avoid vanishing grads
        sample = torch.log(sample / (1 - sample))
        linear.bias.data.copy_(sample)
        return linear
    elif init == "mean":
        linear = nn.Linear(hidden, out)
        assert dataset is not None
        images = next(
            iter(DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4))
        )[0]
        sample = images.mean(0)
        nn.init.constant_(linear.weight, 0.0001)
        sample = torch.log(sample / (1 - sample))
        linear.bias.data.copy_(sample.flatten())
        return linear
    else:
        raise NotImplementedError("init is not implemented.")


def layered_composition(layers, masks, occ_grid):
    # LBCHW size of layers and masks and LLB size for occ_grid
    occ_masks = (1 - occ_grid[..., None, None, None].transpose(0, 1) * masks).prod(
        1
    )  # LBCHW
    return (occ_masks * masks * layers).sum(0)  # BCHW


class DTISprites(nn.Module):
    name = "dti_sprites"
    learn_masks = True

    def __init__(self, dataset, n_sprites, n_objects=1, **kwargs):
        super().__init__()
        if dataset is None:
            raise NotImplementedError
        else:
            img_size = dataset.img_size
            n_ch = dataset.n_channels

        # Prototypes & masks
        size = kwargs.get("sprite_size", img_size)
        self.sprite_size = size
        self.img_size = img_size
        self.return_map_out = kwargs.get("return_map_out", False)
        color_channels = kwargs.get("color_channels", 3)
        self.color_channels = color_channels
        self.add_empty_sprite = kwargs.get("add_empty_sprite", False)
        self.lambda_empty_sprite = kwargs.get("lambda_empty_sprite", 0)
        self.n_sprites = n_sprites + 1 if self.add_empty_sprite else n_sprites

        proto_args = kwargs.get("prototype")
        proto_source = proto_args.get("source", "data")
        assert proto_source in ["data", "generator"]
        self.proto_source = proto_source
        if proto_args.get("data", None) is not None:
            data_args = proto_args.get("data")
            freeze_frg, freeze_bkg, freeze_sprite = data_args.get("freeze", [0, 0, 0])
            value_frg, value_bkg, value_mask = data_args.get("value", [0.5, 0.5, 0.5])
            std = data_args.get("gaussian_weights_std", 25)
            proto_init, bkg_init, mask_init = data_args.get(
                "init", ["constant", "constant", "constant"]
            )
        else:
            freeze_frg, freeze_bkg, freeze_sprite = False, False, False
            value_frg, value_bkg, value_mask = 0.5, 0.5, 0.5
            std = 25
            proto_init, bkg_init, mask_init = "constant", "constant", "constant"

        if proto_source == "data":
            self.prototype_params = nn.Parameter(
                torch.stack(
                    generate_data(
                        dataset, n_sprites, proto_init, value=value_frg, size=size
                    )
                )
            )
            self.prototype_params.requires_grad = False if freeze_frg else True
            self.mask_params = nn.Parameter(
                self.init_masks(n_sprites, mask_init, size, std, value_mask, dataset)
            )
        else:
            if freeze_frg:
                self.prototype_params = nn.Parameter(
                    torch.stack(
                        generate_data(
                            dataset, n_sprites, proto_init, value=value_frg, size=size
                        )
                    ),
                    requires_grad=False,
                )
                latent_out_ch = 1
            else:
                latent_out_ch = color_channels + 1

            gen_name = proto_args.get("generator", "mlp")
            print_warning("Sprites will be generated from latent variables.")
            assert gen_name in ["mlp", "unet"]
            latent_dims = (LATENT_SIZE,) if gen_name == "mlp" else (1, size[0], size[1])
            self.latent_params = nn.Parameter(
                torch.stack(
                    [
                        torch.normal(mean=0.0, std=1.0, size=latent_dims)
                        for k in range(n_sprites)
                    ],
                    dim=0,
                ),
            )
            self.generator = self.init_generator(
                gen_name,
                LATENT_SIZE,
                color_channels,
                latent_out_ch * size[0] * size[1],
                kwargs.get("init_latent_linear", "random"),
                std=std,
                freeze_frg=freeze_frg,
            )
        clamp_name = kwargs.get("use_clamp", "soft")
        self.clamp_func = get_clamp_func(clamp_name)
        self.cur_epoch = 0
        self.n_linear_layers = kwargs.get("n_linear_layers", N_LAYERS)
        self.estimate_minimum = kwargs.get("estimate_minimum", False)
        self.greedy_algo_iter = kwargs.get("greedy_algo_iter", 1)
        self.freeze_milestone = freeze_sprite if freeze_sprite else -1
        assert isinstance(self.freeze_milestone, (int,))
        self.freeze_frg = freeze_frg
        self.freeze_bkg = freeze_bkg
        # Sprite transformers
        L = n_objects
        self.n_objects = n_objects
        self.has_layer_tsf = kwargs.get(
            "transformation_sequence_layer", "identity"
        ) not in ["id", "identity"]
        if self.has_layer_tsf:
            layer_kwargs = deepcopy(kwargs)
            layer_kwargs["transformation_sequence"] = kwargs[
                "transformation_sequence_layer"
            ]
            layer_kwargs["curriculum_learning"] = kwargs["curriculum_learning_layer"]
            self.layer_transformer = Transformer(n_ch, img_size, L, **layer_kwargs)
            self.encoder = self.layer_transformer.encoder
            tsfs = [
                Transformer(
                    n_ch,
                    size,
                    self.n_sprites,
                    encoder=self.encoder,
                    **dict(kwargs, freeze_frg=freeze_frg),
                )
                for k in range(L)
            ]
            self.sprite_transformers = nn.ModuleList(tsfs)
        else:
            if L > 1:
                self.layer_transformer = Transformer(
                    n_ch, img_size, L, transformation_sequence="identity"
                )
            first_tsf = Transformer(
                n_ch, img_size, self.n_sprites, **dict(kwargs, freeze_frg=freeze_frg)
            )
            self.encoder = first_tsf.encoder
            tsfs = [
                Transformer(
                    n_ch,
                    img_size,
                    self.n_sprites,
                    encoder=self.encoder,
                    **dict(kwargs, freeze_frg=freeze_frg),
                )
                for k in range(L - 1)
            ]
            self.sprite_transformers = nn.ModuleList([first_tsf] + tsfs)

        # Background Transformer
        M = kwargs.get("n_backgrounds", 0)
        self.n_backgrounds = M
        self.learn_backgrounds = M > 0
        if self.learn_backgrounds:
            if proto_source == "data":
                self.bkg_params = nn.Parameter(
                    torch.stack(
                        generate_data(dataset, M, init_type=bkg_init, value=value_bkg)
                    )
                )
                self.bkg_params.requires_grad = False if freeze_bkg else True
            else:
                if freeze_bkg:
                    self.bkg_params = nn.Parameter(
                        torch.stack(
                            generate_data(
                                dataset, M, init_type=bkg_init, value=value_bkg
                            )
                        ),
                        requires_grad=False,
                    )
                else:
                    gen_name = proto_args.get("generator", "mlp")
                    print_warning("Background will be generated from latent variables.")
                    latent_dims = (
                        (LATENT_SIZE,)
                        if gen_name == "mlp"
                        else (1, img_size[0], img_size[1])
                    )
                    self.bkg_generator = self.init_generator(
                        gen_name,
                        LATENT_SIZE,
                        color_channels,
                        color_channels * img_size[0] * img_size[1],
                        kwargs.get("init_bkg_linear", "random"),
                        std=None,
                        dataset=dataset,
                    )
                    self.latent_bkg_params = nn.Parameter(
                        torch.stack(
                            [
                                torch.normal(mean=0.0, std=1.0, size=latent_dims)
                                for k in range(M)
                            ]
                        )
                    )

            bkg_kwargs = deepcopy(kwargs)
            bkg_kwargs["transformation_sequence"] = kwargs[
                "transformation_sequence_bkg"
            ]
            bkg_kwargs["curriculum_learning"] = kwargs["curriculum_learning_bkg"]
            bkg_kwargs["padding_mode"] = "border"
            self.bkg_transformer = Transformer(
                n_ch, img_size, M, encoder=self.encoder, **bkg_kwargs
            )

        # Image composition and aux
        self.pred_occlusion = kwargs.get("pred_occlusion", False)
        if self.pred_occlusion:
            nb_out = int(L * (L - 1) / 2)
            norm = kwargs.get("norm_layer")
            self.occ_predictor = create_mlp(
                self.encoder.out_ch, nb_out, N_HIDDEN_UNITS, self.n_linear_layers, norm
            )
            self.occ_predictor[-1].weight.data.zero_()
            self.occ_predictor[-1].bias.data.zero_()
        else:
            self.register_buffer("occ_grid", torch.tril(torch.ones(L, L), diagonal=-1))

        self._criterion = nn.MSELoss(reduction="none")
        self.empty_cluster_threshold = kwargs.get(
            "empty_cluster_threshold", EMPTY_CLUSTER_THRESHOLD / n_sprites
        )
        self._reassign_cluster = kwargs.get("reassign_cluster", True)
        self.inject_noise = kwargs.get("inject_noise", 0)

    @staticmethod
    def init_masks(K, mask_init, size, std, value, dataset):
        if mask_init == "constant":
            masks = torch.full((K, 1, *size), value)
        elif mask_init == "gaussian":
            assert std is not None
            mask = create_gaussian_weights(size, 1, std)
            masks = mask.unsqueeze(0).expand(K, -1, -1, -1)
        elif mask_init == "random":
            masks = torch.rand(K, 1, *size)
        elif mask_init == "sample":
            assert dataset
            masks = torch.stack(
                generate_data(dataset, K, init_type=mask_init, value=value)
            )
            assert masks.shape[1] == 1
        else:
            raise NotImplementedError(f"unknown mask_init: {mask_init}")
        return masks

    @staticmethod
    def init_generator(
        name,
        latent_dim,
        color_channel,
        out_channel,
        init="random",
        freeze_frg=False,
        std=5,
        dataset=None,
    ):
        if name == "unet":
            return UNet(1, color_channel)
        elif name == "mlp":
            linear = init_linear(
                8 * latent_dim,
                out_channel,
                init,
                std=std,
                dataset=dataset,
                freeze_frg=freeze_frg,
            )  # freeze_frg = False by default
            model = nn.Sequential(
                nn.Linear(latent_dim, 8 * latent_dim),
                nn.GroupNorm(8, 8 * latent_dim),
                nn.ReLU(inplace=True),
                linear,
                nn.Sigmoid(),
            )
            return model
        else:
            raise NotImplementedError("Generator not implemented.")

    @property
    def n_prototypes(self):
        return self.n_sprites

    @property
    def masks(self):
        if self.proto_source == "data":
            masks = self.mask_params
        else:
            with torch.no_grad():
                if self.freeze_frg:
                    masks = self.generator(self.latent_params)
                else:
                    masks = self.generator(self.latent_params)[
                        :,
                        self.color_channels
                        * self.sprite_size[0]
                        * self.sprite_size[1] :,
                    ]
            if len(masks.size()) != 4:
                masks = masks.reshape(-1, 1, self.sprite_size[0], self.sprite_size[1])

        if self.add_empty_sprite:
            masks = torch.cat(
                [masks, torch.zeros(1, *masks[0].shape, device=masks.device)]
            )

        if self.inject_noise and self.training:
            return masks
        else:
            return self.clamp_func(masks)

    @property
    def prototypes(self):
        if self.proto_source == "data":
            params = self.prototype_params
        else:
            with torch.no_grad():
                if self.freeze_frg:
                    params = self.prototype_params
                else:
                    params = self.generator(self.latent_params)[
                        :,
                        : self.color_channels
                        * self.sprite_size[0]
                        * self.sprite_size[1],
                    ]
                    if len(params.size()) != 4:
                        params = params.reshape(
                            -1,
                            self.color_channels,
                            self.sprite_size[0],
                            self.sprite_size[1],
                        )

        if self.add_empty_sprite:
            params = torch.cat(
                [params, torch.zeros(1, *params[0].shape, device=params.device)]
            )

        return self.clamp_func(params)

    @property
    def backgrounds(self):
        if self.proto_source == "data":
            params = self.bkg_params
        else:
            with torch.no_grad():
                if self.freeze_bkg:
                    params = self.bkg_params
                else:
                    params = self.bkg_generator(self.latent_bkg_params)
                    if len(params.size()) != 4:
                        params = params.reshape(
                            -1, self.color_channels, self.img_size[0], self.img_size[1]
                        )
        return self.clamp_func(params)

    @property
    def is_layer_tsf_id(self):
        if hasattr(self, "layer_transformer"):
            return self.layer_transformer.only_id_activated
        else:
            return False

    @property
    def are_sprite_frozen(self):
        return (
            True
            if self.freeze_milestone > 0 and self.cur_epoch < self.freeze_milestone
            else False
        )

    def cluster_parameters(self):
        if self.proto_source == "data":
            params = [self.prototype_params, self.mask_params]
            if self.learn_backgrounds:
                params.append(self.bkg_params)
        else:
            params = list(chain(*[self.generator.parameters()])) + [self.latent_params]
            if self.learn_backgrounds and not self.freeze_bkg:
                params.append(self.latent_bkg_params)
                params.extend(list(chain(*[self.bkg_generator.parameters()])))
        return iter(params)

    def transformer_parameters(self):
        params = [t.get_parameters() for t in self.sprite_transformers]
        if hasattr(self, "layer_transformer"):
            params.append(self.layer_transformer.get_parameters())
        if self.learn_backgrounds:
            params.append(self.bkg_transformer.get_parameters())
        if self.pred_occlusion:
            params.append(self.occ_predictor.parameters())
        return chain(*params)

    def forward(self, x, img_masks=None, epoch=None):
        B, C, H, W = x.size()
        L, K, M = self.n_objects, self.n_sprites, self.n_backgrounds or 1
        tsf_layers, tsf_masks, tsf_bkgs, occ_grid, class_prob = self.predict(
            x, epoch=epoch
        )

        if class_prob is None:
            target = self.compose(
                tsf_layers, tsf_masks, occ_grid, tsf_bkgs, class_prob
            )  # B(K**L*M)CHW
            x = x.unsqueeze(1).expand(-1, K**L * M, -1, -1, -1)
            if img_masks != None:
                img_masks = img_masks.unsqueeze(1).expand(-1, K**L * M, -1, -1, -1)
            distances = self.criterion(x, target, weights=img_masks)
            loss = distances.min(1)[0].mean()
        else:
            target = self.compose(
                tsf_layers, tsf_masks, occ_grid, tsf_bkgs, class_prob
            )  # BCHW
            if img_masks != None:
                img_masks = img_masks.unsqueeze(1)
            loss = self.criterion(
                x.unsqueeze(1), target.unsqueeze(1), weights=img_masks
            ).mean()
            distances = 1 - class_prob.permute(2, 0, 1).flatten(1)  # B(L*K)

        return loss, distances

    def predict(self, x, epoch=None):
        B, C, H, W = x.size()
        h, w = self.prototypes.shape[2:]
        L, K, M = self.n_objects, self.n_sprites, self.n_backgrounds or 1
        if hasattr(self, "generator"):
            out = self.generator(self.latent_params)
            if self.freeze_frg:
                prototypes = self.prototypes
                masks = out.reshape(-1, 1, self.sprite_size[0], self.sprite_size[1])
            else:
                prototypes = out[
                    :, : self.color_channels * self.prite_size[0] * self.sprite_size[1]
                ].reshape(
                    -1, self.color_channels, self.sprite_size[0], self.sprite_size[1]
                )
                masks = out[
                    :, self.color_channels * self.sprite_size[0] * self.sprite_size[1] :
                ].reshape(-1, 1, self.sprite_size[0], self.sprite_size[1])
            if self.add_empty_sprite:
                if not self.freeze_frg:
                    prototypes = torch.cat(
                        [
                            prototypes,
                            torch.zeros(
                                1, *prototypes[0].shape, device=prototypes.device
                            ),
                        ]
                    )
                masks = torch.cat(
                    [masks, torch.zeros(1, *masks[0].shape, device=masks.device)]
                )
            prototypes = self.clamp_func(prototypes)
            if self.inject_noise and self.training:
                masks = masks
            else:
                masks = self.clamp_func(masks)
        else:
            prototypes = self.prototypes
            masks = self.masks

        prototypes = prototypes.unsqueeze(1).expand(K, B, C, -1, -1)
        masks = masks.unsqueeze(1).expand(K, B, 1, -1, -1)
        sprites = torch.cat([prototypes, masks], dim=2)
        if self.inject_noise and self.training:
            # XXX we use a canva to inject noise after transformations to avoid gridding artefacts
            if self.add_empty_sprite:
                canvas = torch.cat(
                    [torch.ones(K - 1, B, 1, h, w), torch.zeros(1, B, 1, h, w)]
                ).to(x.device)
            else:
                canvas = torch.ones(K, B, 1, h, w, device=x.device)
            sprites = torch.cat([sprites, canvas], dim=2)
        if self.are_sprite_frozen:
            sprites = sprites.detach()
        features = self.encoder(x)
        tsf_sprites = torch.stack(
            [self.sprite_transformers[k](x, sprites, features)[1] for k in range(L)],
            dim=0,
        )
        if self.has_layer_tsf:
            layer_features = features.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
            tsf_layers = self.layer_transformer(
                x, tsf_sprites.view(L, B * K, -1, h, w), layer_features
            )[1]
            tsf_layers = tsf_layers.view(B, K, L, -1, H, W).transpose(0, 2)  # LKBCHW
        else:
            tsf_layers = tsf_sprites.transpose(1, 2)  # LKBCHW

        if self.inject_noise and self.training:
            tsf_layers, tsf_masks, tsf_noise = torch.split(tsf_layers, [C, 1, 1], dim=3)
        else:
            tsf_layers, tsf_masks = torch.split(tsf_layers, [C, 1], dim=3)

        if self.learn_backgrounds:
            if hasattr(self, "generator"):
                if not self.freeze_bkg:
                    backgrounds = self.bkg_generator(self.latent_bkg_params).reshape(
                        -1, self.color_channels, self.img_size[0], self.img_size[1]
                    )
                    backgrounds = self.clamp_func(backgrounds)
                else:
                    backgrounds = self.backgrounds
            else:
                backgrounds = self.backgrounds
            backgrounds = backgrounds.unsqueeze(1).expand(M, B, C, -1, -1)
            tsf_bkgs = self.bkg_transformer(x, backgrounds, features)[1].transpose(
                0, 1
            )  # MBCHW
        else:
            tsf_bkgs = None

        if self.inject_noise and self.training:  #  and epoch >= 500:
            noise = (
                torch.rand(K, 1, H, W, device=x.device)[None, None, ...]
                .expand(L, B, K, 1, H, W)
                .transpose(1, 2)
            )
            tsf_masks = tsf_masks + tsf_noise * (
                2 * self.inject_noise * noise - self.inject_noise
            )
            tsf_masks = self.clamp_func(tsf_masks)

        occ_grid = self.predict_occlusion_grid(x, features)  # LLB
        if self.estimate_minimum:
            class_prob = self.greedy_algo_selection(
                x, tsf_layers, tsf_masks, tsf_bkgs, occ_grid
            )  # LKB
            self._class_prob = class_prob  # for monitoring and debug only
        else:
            class_prob = None

        return tsf_layers, tsf_masks, tsf_bkgs, occ_grid, class_prob

    def predict_occlusion_grid(self, x, features):
        B, L = x.size(0), self.n_objects
        if self.pred_occlusion:
            inp = features if features is not None else x
            occ_grid = self.occ_predictor(inp)  # view(-1, L, L)
            occ_grid = torch.sigmoid(occ_grid)
            grid = torch.zeros(B, L, L, device=x.device)
            indices = torch.tril_indices(row=L, col=L, offset=-1)
            grid[:, indices[0], indices[1]] = occ_grid
            occ_grid = grid + torch.triu(1 - grid.transpose(1, 2), diagonal=1)
        else:
            occ_grid = self.occ_grid.unsqueeze(0).expand(B, -1, -1)

        return occ_grid.permute(1, 2, 0)  # LLB

    @torch.no_grad()
    def greedy_algo_selection(self, x, layers, masks, bkgs, occ_grid):
        L, K, B, C, H, W = layers.shape
        if self.add_empty_sprite and self.are_sprite_frozen:
            layers, masks = layers[:, :-1], masks[:, :-1]
            K = K - 1
        x, device = x.unsqueeze(0).expand(K, -1, -1, -1, -1), x.device
        bkgs = torch.zeros(1, B, C, H, W, device=device) if bkgs is None else bkgs
        cur_layers = torch.cat([bkgs, torch.zeros(L, B, C, H, W, device=device)])
        cur_masks = torch.cat(
            [
                torch.ones(1, B, 1, H, W, device=device),
                torch.zeros(L, B, 1, H, W, device=device),
            ]
        )
        one, zero = torch.ones(B, L, 1, device=device), torch.zeros(
            B, 1, L + 1, device=device
        )
        occ_grid = torch.cat(
            [zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1
        ).permute(1, 2, 0)

        resps, diff_select = torch.zeros(L, K, B, device=device), [[], []]
        for step in range(self.greedy_algo_iter):
            for l, (layer, mask) in enumerate(zip(layers, masks), start=1):
                recons = []
                for k in range(K):
                    tmp_layers = torch.cat(
                        [cur_layers[:l], layer[[k]], cur_layers[l + 1 :]]
                    )
                    tmp_masks = torch.cat(
                        [cur_masks[:l], mask[[k]], cur_masks[l + 1 :]]
                    )
                    recons.append(layered_composition(tmp_layers, tmp_masks, occ_grid))
                distance = ((x - torch.stack(recons)) ** 2).flatten(2).mean(2)
                if self.add_empty_sprite and not self.are_sprite_frozen:
                    distance += (
                        self.lambda_empty_sprite
                        * torch.Tensor([1] * (K - 1) + [0]).to(device)[:, None]
                    )
                resp = torch.zeros(K, B, device=device).scatter_(
                    0, distance.argmin(0, keepdim=True), 1
                )
                resps[l - 1] = resp
                cur_layers[l] = (layer * resp[..., None, None, None]).sum(axis=0)
                cur_masks[l] = (mask * resp[..., None, None, None]).sum(axis=0)

            if True:
                # For debug purposes only
                if step == 0:
                    indices = resps.argmax(1).flatten()
                else:
                    new_indices = resps.argmax(1).flatten()
                    diff_select[0].append(str(step))
                    diff_select[1].append(
                        (new_indices != indices).float().mean().item()
                    )
                    indices = new_indices
        # For debug purposes only
        if step > 0:
            self._diff_selections = diff_select

        if self.add_empty_sprite and self.are_sprite_frozen:
            resps = torch.cat([resps, torch.zeros(L, 1, B, device=device)], dim=1)
        return resps

    def compose(self, layers, masks, occ_grid, backgrounds=None, class_prob=None):
        L, K, B, C, H, W = layers.shape
        device = occ_grid.device

        if class_prob is not None:
            masks = (masks * class_prob[..., None, None, None]).sum(axis=1)
            layers = (layers * class_prob[..., None, None, None]).sum(axis=1)
            size = (B, C, H, W)
            if backgrounds is not None:
                masks = torch.cat([torch.ones(1, B, 1, H, W, device=device), masks])
                layers = torch.cat([backgrounds, layers])
                one, zero = torch.ones(B, L, 1, device=device), torch.zeros(
                    B, 1, L + 1, device=device
                )
                occ_grid = torch.cat(
                    [zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1
                ).permute(1, 2, 0)
            return layered_composition(layers, masks, occ_grid)

        else:
            layers = [
                layers[k][(None,) * (L - 1)].transpose(k, L - 1) for k in range(L)
            ]  # L elements of size K1.. 1BCHW
            masks = [
                masks[k][(None,) * (L - 1)].transpose(k, L - 1) for k in range(L)
            ]  # L elements of size K1...1BCHW
            size = (K,) * L + (B, C, H, W)
            if backgrounds is not None:
                M = backgrounds.size(0)
                backgrounds = backgrounds[(None,) * L].transpose(0, L)  # M1..1BCHW
                layers = [backgrounds] + [layers[k][None] for k in range(L)]
                masks = [torch.ones((1,) * (L + 1) + (B, C, H, W)).to(device)] + [
                    masks[k][None] for k in range(L)
                ]
                one, zero = torch.ones(B, L, 1, device=device), torch.zeros(
                    B, 1, L + 1, device=device
                )
                occ_grid = torch.cat(
                    [zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1
                ).permute(1, 2, 0)
                size = (M,) + size
            else:
                M = 1

            occ_grid = occ_grid[..., None, None, None]
            res = torch.zeros(size, device=device)
            for k in range(len(layers)):
                if backgrounds is not None:
                    j_start = 1 if self.pred_occlusion else k + 1
                else:
                    j_start = 0 if self.pred_occlusion else k + 1
                occ_masks = torch.ones(size, device=device)
                for j in range(j_start, len(layers)):
                    if j != k:
                        occ_masks *= 1 - occ_grid[j, k] * masks[j]
                res += occ_masks * masks[k] * layers[k]
            return res.view(K**L * M, B, C, H, W).transpose(0, 1)

    def criterion(self, inp, target, weights=None, reduction="mean"):
        dist = self._criterion(inp, target)
        if weights is not None:
            dist = dist * weights
        if reduction == "mean":
            return dist.flatten(2).mean(2)
        elif reduction == "sum":
            return dist.flatten(2).sum(2)
        elif reduction == "none":
            return dist
        else:
            raise NotImplementedError

    @torch.no_grad()
    def transform(
        self,
        x,
        with_composition=False,
        pred_semantic_labels=False,
        pred_instance_labels=False,
        with_bkg=True,
        hard_occ_grid=False,
    ):
        B, C, H, W = x.size()
        L, K = self.n_objects, self.n_sprites
        tsf_layers, tsf_masks, tsf_bkgs, occ_grid, class_prob = self.predict(x)
        if class_prob is not None:
            class_oh = torch.zeros(class_prob.shape, device=x.device).scatter_(
                1, class_prob.argmax(1, keepdim=True), 1
            )
        else:
            class_oh = None

        if pred_semantic_labels:
            label_layers = (
                torch.arange(1, K + 1, device=x.device)[(None,) * 4]
                .transpose(0, 4)
                .expand(L, -1, B, 1, H, W)
            )
            true_occ_grid = (occ_grid > 0.5).float()
            target = self.compose(
                label_layers,
                (tsf_masks > 0.5).long(),
                true_occ_grid,
                class_prob=class_oh,
            ).squeeze(1)
            if self.return_map_out:
                binary_masks = (tsf_masks > 0.5).long()
                bboxes = get_bbox_from_mask(binary_masks)
                class_ids = class_oh
                return target.clamp(0, self.n_sprites).long(), bboxes, class_ids
            else:
                return target.clamp(0, self.n_sprites).long()

        elif pred_instance_labels:
            label_layers = (
                torch.arange(1, L + 1, device=x.device)[(None,) * 5]
                .transpose(0, 5)
                .expand(-1, K, B, 1, H, W)
            )
            true_occ_grid = (occ_grid > 0.5).float()
            target = self.compose(
                label_layers,
                (tsf_masks > 0.5).long(),
                true_occ_grid,
                class_prob=class_oh,
            ).squeeze(1)
            target = target.clamp(0, L).long()
            if not with_bkg and class_oh is not None:
                bkg_idx = target == 0
                tsf_layers = (tsf_layers * class_oh[..., None, None, None]).sum(axis=1)
                new_target = ((tsf_layers - x) ** 2).sum(2).argmin(0).long() + 1
                target[bkg_idx] = new_target[bkg_idx]
            return target

        else:
            occ_grid = (occ_grid > 0.5).float() if hard_occ_grid else occ_grid
            tsf_layers, tsf_masks = tsf_layers.clamp(0, 1), tsf_masks.clamp(0, 1)
            if tsf_bkgs is not None:
                tsf_bkgs = tsf_bkgs.clamp(0, 1)
            target = self.compose(tsf_layers, tsf_masks, occ_grid, tsf_bkgs, class_prob)
            if class_prob is not None:
                target = target.unsqueeze(1)

            if with_composition:
                compo = []
                for k in range(L):
                    compo += [
                        tsf_layers[k].transpose(0, 1),
                        tsf_masks[k].transpose(0, 1),
                    ]
                if self.learn_backgrounds:
                    compo.insert(2, tsf_bkgs.transpose(0, 1))
                return target, compo
            else:
                return target

    def step(self):
        self.cur_epoch += 1
        [tsf.step() for tsf in self.sprite_transformers]
        if hasattr(self, "layer_transformer"):
            self.layer_transformer.step()
        if self.learn_backgrounds:
            self.bkg_transformer.step()

    def set_optimizer(self, opt):
        self.optimizer = opt
        [tsf.set_optimizer(opt) for tsf in self.sprite_transformers]
        if hasattr(self, "layer_transformer"):
            self.layer_transformer.set_optimizer(opt)
        if self.learn_backgrounds:
            self.bkg_transformer.set_optimizer(opt)

    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in state_dict.items():
            if name in state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                if "activations" in name and state[name].shape != param.shape:
                    state[name].copy_(
                        torch.Tensor([True] * state[name].size(0)).to(param.device)
                    )
                else:
                    state[name].copy_(param)
            elif name == "prototypes":
                state["prototype_params"].copy_(param)
            elif name == "backgrounds":
                state["bkg_params"].copy_(param)
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f"load_state_dict: {unloaded_params} not found")

    def reassign_empty_clusters(self, proportions):
        if not self._reassign_cluster or self.are_sprite_frozen:
            return [], 0
        if self.add_empty_sprite:
            proportions = proportions[:-1] / max(proportions[:-1])

        N, threshold = len(proportions), self.empty_cluster_threshold
        reassigned = []
        idx = torch.argmax(proportions).item()
        for i in range(N):
            if proportions[i] < threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)
                break  # if one cluster is split, stop reassigning
        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)

        return reassigned, idx

    def restart_branch_from(self, i, j):
        if hasattr(self, "generator"):
            self.latent_params[i].data.copy_(
                copy_with_noise(self.latent_params[j], NOISE_SCALE)
            )
            params = [self.latent_params]
        else:
            self.mask_params[i].data.copy_(self.mask_params[j].detach().clone())
            params = [self.mask_params]
            if not self.freeze_frg:
                self.prototype_params[i].data.copy_(
                    copy_with_noise(self.prototype_params[j], NOISE_SCALE)
                )
                params.extend([self.prototype_params])
        [
            tsf.restart_branch_from(i, j, noise_scale=0)
            for tsf in self.sprite_transformers
        ]

        if hasattr(self, "optimizer"):
            opt = self.optimizer
            if isinstance(opt, (Adam,)):
                for param in params:
                    opt.state[param]["exp_avg"][i] = opt.state[param]["exp_avg"][j]
                    opt.state[param]["exp_avg_sq"][i] = opt.state[param]["exp_avg_sq"][
                        j
                    ]
            elif isinstance(opt, (RMSprop,)):
                for param in params:
                    opt.state[param]["square_avg"][i] = opt.state[param]["square_avg"][
                        j
                    ]
            else:
                raise NotImplementedError(
                    "unknown optimizer: you should define how to reinstanciate statistics if any"
                )
