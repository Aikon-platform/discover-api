# import os
# import sys
#
# # Add the package root to sys.path
# package_root = os.path.dirname(os.path.abspath(__file__))
# if package_root not in sys.path:
#     sys.path.append(package_root)


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from .registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def build_model(args):
    from .dino.dino import build_dino

    return build_dino(args)
