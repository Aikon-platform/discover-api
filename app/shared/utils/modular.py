"""
This module is used to auto import all installed apps
"""
from flask import Flask
from importlib import import_module
from typing import List

def auto_import_apps(flask_app: Flask, INSTALLED_APPS: List[str], package: str):
    """
    Auto import all installed apps, and register their blueprints and tasks.
    """
    for app in INSTALLED_APPS:
        # import routes and tasks for each installed app
        # exec(f"from .{app}.routes import *")

        module = import_module(f".{app}.routes", package=package)
        if hasattr(module, "blueprint"):
            flask_app.register_blueprint(module.blueprint)

        module = import_module(f".{app}.tasks", package=package)
