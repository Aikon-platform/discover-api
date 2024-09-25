from importlib import import_module

def auto_import_apps(flask_app, INSTALLED_APPS, package):
    for app in INSTALLED_APPS:
        # import routes and tasks for each installed app
        # exec(f"from .{app}.routes import *")

        module = import_module(f".{app}.routes", package=package)
        if hasattr(module, "blueprint"):
            flask_app.register_blueprint(module.blueprint)

        module = import_module(f".{app}.tasks", package=package)
