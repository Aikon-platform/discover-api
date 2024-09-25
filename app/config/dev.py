from .base import *


BASE_URL = f"http://localhost:{ENV('API_DEV_PORT', default=5000)}"
