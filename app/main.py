"""
A script that initializes the Flask app and the Dramatiq broker.
"""

import os

from . import config
from flask import Flask

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq_abort import Abortable, backends
from dramatiq.middleware import CurrentMessage, Prometheus
from dramatiq.results.backends import RedisBackend
from .shared.utils.logging import LoggedResults
from .shared.utils.modular import auto_import_apps

# Flask setup
app = Flask(__name__)
app.config.from_object(config.FLASK_CONFIG)

# Dramatiq setup
broker = RedisBroker(url=config.BROKER_URL)

# Remove the Prometheus middleware if it's added by default
# broker.middleware = [m for m in broker.middleware if not isinstance(m, Prometheus)]

# if os.getenv("TESTS") == "true":
#     import os
#     from dramatiq.brokers.stub import StubBroker
#     broker = StubBroker()
#     broker.emit_after("process_boot")

event_backend = backends.RedisBackend(client=broker.client)
abortable = Abortable(backend=event_backend)

result_backend = RedisBackend(client=broker.client)
results = LoggedResults(backend=result_backend)

broker.add_middleware(abortable)
broker.add_middleware(CurrentMessage())
broker.add_middleware(results)

dramatiq.set_broker(broker)

# Import routes and tasks
auto_import_apps(app, config.INSTALLED_APPS, __package__)
