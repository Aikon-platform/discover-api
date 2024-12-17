Usage
=====

Development
-----------

Copy the file ``.env.template`` to a file ``.env``. Change its content to match your setup (especially regarding the paths).

You need to install redis and python:

.. code-block:: bash

    sudo apt-get install redis-server python3-venv python3-dev


.. comment: Configure Redis
.. comment: 
.. code-block:: bash

    # Find config file
    sudo find / -name redis.conf
    vi <path/to/redis.conf>

.. comment: Find (``/`` command then type ``requirepass``) and modify directive (uncomment and set password):
.. code-block:: bash
    
    requirepass <redis_password>

You need to init the submodules (for ``dticlustering`` you need access to the `dti-sprites <https://github.com/sonatbaltaci/dti-sprites>`_ project):

.. code-block:: bash
    
    git submodule init
    git submodule update

    # once initiate, update the submodule code with
    git submodule update --remote


Create a python virtual environment and install the required packages:

.. code-block:: bash

    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt

You can now run the API worker:

.. code-block:: bash

    ./venv/bin/dramatiq app.main -p 1 -t 1

And the server:

.. code-block:: bash

    ./venv/bin/flask --app app.main run --debug


Adding new demo
^^^^^^^^^^^^^^^

1. Create a new demo folder, containing at least a ``__init__.py``, ``routes.py`` and ``tasks.py`` files
2. Add relevant variables in ``.env.template`` and generate the corresponding ``.env`` file
3. If necessary, configure a new xaccel redirection in the nginx configuration file (``docker-confs/nginx.conf``)
4. Add the demo name (i.e. folder name) to the list ``INSTALLED_APPS`` in ``.env``

Production deployment
---------------------

See ``README.md``