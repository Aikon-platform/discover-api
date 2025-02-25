# Deploy with Docker

Requirements:
- Docker
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Python >=3.10

Create a user (replace `<docker-user>` by the name you want) to run the Docker
```bash
# OPTIONAL: create a user to run the docker
DOCKER_USER=<docker-user>

sudo useradd -m $DOCKER_USER # Create user
sudo passwd $DOCKER_USER # Set password
sudo usermod -aG sudo $DOCKER_USER # Add user to sudo group

sudo -iu $DOCKER_USER # Connect as docker user
sudo usermod -aG docker $USER # add user to docker group
su - ${USER} # Reload session for the action to take effect
```

Configure SSH connexion to GitHub for user:
- Generate key with `ssh-keygen`
- Copy key `cat ~/.ssh/id_ed25519.pub`
- [Add SSH key](https://github.com/settings/ssh/new) to your GitHub account

Clone and init submodules
```bash
git clone git@github.com:Aikon-platform/aikon-api.git
cd aikon-api/

# OPTIONAL: if you are deploying demos using submodules (like dticlustering and vectorization)
git submodule init
git submodule update
```

Copy the file `.env` to a file `.env.prod` and change `TARGET=prod`.

```bash
cp .env.template .env.prod
sed -i -e 's/^TARGET=.*/TARGET="prod"/' .env.prod

# OPTIONAL: modify other variables, notably INSTALLED_APPS
vi .env.prod
```

In [`docker.sh`](docker.sh), modify the variables depending on your setup:
- `DATA_FOLDER`: absolute path to directory where results are stored
- `DEMO_UID`: Universally Unique Identifier of the `$DOCKER_USER` (`id -u <docker-user>`)
- `DEVICE_NB`: GPU number to be used by container (get available GPUs with `nvidia-smi`)
- `CUDA_HOME`: path to CUDA installation (e.g. `/usr/local/cuda-11.1`)

To find your `CUDA_HOME` (usually located either in `/usr/local/cuda` or `/usr/lib/cuda`):
```bash
# find CUDA version with (pay attention to version mismatches)
nvcc --version
nvidia-smi

# CUDA_HOME is usually parent dir of
which nvcc
```

Create the folder matching `DATA_FOLDER` in the `docker.sh` to store results of experiments and set its permissions:
```bash
RESULT_PATH=<path/to/results/> # e.g. /media/$DOCKER_USER/

mkdir $RESULT_PATH
sudo chmod o+X </path/to>
sudo chmod -R u+rwX <path/to/results/>
sudo chown -R $DOCKER_USER:$DOCKER_USER $RESULT_PATH
```

#### Download models

[//]: # (TODO: Add instructions to download models)

#### Build Docker

Build the docker using the premade script:

```bash
bash docker.sh rebuild
```

To compile cuda operators for `vectorization` / `regions`, once built:
```bash
docker exec -it demoapi /bin/bash
# inside the container
/home/demoapi# source venv/bin/activate

# for vectorization
/home/demoapi# python /home/${USER}/api/app/vectorization/lib/src/models/dino/ops/setup.py build install
/home/demoapi# python /home/${USER}/api/app/vectorization/lib/src/models/dino/ops/test.py

# for regions
/home/demoapi# cd /home/${USER}/api/app/regions/lib/line_predictor/
/home/demoapi# python ./dino/ops/setup.py build install
```
Then restart the container with `docker restart aikondemo`

Inside `$DATA_FOLDER/data`, add models and necessary files for the demos inside their respective sub-folders.

It should have started the docker, check it is the case with:
- `docker logs aikondemo --tail 50`: show last 50 log messages
- `docker ps`: show running docker containers
- `curl 127.0.0.1:8001/<installed_app>/monitor`: show if container receives requests
- `docker exec aikondemo /bin/nvidia-smi`: checks that docker communicates with nvidia
- `docker exec -it aikondemo /bin/bash`: enter the docker container

The API is now accessible locally at `http://localhost:8001`.

<details>
  <summary>#### Secure connection with [spiped](https://www.tarsnap.com/spiped.html)</summary>

> ⚠️ If you are not using `spiped` modify the `docker.sh` file to expose `0.0.0.0:8001:8001` instead of `127.0.0.1:8001:8001`

A good thing is to tunnel securely the connection between API and front. For `discover-demo.enpc.fr`, it is done with `spiped`, based on [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-encrypt-traffic-to-redis-with-spiped-on-ubuntu-16-04).
The Docker process running on port `localhost:8001` is encrypted and redirected to port `8080`.
The front server decrypts the traffic and redirects it to `localhost:8001`.

```bash
sudo apt-get update
sudo apt-get install spiped
sudo mkdir /etc/spiped
sudo dd if=/dev/urandom of=/etc/spiped/discover.key bs=32 count=1 # Generate key
sudo chmod 644 /etc/spiped/discover.key
```

Create service config file for spiped (`sudo vi /etc/systemd/system/spiped-discover.service`):
- Get `<docker-ip>` with `docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' aikondemo` or use `127.0.0.1`
- Pick Docker port (here `8001`) depending on `EXPOSE` in [`Dockerfile`](Dockerfile)

```bash
[Unit]
Description=Spiped connection for docker container
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=300

[Service]
# Redirects <docker-ip>:8001 to 0.0.0.0:8080 and encrypts it with discover.key on the way
ExecStart=/usr/bin/spiped -F -d -s [0.0.0.0]:8080 -t [<docker-ip>]:8001 -k /etc/spiped/discover.key
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Open port to external requests and enable spiped service
```bash
sudo ufw allow 8080 # open firewall and allow incoming traffic on port 8080

sudo systemctl daemon-reload
sudo systemctl start spiped-discover.service
sudo systemctl enable spiped-discover.service
```

Transfer key to front ([`spiped`](https://github.com/tarsnap/spiped) uses symmetric encryption with same keys on both servers)
```bash
# on your local machine
scp <gpu-host>:/etc/spiped/discover.key <front-host>:~ # Assuming you have configured direct ssh connection to front and gpu

# on front machine
ssh <front-host>
sudo apt-get install spiped
sudo chmod 644 ~/discover.key
sudo mkdir /etc/spiped
sudo cp ~/discover.key /etc/spiped/ # Copy key to spiped folder
```

Create service config file for spiped on front machine (`sudo vi /etc/systemd/system/spiped-connect.service`)
- Get `<gpu-ip>` with `hostname -I` on the machine where is deployed the API.

⚠️ Note to match the output IP (`127.0.0.1:8001` in this example) to the `API_URL` in [`front/.env`](../front/.env)

```bash
[Unit]
Description=Spiped connection to API
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=300

[Service]
# Redirects <gpu-ip>:8080 output to 127.0.0.1:8001 and decrypts it with discover.key on the way
ExecStart=/usr/bin/spiped -F -e -s [127.0.0.1]:8001 -t [<gpu-ip>]:8080 -k /etc/spiped/discover.key
Restart=Always

[Install]
WantedBy=multi-user.target
```

Enable service
```bash
sudo systemctl daemon-reload
sudo systemctl start spiped-connect.service
sudo systemctl enable spiped-connect.service
```

Test connexion between worker and front
```bash
curl --http0.9 <gpu-ip>:8080/<installed_app>/monitor # outputs the encrypted message
curl localhost:8001/<installed_app>/monitor # outputs decrypted message
```
</details>

### Update

Just run:

```bash
bash docker.sh pull
```

**Note:** as a redis server is encapsulated inside the docker, its data is **non-persistent**: any task scheduled before a `bash docker.sh <anything>` will be forgotten. Result files are kept, though, as they are in the persistent storage.

*Note 2:* Docker won't be able to access the host's `http://localhost:8000/` easily, so it is not advised to use the Docker build to develop if that's the only way to access the frontend.

[//]: # (Configure Redis)
[//]: # (```bash)
[//]: # (# Find config file)
[//]: # (sudo find / -name redis.)
[//]: # (vi <path/to/redis.conf>)
[//]: # (```)
[//]: # (Find &#40;`/` command then type `requirepass`&#41; and modify directive &#40;uncomment and set password&#41;:)
[//]: # (```bash)
[//]: # (requirepass <redis_password>)
[//]: # (```)
