DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
API_ROOT="$(dirname "$DOCKER_DIR")"

source "$DOCKER_DIR"/utils.sh

if [ ! -f "$APP_ROOT"/.env ]; then
    cp "$API_ROOT"/.env.template "$API_ROOT"/.env.prod
    update_env "$API_ROOT"/.env.prod
fi

# if docker/.env does not exist, create it
if [ ! -f "$DOCKER_DIR"/.env ]; then
    cp "$DOCKER_DIR"/.env.template "$DOCKER_DIR"/.env
    update_env "$DOCKER_DIR"/.env
fi

source "$API_ROOT"/.env.prod
source "$DOCKER_DIR"/.env

# if $DATA_FOLDER does not exist
if [ ! -d "$DATA_FOLDER" ]; then
    # Create $DATA_FOLDER folder with right permissions for user $USERID
    sudo mkdir -p "$DATA_FOLDER"
    sudo chown -R "$USERID:$USERID" "$DATA_FOLDER"
    sudo chmod -R 775 "$DATA_FOLDER"
fi
