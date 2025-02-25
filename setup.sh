#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$SCRIPT_DIR"/docker/utils.sh

echoTitle "REQUIREMENTS INSTALL"

colorEcho yellow "\nSystem packages ..."
sudo apt-get install redis-server python3.10 python3.10-venv python3.10-dev curl

colorEcho yellow "\nAPI virtual env ..."
python3.10 -m venv venv
venv/bin/pip install wheel>=0.45.1
venv/bin/pip install -r requirements-dev.txt
venv/bin/pip install python-dotenv

echoTitle "SET UP ENVIRONMENT VARIABLES"

ENV="$SCRIPT_DIR"/.env
DEV_ENV="$SCRIPT_DIR"/.env.dev

cp "$ENV".template "$ENV"
cp "$DEV_ENV".template "$DEV_ENV"

colorEcho yellow "\nSetting $ENV ..."
update_env "$ENV"

. "$ENV"

colorEcho yellow "\nSetting $DEV_ENV ..."
update_env "$DEV_ENV"

if [ "$TARGET" == "dev" ]; then
    echoTitle "PRE-COMMIT INSTALL"
    venv/bin/pip install pre-commit
    pre-commit install
fi

set_redis() {
    redis_psw="$1"
    REDIS_CONF=$(redis-cli INFO | grep config_file | awk -F: '{print $2}' | tr -d '[:space:]')
    colorEcho yellow "\n\nModifying Redis configuration file $REDIS_CONF ..."

    # use the same redis password for api and front
    $SED_CMD "s~^REDIS_PASSWORD=.*~REDIS_PASSWORD=\"$redis_psw\"~" "$FRONT_ENV"

    sudo $SED_CMD "s/\nrequirepass [^ ]*/requirepass $redis_psw/" "$REDIS_CONF"
    sudo $SED_CMD "s/# requirepass [^ ]*/requirepass $redis_psw/" "$REDIS_CONF"

    sudo systemctl restart redis
    # brew services restart redis # MacOs
}
# NOTE uncomment to use Redis password
# set_redis $REDIS_PASSWORD

echoTitle "DOWNLOADING SUBMODULES"
git submodule init
git submodule update

echoTitle "🎉 API SET UP COMPLETED ! 🎉"
