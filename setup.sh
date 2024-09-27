#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

colorEcho() {
    Color_Off="\033[0m"
    Red="\033[1;91m"        # Red
    Green="\033[1;92m"      # Green
    Yellow="\033[1;93m"     # Yellow
    Blue="\033[1;94m"       # Blue
    Purple="\033[1;95m"     # Purple
    Cyan="\033[1;96m"       # Cyan

    case "$1" in
        "green") echo -e "$Green$2$Color_Off";;
        "red") echo -e "$Red$2$Color_Off";;
        "blue") echo -e "$Blue$2$Color_Off";;
        "yellow") echo -e "$Yellow$2$Color_Off";;
        "purple") echo -e "$Purple$2$Color_Off";;
        "cyan") echo -e "$Cyan$2$Color_Off";;
        *) echo "$2";;
    esac
}

echoTitle(){
    sep_line="========================================"
    len_title=${#1}

    if [ "$len_title" -gt 40 ]; then
        sep_line=$(printf "%0.s=" $(seq 1 $len_title))
        title="$1"
    else
        diff=$((38 - len_title))
        half_diff=$((diff / 2))
        sep=$(printf "%0.s=" $(seq 1 $half_diff))

        if [ $((diff % 2)) -ne 0 ]; then
            title="$sep $1 $sep="
        else
            title="$sep $1 $sep"
        fi
    fi

    colorEcho purple "\n\n$sep_line\n$title\n$sep_line"
}

echoTitle "REQUIREMENTS INSTALL"

colorEcho yellow "\nSystem packages ..."
sudo apt-get install redis-server python3.10 python3.10-venv python3.10-dev curl

colorEcho yellow "\nAPI virtual env ..."
python3.10 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install python-dotenv

echoTitle "SET UP ENVIRONMENT VARIABLES"

generate_random_string() {
    echo "$(openssl rand -base64 32 | tr -d '/\n')"
}

prompt_user() {
    env_var=$(colorEcho 'red' "$1")
    default_val="$2"
    current_val="$3"
    desc="$4"

    if [ "$2" != "$3" ]; then
        default="Press enter for $(colorEcho 'cyan' "$default_val")"
    elif [ -n "$current_val" ]; then
        default="Press enter to keep $(colorEcho 'cyan' "$current_val")"
        default_val=$current_val
    fi

    read -p "$env_var $desc"$'\n'"$default: " value
    echo "${value:-$default_val}"
}

get_env_value() {
    param=$1
    env_file=$2
    value=$(awk -F= -v param="$param" '/^[^#]/ && $1 == param {gsub(/"/, "", $2); print $2}' "$env_file")
    echo "$value"
}

get_os() {
    unameOut="$(uname -s)"
    case "${unameOut}" in
        Linux*)     os=Linux;;
        Darwin*)    os=Mac;;
        CYGWIN*)    os=Cygwin;;
        MINGW*)     os=MinGw;;
        MSYS_NT*)   os=Git;;
        *)          os="UNKNOWN:${unameOut}"
    esac
    echo "${os}"
}

update_env() {
    env_file=$1
    IFS=$'\n' read -d '' -r -a lines < "$env_file"  # Read file into array
    for line in "${lines[@]}"; do
        if [[ $line =~ ^[^#]*= ]]; then
            param=$(echo "$line" | cut -d'=' -f1)
            current_val=$(get_env_value "$param" "$env_file")

            # Extract description from previous line if it exists
            desc=""
            if [[ $prev_line =~ ^# ]]; then
                desc=$(echo "$prev_line" | sed 's/^#\s*//')
            fi

            case $param in
                *PASSWORD*)
                    default_val="$(generate_random_string)"
                    ;;
                *SECRET*)
                    default_val="$(generate_random_string)"
                    ;;
                *)
                    default_val="$current_val"
                    ;;
            esac

            new_value=$(prompt_user "$param" "$default_val" "$current_val" "$desc")
            sed -i -e "s~^$param=.*~$param=\"$new_value\"~" "$env_file"
        fi
        prev_line="$line"
    done
}

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
    sed -i '' -e "s~^REDIS_PASSWORD=.*~REDIS_PASSWORD=\"$redis_psw\"~" "$FRONT_ENV"

    sudo sed -i -e "s/\nrequirepass [^ ]*/requirepass $redis_psw/" "$REDIS_CONF"
    sudo sed -i -e "s/# requirepass [^ ]*/requirepass $redis_psw/" "$REDIS_CONF"

    sudo systemctl restart redis
    # brew services restart redis # MacOs
}
# NOTE uncomment to use Redis password
# set_redis $REDIS_PASSWORD

echoTitle "DOWNLOADING DTI SUBMODULE"
git submodule init
git submodule update

echoTitle "SETUP COMPLETE"
