#!/bin/bash

# Prompt the user to choose the environment
echo "Choose the environment: (1) for OncoFEM, (2) for just pip installation"
read env_choice

if [[ "$env_choice" == "1" ]]; then
  conda activate oncofem
elif [[ "$env_choice" == "2" ]]; then
  echo "only pip installation is chosen."
else
  echo "Invalid choice. Please choose either '1' for OncoSTR or '2' for OncoFEM."
  exit 1
fi

pip install -r requirements.txt
python3 -m pip install .

#!/bin/bash

if [[ -z "${ONCOTUM_DIR}" ]]; then
    ONCOTUM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

USER_HOME="$HOME"

add_to_path_unix() {
    if ! grep -q "export ONCOTUM=" ~/.bashrc; then
        echo "export ONCOTUM=$ONCOTUM_DIR" >> ~/.bashrc
        echo "ONCOTUM has been added to your PATH."
        echo "Please run 'source ~/.bashrc' to apply the changes."
    else
        echo "ONCOTUM is already set in your PATH."
    fi
}

add_to_path_macos() {
    if ! grep -q "export ONCOTUM=" ~/.zshrc; then
        echo "export ONCOTUM=$ONCOTUM_DIR" >> ~/.zshrc
        echo 'export PATH="$ONCOTUM/bin:$PATH"' >> ~/.zshrc
        echo "ONCOTUM has been added to your PATH."
        echo "Please run 'source ~/.zshrc' to apply the changes."
    else
        echo "ONCOTUM is already set in your PATH."
    fi
}

add_to_path_windows() {
    local script_file="$HOME/set_config.bat"
    if ! grep -q "setx PATH" "$script_file" 2>/dev/null; then
        echo "@echo off" > "$script_file"
        echo "setx PATH \"%PATH%;$ONCOTUM_DIR\"" >> "$script_file"
        echo "ONCOTUM has been added to your PATH."
        echo "Please restart your command prompt to apply the changes."
    else
        echo "ONCOTUM is already set in your PATH."
    fi
}

create_config_file(){
    CONFIG_FILE="$ONCOTUM_DIR/config.ini"
    {
        echo "[directories]"
        echo "STUDIES_DIR: $USER_HOME/studies/"
        echo "RUN_DIR: $USER_HOME/run/"
        echo "[models]"
        echo "FULL_MODEL_DIR: $ONCOTUM_DIR/data/tumor_segmentation/full/hyperparam.yaml"
        echo "CYCLE_1_4_MODEL_T1_DIR: $ONCOTUM_DIR/data/tumor_segmentation/t1/hyperparam.yaml"
        echo "CYCLE_1_4_MODEL_T1GD_DIR: $ONCOTUM_DIR/data/tumor_segmentation/t1gd/hyperparam.yaml"
        echo "CYCLE_1_4_MODEL_T2_DIR: $ONCOTUM_DIR/data/tumor_segmentation/t2/hyperparam.yaml"
        echo "CYCLE_1_4_MODEL_FLAIR_DIR: $ONCOTUM_DIR/data/tumor_segmentation/flair/hyperparam.yaml"

    } > "$CONFIG_FILE"
    echo "Config file created."
}

case "$(uname -s)" in
    Linux*)     add_to_path_unix ;;
    Darwin*)    add_to_path_macos ;;
    *)          echo "Unsupported OS. Please add the ONCOTUM directory to your PATH manually." ;;
esac

if [[ "$OS" == "Windows_NT" ]]; then
    add_to_path_windows
fi

echo "Downloading weights started!"
curl --output data https://darus.uni-stuttgart.de/api/access/dataset/:persistentId/?persistentId=doi:10.18419/darus-4647
echo "Download weights complete!"

create_config_file

echo "Installation of oncostr is completed successfully!"
echo "Please open a new python terminal and run 'import oncotum' to check if oncotum is installed correctly."
exit 0