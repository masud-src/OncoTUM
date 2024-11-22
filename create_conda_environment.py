import subprocess
from typing import Union


def is_package_installed(env_name, package_name) -> Union[str, bool]:
    """Check if a package is installed in a given conda environment."""
    try:
        command = ["conda", "list", "-n", env_name, package_name]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return package_name in result.stdout
    except subprocess.CalledProcessError:
        return False


def add_packages_to_env(env_name, packages_with_channels) -> None:
    """Add packages to an existing environment with specified channels if not already installed."""
    for package, channel in packages_with_channels:
        if not is_package_installed(env_name, package):
            try:
                print(f"Installing {package} from {channel} in environment '{env_name}'...")
                subprocess.run(["conda", "install", "-n", env_name, "-c", channel, "-y", package], check=True)
                print(f"{package} installed successfully.")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package} in environment '{env_name}'.")
        else:
            print(f"{package} is already installed in '{env_name}'.")


def create_new_env(env_name, packages_with_channels) -> None:
    """Create a new isolated environment with specified packages from their respective channels."""
    try:
        # Construct the command with each package and its specific channel
        command = ["conda", "create", "-n", env_name, "-y"]
        for package, channel in packages_with_channels:
            command.extend(["-c", channel, package])

        subprocess.run(command, check=True)
        print(f"New environment '{env_name}' created with specified packages and channels.")
    except subprocess.CalledProcessError:
        print(f"Failed to create new environment '{env_name}'.")


def main():
    env_name = "oncofem"
    new_env_name = "oncotum"
    # Define each package with its specific channel
    packages_with_channels = [("nibabel", "conda-forge")]

    action = input(
        f"Would you like to (1) add packages to the existing environment '{env_name}' or (2) create a new environment '{new_env_name}'? Enter 1 or 2: ").strip()

    if action == '1':
        add_packages_to_env(env_name, packages_with_channels)
    elif action == '2':
        create_new_env(new_env_name, packages_with_channels)
    else:
        print("Invalid option. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
