import subprocess
from collections import defaultdict
from typing import List, Tuple, Union


def is_package_installed(env_name: str, package_name: str) -> Union[str, bool]:
    """Check if a package is installed in a given conda environment."""
    try:
        command = ["conda", "list", "-n", env_name, package_name]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return package_name in result.stdout
    except subprocess.CalledProcessError:
        return False


def install_conda_packages(env_name: str, channel: str, packages: List[str]) -> None:
    """Install a list of packages from a specific channel."""
    try:
        print(f"Installing packages from channel '{channel}' in environment '{env_name}': {packages}")
        command = ["conda", "install", "-n", env_name, "-y", "-c", channel] + packages
        subprocess.run(command, check=True)
        print(f"Packages installed successfully from '{channel}'.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages from channel '{channel}': {e}")


def install_pip_packages(env_name: str, packages: List[str]) -> None:
    """Install pip packages in a given conda environment."""
    try:
        print(f"Installing pip packages in environment '{env_name}': {packages}")
        command = ["pip", "install"] + packages
        subprocess.run(command, check=True, env={"CONDA_DEFAULT_ENV": env_name})
        print("Pip packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install pip packages: {e}")


def add_packages_to_env(env_name: str, packages_with_channels: List[Tuple[str, str]]) -> None:
    """Add packages to an existing environment, grouped by channels."""
    grouped_packages = defaultdict(list)

    # Group packages by channel
    for package, channel in packages_with_channels:
        grouped_packages[channel].append(package)

    for channel, packages in grouped_packages.items():
        if channel == "pip":
            install_pip_packages(env_name, packages)
        else:
            install_conda_packages(env_name, channel, packages)


def create_new_env(env_name: str, packages_with_channels: List[Tuple[str, str]]) -> None:
    """Create a new conda environment and install packages grouped by channels."""
    try:
        print(f"Creating new environment '{env_name}'...")
        subprocess.run(["conda", "create", "-n", env_name, "-y"], check=True)
        print(f"Environment '{env_name}' created successfully.")
        add_packages_to_env(env_name, packages_with_channels)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create new environment '{env_name}': {e}")


def main():
    env_name = "oncofem"
    new_env_name = "oncostr"

    # Define each package with its specific channel
    packages_with_channels = [
        ("torch>=1.6.0", "conda-forge"),
        ("SimpleITK>=1.2.4", "conda-forge"),
        ("numpy>=1.18.4", "conda-forge"),
        ("scikit-learn>=0.23.1", "conda-forge"),
        ("pandas>=1.0.3", "conda-forge"),
        ("PyYAML>=5.3.1", "conda-forge"),
        ("matplotlib>=3.2.1", "conda-forge"),
        ("scipy>=1.4.1", "conda-forge"),
        ("git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer.git", "pip"),
    ]

    action = input(
        f"Would you like to (1) add packages to the existing environment '{env_name}' "
        f"or (2) create a new environment '{new_env_name}'? Enter 1 or 2: ").strip()

    if action == '1':
        add_packages_to_env(env_name, packages_with_channels)
    elif action == '2':
        create_new_env(new_env_name, packages_with_channels)
    else:
        print("Invalid option. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
