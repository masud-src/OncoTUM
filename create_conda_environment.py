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
                if channel == "pip":
                    print(f"Installing {package} via pip in environment '{env_name}'...")
                    subprocess.run(
                        ["pip", "install", package],
                        check=True,
                        env={"CONDA_DEFAULT_ENV": env_name}  # Ensure pip installs in the correct Conda env
                    )
                else:
                    print(f"Installing {package} from {channel} in environment '{env_name}'...")
                    subprocess.run(
                        ["conda", "install", "-n", env_name, "-c", channel, "-y", package],
                        check=True
                    )
                print(f"{package} installed successfully.")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package} in environment '{env_name}'.")
        else:
            print(f"{package} is already installed in '{env_name}'.")



def create_new_env(env_name, packages_with_channels) -> None:
    """Create a new isolated environment with specified packages from their respective channels."""
    try:
        # Construct the command for conda packages
        conda_command = ["conda", "create", "-n", env_name, "-y"]
        pip_packages = []

        for package, channel in packages_with_channels:
            if channel == "pip":
                pip_packages.append(package)  # Collect pip packages separately
            else:
                conda_command.extend(["-c", channel, package])

        # Create the environment with conda packages
        subprocess.run(conda_command, check=True)

        # Install pip packages in the new environment
        if pip_packages:
            print(f"Installing pip packages in '{env_name}'...")
            for pip_package in pip_packages:
                subprocess.run(
                    ["pip", "install", pip_package],
                    check=True,
                    env={"CONDA_DEFAULT_ENV": env_name}  # Ensure pip installs in the correct Conda env
                )
        print(f"New environment '{env_name}' created with specified packages and channels.")
    except subprocess.CalledProcessError:
        print(f"Failed to create new environment '{env_name}'.")


def main():
    env_name = "oncofem"
    new_env_name = "oncostr"

    # Define each package with its specific channel
    # Note: For pip packages like Ranger, we set the channel to 'pip'
    packages_with_channels = [
        ("torch>=1.6.0", "conda-forge"),
        ("SimpleITK>=1.2.4", "conda-forge"),
        ("numpy>=1.18.4", "conda-forge"),
        ("scikit-learn>=0.23.1", "conda-forge"),
        ("pandas>=1.0.3", "conda-forge"),
        ("PyYAML>=5.3.1", "conda-forge"),
        ("matplotlib>=3.2.1", "conda-forge"),
        ("scipy>=1.4.1", "conda-forge"),
        ("git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer.git", "pip")
    ]

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
