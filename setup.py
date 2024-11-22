from setuptools import setup, find_packages

setup(
    name="oncotum",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fslpy",
        "nibabel",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8"
        ]
    },
    python_requires=">=3.6",
    author="Marlon Suditsch",
    author_email="m.suditsch@outlook.com",
    description="Tumor Segmentation of Onco",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/masud-src/OncoTUM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL 3",
        "Operating System :: OS Independent",
    ],
)
