from setuptools import find_packages, setup

setup(
    name="MOVE",
    description="Multi-omics variational autoencoder",
    url="https://github.com/RasmussenLab/MOVE",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    classifiers=[
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
