from setuptools import setup, find_packages

setup(
    name="biorsp",
    version="0.1.0",
    description="BioRSP: Radar Scanning Plot for single-cell directional gene expression analysis",
    author="BioRSP Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "anndata",
        "scanpy",
        "scikit-learn",
        "tqdm",
    ],
    python_requires=">=3.8",
)
