from setuptools import setup, find_packages

setup(
    name="biorsp",
    version="1.0.0-alpha",
    description="Enhancing single-cell gene expression analysis by simulating radar scanning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Zeyu Yao",
    author_email="cytronicoder@gmail.com",
    url="https://github.com/alphamind-club/biorsp/",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "umap-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
