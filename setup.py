from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytorch-autotune",
    version="1.0.1",  # Bumped version for update
    author="Chinmay Shrivastava",
    author_email="cshrivastava2000@gmail.com",
    description="Automatic 4x training speedup for PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonSnow1807/pytorch-autotune",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.19.0",
    ],
    keywords="pytorch optimization speedup training acceleration autotune",
    project_urls={
        "Bug Reports": "https://github.com/JonSnow1807/pytorch-autotune/issues",
        "Source": "https://github.com/JonSnow1807/pytorch-autotune",
        "Documentation": "https://github.com/JonSnow1807/pytorch-autotune#readme",
    },
)