from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup( 
    name="antifragile_vbnf",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Antifragile Conditional Autoregressive Latent Normalizing Flows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/antifragile-vbnf",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/antifragile-vbnf/issues",
        "Documentation": "https://github.com/yourusername/antifragile-vbnf#readme",
        "Source Code": "https://github.com/yourusername/antifragile-vbnf",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "pre-commit>=2.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "notebook>=6.0",
        ],
        "visualization": [
            "plotly>=5.0",
            "bokeh>=2.0",
            "ipywidgets>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "antifragile-vbnf=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "antifragile_vbnf": [
            "configs/*.json",
            "data/*.csv",
        ],
    },
    zip_safe=False,
    keywords=[
        "antifragile",
        "normalizing flows",
        "machine learning",
        "robustness",
        "volatility",
        "stress testing",
        "conditional flows",
        "deep learning",
        "pytorch",
    ],
)