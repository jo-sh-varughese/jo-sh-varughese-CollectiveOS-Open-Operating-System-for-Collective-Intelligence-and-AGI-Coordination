from setuptools import setup, find_packages

setup(
    name="collectiveos-bench",
    version="0.1.0",
    author="Johan Varughese",
    description="A Benchmark Suite for Emergent Cooperation, Institutions, "
                "and Collective Intelligence in Multi-Agent Systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/johanvarughese/collectiveos-bench",
    packages=find_packages(exclude=["tests*", "paper*", "docs*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "gym>=0.26.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=[
        "multi-agent reinforcement learning",
        "emergent cooperation",
        "collective intelligence",
        "institutional dynamics",
        "benchmark",
        "social dilemma",
    ],
)
