"""
   setuptools-based setup module

"""

from setuptools import setup

required_libraries=[
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "jupyter",
    "scipy",
    "annoy",
    "seaborn",
]

setup(
    name="clustering_by_classification",
    version="0.1.0",
    description="using arbitrary classifiers to cluster data",
    url="https://github.com/jattenberg/clustering_by_classification",
    install_requires=required_libraries
)
