from setuptools import setup, find_packages

setup(
    name="MachineLearningScripts",      
    version="0.1.0",
    author="James Lynch",
    description="From‐scratch ML algorithms",
    packages=find_packages(),           # this will pick up MachineLearningScripts and its sub-packages
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",                  # any runtime deps you actually use
    ],
)