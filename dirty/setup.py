from setuptools import find_namespace_packages, setup

with open("README.md", "r") as f:
    README = f.read()

setup(
    name="cmu-dirty",
    version="0.0.0",
    author="CMU STRUDEL",
    description="CMU-STRUDEL DIRTY tools for variable type and name prediction",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/CMUSTRUDEL/DIRTY/",
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},  # the root package '' corresponds to the src dir
    # include_package_data=True,
    zip_safe=False,
    install_requires=[
        "jsonnet~=0.17.0",
        "numpy~=1.19.5",
        "pytorch_lightning~=1.2.10",
        "sentencepiece~=0.1.95",
        "torch~=1.8.1",
        "ujson~=4.0.2",
        "wandb~=0.10.29",
        "webdataset~=0.1.6",
        "docopt~=0.6.2",
        "scikit-learn~=0.24.2",
        "csvnpm-utils~=0.0.0",
    ],
    extras_require={
        "test": [
            "pytest-cov~=2.8.1",
            "pytest~=6.2.4",
            "mypy==0.812",
            "darglint~=1.8.0",
            "pipdeptree~=2.0.0",
            "safety~=1.9.0",
            "coverage==5.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "dirty-exp = dirty.exp:main",
            "dirty-evaluate = dirty.utils.evaluate:main",
        ]
    },
)
