from setuptools import find_namespace_packages, setup

with open("README.md", "r") as f:
    README = f.read()

setup(
    name="csvnpm-utils",
    version="0.0.0",
    author="CMU STRUDEL",
    description="CMU-STRUDEL Variable Name Prediction Model Utilities",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/CMUSTRUDEL/DIRTY/",
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},  # the root package '' corresponds to the src dir
    # include_package_data=True,
    zip_safe=False,
    install_requires=["pygments~=2.9.0", "tqdm~=4.60.0", "jsonlines~=2.0.0"],
    extras_require={
        "test": [
            "pytest-cov~=2.8.1",
            "pytest~=5.2.2",
            "mypy==0.782",
            "darglint~=1.4.1",
            "pipdeptree~=1.0.0",
            "safety~=1.9.0",
            "coverage==5.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "csvnpm-decompiler = csvnpm.dataset_gen.generate:main",
            "csvnpm-download = csvnpm.download:main",
        ]
    },
)
