from setuptools import setup, find_packages

setup(
    name="ethoPyRT",
    version="0.1.23",
    author="Maximilian Knoll",
    author_email="m.knoll@dkfz.de",
    description="Helper function for analysis of Varian ETHOS data",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mknoll/ethoPy",
    download_url="https://github.com/mknoll/ethoPyRT/archive/refs/tags/v_01.tar.gz",
    license="MIT",
    keywords="Radiotherapy, ETHOS",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "nibabel",
        "numpy",
        "platipy",
        #"pyradiomics",
        "SimpleITK",
        "dcmrtstruct2nii @ git+https://github.com/mknoll/dcmrtstruct2nii.git@master"
    ],
)

