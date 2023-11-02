from setuptools import setup

setup(
    name="hik",
    version="0.0.1",
    author="Julian Tanke",
    author_email="tanke@iai.uni-bonn.de",
    license="MIT",
    packages=[
        "hik",
        "hik/vis",
        "hik/data",
        "hik/transforms",
        "hik/eval",
    ],
    install_requires=[
        "numpy",
        "numba",
        "einops",
        "matplotlib",
        "scipy",
        "tqdm",
        "scikit-learn",
        "smplx",
        "opencv-python",
    ],
)
