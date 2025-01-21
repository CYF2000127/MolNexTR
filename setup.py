import setuptools


setuptools.setup(
    name="MolNexTR",
    version="1.0.0",
    description="MolNexTR, a novel graph generation model",
    entry_points={
        "console_scripts": ["molnextr = MolNexTR.molnextr:main"],
    },
    url="https://github.com/Kohulan/MolNexTR",
    packages=setuptools.find_packages(),
    license="Apache License 2.0",
    install_requires=[
        "pytorch-lightning",
        "opencv-python",
        "pystow",
        "pillow",
        "pyparsing",
        "six",
        "albumentations==1.1.0",
        "SmilesPE",
        "timm==0.4.12",
        "pyonmttok==1.37.1",
        "OpenNMT-py==2.2.0",

    ],
    package_data={"MolNexTR": ["decoding/*.*", "indigo/*.*", "models/*.*","vocab/*.*"]},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)