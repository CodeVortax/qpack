from setuptools import setup, find_packages

setup(
    name="qpack",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "rich>=13.0.0",
        "arabic-reshaper>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "qpack = qpack.cli:main",
        ],
    },
    python_requires=">=3.8",
)
