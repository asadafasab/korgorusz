from setuptools import find_packages
from setuptools import setup


setup(
    name="korgorusz",
    version="0.1.0",
    author="asadafasab",
    author_email="markedmarkv@gmail.com",
    url="https://github.com/asadafasab/korgorusz",
    description="Simple set of machine learning alghoritms",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["numpy"],
    setup_requires=['wheel'],
    extras_require={"dev": ["pytest", "black", "mypy"]},
)
