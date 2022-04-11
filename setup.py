"""Python setup.py for fno package"""
import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
  """Read the contents of a text file safely.
  >>> read("fno", "VERSION")
  '0.1.0'
  >>> read("README.md")
  ...
  """

  content = ""
  with io.open(
    os.path.join(os.path.dirname(__file__), *paths),
    encoding=kwargs.get("encoding", "utf8"),
  ) as open_file:
    content = open_file.read().strip()
  return content


def read_requirements(path):
  return [
    line.strip()
    for line in read(path).split("\n")
    if not line.startswith(('"', "#", "-", "git+"))
  ]

setup(
    name="fno",
    version=read("fno", "VERSION"),
    description="Fourier Neural Operator",
    url="https://github.com/astanziola/fno-jax",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Antonio Stanziola, UCL BUG",
    author_email="a.stanziola@ucl.ac.uk",
    packages=find_packages(exclude=["data", ".github", "scripts"]),
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    license="MIT License"
)
