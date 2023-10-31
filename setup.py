from setuptools import setup, find_packages

setup(
    name='after_subtle',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy','pandas','matplotlib','scipy','PyYAML'],
    python_requires='>=3.7'
)