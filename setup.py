from distutils.core import setup
from setuptools import find_packages

setup(
    name='act',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'act': ['config/*.json']
    },
    license='MIT License',
    # long_description=open('README.md').read(),
)