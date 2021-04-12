from setuptools import setup, find_packages

with open('requirements.txt', 'r') as fin:
    reqs = fin.read().split('\n')

setup(
    name='blocksparsednn',
    version='1.0.0',
    description='Block Sparse Deep Neural Network Training.',
    packages=find_packages(),
    install_requires=reqs
)
