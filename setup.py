from setuptools import setup, find_packages

setup(
    name='trocr-train',
    version='0.0.1',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'datasets',
        'torch',
        'scikit-learn',
        'Pillow',
        'numpy',
        'transformers'
    ],
    entry_points={
        'console_scripts': [
            'trocr-train=main:main',
        ],
    },
)
