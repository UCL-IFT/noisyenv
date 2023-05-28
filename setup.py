from setuptools import setup, find_packages

setup(
    name='noisyenv',
    version='0.1.0',
    author='Raad Khraishi',
    author_email='raad.khraishi@ucl.ac.uk',
    description='Simple noisy environment augmentation for reinforcement learning',
    packages=find_packages(),
    install_requires=[
        'gymnasium>=0.26.1',
        'numpy>=1.21.6'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)