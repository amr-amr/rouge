from setuptools import setup

__version__ = '0.0.1'

setup(
    name='rouge',
    version=__version__,
    description='Pure python implementation of ROUGE, designed for extendability and use in an RL setting',
    url='https://github.com/amr-amr/rouge.git',
    author='amr-amr',
    author_email='amr-amr@amr.com',
    keywords=[
        'NLP', 'ROUGE',
        'natural language processing',
        'computational linguistics',
        'automatic summarization',
    ],
    packages=[
        'rouge',
        'rouge.utils',
    ],
    package_data={
        'rouge': ['data/*.txt',
                  'data/WordNet-2.0-Exceptions/*.exc'],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Text Processing :: Linguistic',
        "Operating System :: OS Independent",
    ],
    license='LICENSE.txt',
    long_description=open('README.md').read(),
)