from setuptools import setup, find_packages

setup(
    name='tsplearn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib',
        'pandas',
        'dill',
        'cvxpy',
        'seaborn'
        'networkx'
    ],
    entry_points={
        'console_scripts': [
            '',
        ],
    },
    author='Enrico Grimaldi',
    author_email='engrima2000@gmail.com',
    description='A short description of your project.',
    keywords='Some keywords relevant to your project',
    url='https://github.com/Engrima18/TSP-DictionaryLearning',  
)
