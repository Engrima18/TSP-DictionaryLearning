from setuptools import setup, find_packages

# Read requirements.txt for dependencies
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()
    
setup(
    name='tsplearn',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.7',
    # entry_points={
    #     'console_scripts': [
    #         '',
    #     ],
    # },
    author='Engrima18',
    author_email='engrima2000@gmail.com',
    description='Topological Signal Processing Dictionary Learning Framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords='',
    url='https://github.com/Engrima18/TSP-DictionaryLearning'
)
