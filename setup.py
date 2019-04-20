from setuptools import setup, find_packages

setup(
    name='smcmodel',
    packages=find_packages(),
    version='0.0.1',
    include_package_data=True,
    description='Provide estimation and simulation capabilities for sequential Monte Carlo (SMC) models',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/smcmodel',
    author='Ted Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    install_requires=[
        'tensorflow==1.13.1',
        'numpy==1.16.2'
    ],
    keywords=['bayes', 'smc'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
