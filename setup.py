import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

# Dependencies (format is 'PYPI_PACKAGE_NAME[>=]=VERSION_NUMBER')
BASE_DEPENDENCIES = [
    'wf-datetime-conversion>=0.0.1',
    'tensorflow>=2.2',
    'numpy>=1.19.0',
    'python-dateutil>=2.8.1',
    'tqdm>=4.46.1'
]

# TEST_DEPENDENCIES = [
# ]
#
# LOCAL_DEPENDENCIES = [
# ]

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='wf-smcmodel',
    packages=find_packages(),
    version=VERSION,
    include_package_data=True,
    description='Provide estimation and simulation capabilities for sequential Monte Carlo (SMC) models',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/wf-smcmodel',
    author='Theodore Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    install_requires=BASE_DEPENDENCIES,
    # tests_require=TEST_DEPENDENCIES,
    # extras_require = {
    #     'test': TEST_DEPENDENCIES,
    #     'local': LOCAL_DEPENDENCIES
    # },
    keywords=['Bayes', 'SMC'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
