from setuptools import find_packages, setup

setup(
    name='titanic',
    # packages=find_packages(),
	packages=['titanic'],
    # package_dir={'': 'src'},
	package_dir={'': 'src'},
    version='0.0.0',
    description='Demo data science project using Titanic dataset from Kaggle.',
    author='Alvaro Mendoza',
    license='MIT',
	entry_points={
            'console_scripts': ['titanic=titanic.cli:cli']
    }
)
