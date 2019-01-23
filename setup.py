from setuptools import setup

setup(
    name='titanic',
	packages=['titanic'],
	package_dir={'': 'src'},
	use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='Demo data science project using Titanic dataset from Kaggle.',
    author='Alvaro Mendoza',
    license='MIT',
	entry_points={
            'console_scripts': ['titanic=titanic.cli:cli']
    }
)
