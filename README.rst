============================================
Project Pytanic: data science pipeline as Python package
============================================
Jupyter is great for data science work as long as workload is small enough to fit into one notebook. And what if it isn't? That's when all sorts of problems begin to surface. Notebooks are difficult to test, difficult to debug, and are a source of unceasing trouble in version control. There is a strong case to be made that the natural solution to this is to relegate notebooks to what they do best – to data exploration and data presentation, and do all the heavy lifting in .py files organized in a fully-fledged Python package.

The present project is a case study of this approach. It is intended as a testbed and a structural basis for future projects. In its core it has an automated pipeline which downloads data from Kaggle's ``Titanic`` competition, performs exploratory analysis, trains machine learning models, and generates submission. The pipeline is built as a Python package with practically all the necessary attributes.

Design goals
------------
When I started working on this project, I compiled the following wish-list of features I wanted it to have:

  **Structure**
  
  - The main pipeline must consist of Python scripts in .py files.
  - Jupyter notebooks should be used only for peripheral stages like exploratory analysis or presentation of results.
  - Standardized folder structure reusable for other projects.
  - Code potentially reusable for other projects must be grouped into importable modules.
  - Scripts containing project-specific specialized code must also be importable for testing.
  - To facilitate unit testing, code should mostly consist of parametrized functions.
  
  **Programmability**
  
  - Installable as a Python package together with all its dependencies using ``pip`` or ``conda``.  
  - Source code must be fully version-controlled with ``Git``.
  - Automated execution of the entire pipeline (including execution of Jupyter notebooks) powered by a workflow management system.
  - Workflow management system should be able to generate DAG of the pipeline as image.
  - Working command line interface.
  - Clear separation of data manipulation scripts, CLI, and WMS. No unnecessary mixing. No two-way dependencies.
  - Maximal possible coverage with unit tests.
  - Raw data files must be downloadable, immutable, and excluded from version control.
  - All interim data files must be automatically generated based on raw files.
  - Training machine learning models should write all relevant parameters to log files.
  - Writing log files must be done using ``logging`` module from the standard library.
  - Centralized, single source version numbering scheme, preferably compliant with ``PEP 440`` and SemVer.
  - Compliance with ``PEP 8``, ``PEP 257``, and ``numpy`` docstring style.
  - Docs should be written in reStructuredText.
  
  **Analysis**
  
  - Reproducibility. All who install the package must be able to obtain the same results.
  - Prevent data leakage. Push all leaking preprocessing transformations to cross-validation stage.
  - Make use of custom transformers and estimators compatible with ``scikit-learn`` API.

As of the current version, I am reasonably satisfied with the project as most of the design goals have been achieved.

Tech choices
------------
**Dependencies management**

My attempts to install dependencies in a fresh virtual environment automatically using ``pip`` and ``requirements.txt`` failed miserably. Apparently, ``scipy`` stack still has considerable problems with installation on Windows. In the end I solved the problem using ``conda`` and ``environment.yml``.

**Workflow automation**

For a small pipeline like this, it would have been perfectly possible to chain separate scripts together using a master script. This approach I, however, discarded from the very outset as difficult to scale for larger projects (and not particularly fun either). As far as dedicated workflow automation tools go, there is a surprisingly large amount of options starting from the tried and true workhorse ``GNU Make`` and ending with full-blown frameworks like ``Luigi`` or ``Apache Airflow``.

I have seen a lot of projects that used ``Make`` as a tool of choice, but the syntax of its DSL looks like Sumerian cuneiform to me. ``Luigi``, on the other hand, is a pure Python package. However, it struck me as too verbose for the task at hand.

After some deliberation, for this project I selected ``Snakemake``, a ``Make``-inspired workflow management system popular in bioinformatics community. What enticed me the most, was the simple, human readable syntax of its language, which is based on Python and can be seamlessly combined with ordinary Python code. For all that, ``Snakemake`` has its downsides. Some of its dependencies are tricky to install, and the package itself is barely compatible with the most popular operating system in the world. A number of ``Snakemake``'s features are either untested on Windows or does not work at all. Eventually I managed to make it do what it was supposed to do, but it took me a lot of trial and error.

**Command line interface**

``argparse`` from the standard library is too clunky for my taste, and ``docopt`` is said to be tricky to get to work correctly. This left me with ``click``, which indeed proved to be a good tool.

**Testing framework**

I do like ``pytest``. Developers of ``pandas`` switched to ``pytest`` as their testing framework not without good reason. It is clean, easy to use, scalable, and does not require subclassing from TestCase, remembering numerous assertSomething functions or using tedious setup/teardown procedures.

**Version numbering**

From the plethora of existing options, I chose ``setuptools_scm``. Upon installation of setup.py it reads version number of the package directly from the last ``Git`` tag.

Structure
---------
The functional structure of the project directory looks like this::

  titanic               <- Project root directory.
  |
  +-- data              <- Datasets consumed and generated by the main pipeline.
  |   |
  |   +-- external      <- Data from third party sources.
  |   +-- interim       <- Intermediate data that has been transformed.
  |   +-- processed     <- Final, canonical datasets for modelling.
  |   +-- raw           <- Original immutable data.
  +-- logs              <- Log files generated by the package.
  +-- misc              <- Everything that does not belong in other directories.
  +-- models            <- Serialized machine learning models.
  +-- notebooks         <- Jupyter notebooks as .ipynb and .py files.
  +-- references        <- Data dictionaries, manuals, and other explanatory materials.
  +-- reports           <- Generated analysis as HTML, PDF, LaTeX, etc.
  +-- results           <- Generated graphics and figures to be used in reporting.
  +-- src               <- Source code of the package.
  |   | 
  |   +--titanic        <- This directory is required for the package to be imported as 'titanic' and not 'src'.
  |      |                 In its root it contains modules with reusable code.
  |      +-- srcipts    <- Project-specific data manipulation scripts.
  +-- tests             <- Unit tests.
  |   |
  |   +-- data          <- Mock datasets used in unit testing.
  +-- .gitignore        <- List of files and directories excluded from version control.
  +-- environment.yml   <- List of dependencies and their versions for installation with conda.
  +-- LICENCE           <- Declaration of licence.
  +-- README.rst        <- Top-level public README.
  +-- setup.py          <- Package metadata for installation.
  +-- Snakefile         <- File with build directives used by Snakemake.

The order of tasks that constitute the pipeline of this project is illustrated below:

.. image:: https://github.com/alvaromendoza/pytanic/blob/develop/misc/images/dag.svg

Logically, of course, feature engineering depends on exploratory analysis, and making submission depends on model comparison. In a technical sense, however, the two tasks implemented in Jupyter notebooks have no bearing on the main branch of the pipeline.

Installation
------------
This package has been created and testes using Anaconda 5.2.0 distribution of Python 3.6.5.

In order to create a new virtual environment with all project dependencies, ``cd`` to project root and issue the following command:

.. code-block:: shell-session

   conda env create -f environment.yml

Install the souce code from the project root using ``pip``:

.. code-block:: shell-session

   pip install .

If you want changes in the source code to be directly reflected in the installed version in real time, install the package in development mode:

.. code-block:: shell-session

   pip install -e .

Uninstalling the package is as simple as:

.. code-block:: shell-session

   pip uninstall titanic

Usage
-----
Firstly, I have to emphasize that all commands listed in this section must be issued from the project root, and failure to do so may lead to undesirable results. To my great disappointment, I have not yet found a way to make installed code remember the path to directory from which it was installed save for hardcoding it, which I do not want to do.

**Scripts**

It is possible to run the tasks of the pipeline using their corresponding scripts directly. For example:

.. code-block:: shell-session

   python src/titanic/scripts/download_data.py

Here I should add that downloading competition datasets from Kaggle programmatically requires making use of official Kaggle API which in turn requires authentication. Detailed instructions on how to do it can be found here_.

.. _here: https://github.com/Kaggle/kaggle-api

**Command line interface**

Commands of the CLI are essentially shortcuts to the scripts, some of which are parametrized. Command ``titanic`` is the entry point to CLI. All other commands are its subcommands and can be issued using ``titanic <command>`` format.

Full list of commands can be viewed using ``titanic --help``:

.. code-block:: shell-session

   (myenv) d:\projects\titanic>titanic --help
   Usage: titanic [OPTIONS] COMMAND [ARGS]...

     CLI entry point of project Titanic.

   Options:
     --help  Show this message and exit.

   Commands:
     clean       Delete automatically generated files.
     compmod     Run Jupyter notebook with model comparison.
     crossval    Cross-validate machine learning models.
     download    Download Titanic competition data files from...
     eda         Run Jupyter notebook with exploratory data...
     features    Perform feature engineering on training and...
     submission  Make prediction on test set and create...

The ``--help`` option can also be used with individual commands:

.. code-block:: shell-session

   (myenv) d:\projects\titanic>titanic clean --help
   Usage: titanic clean [OPTIONS]
   
     Delete automatically generated files.
   
   Options:
     -a, --allfiles  Delete all files.
     -d, --data      Delete files only in data directory.
     -l, --logs      Delete files only in logs directory.
     -m, --models    Delete files only in models directory.
     -r, --results   Delete files only in results directory.
     --help          Show this message and exit.

It should be noted that both scripts and CLI commands rely on the assumption that file dependencies of a task already exist (for instance, feature engineering requires downloaded raw data files). If this is not the case, you should use ``Snakemake`` interface instead.

**Snakemake**

The way ``Snakemake`` runs the pipeline is defined in the file named ``Snakefile`` in the project root. The file mostly consists of human-readable ``rules`` that specify input and output files for each task as well as commands to create output from input. These commands can be expressed as shell commands, references to Python scripts or raw Python code. In our particular case, rules in the ``Snakefile`` are defined using commands of the project's CLI.

``Snakemake`` detects dependencies between rules by matching file names and thus allows us to run the whole pipeline or its part based on which files already exist and which need to be created. A rule is executed not only when one or more of its direct or indirect file dependencies are missing, but also when these dependencies are newer than one or more of the output files of the rule.

In order to run the whole pipeline, simply type ``snakemake`` in the command line. Parts of the pipeline can be run using ``snakemake <rule name>`` or ``snakemake <output file name>`` format. For instance:

.. code-block:: shell-session

   snakemake submission

or

.. code-block:: shell-session

   snakemake results/submission.csv

``Snakemake`` can also generate images of direct acyclic graphs of specified rules using ``Graphviz``. For example:

.. code-block:: shell-session

   snakemake --dag submission | dot -Tsvg > dag_submission.svg

This command produces the following graph:

.. image:: https://github.com/alvaromendoza/pytanic/blob/develop/misc/images/dag_submission.svg

Rules that don't need to be run because their output files are up-to-date are shown in dashed rectangles.

See also
--------
  * `Cookiecutter Data Science: A logical, reasonably standardized, but flexible project structure for doing and sharing data science work`__.
  __ https://drivendata.github.io/cookiecutter-data-science/
  * `Prototyping to tested code, or developing across notebooks and modules`__.
  __ https://cprohm.de/article/notebooks-and-modules.html
  * `Building package for machine learning project in Python`__.
  __ https://towardsdatascience.com/building-package-for-machine-learning-project-in-python-3fc16f541693
  * `Structure and automated workflow for a machine learning project`__.
  __ https://towardsdatascience.com/structure-and-automated-workflow-for-a-machine-learning-project-2fa30d661c1e
  * `Working efficiently with JupyterLab Notebooks`__.
  __ https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/
  * `Snakemake docs`__.
  __ https://snakemake.readthedocs.io/en/stable/index.html
