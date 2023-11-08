"""Python setup.py for plotsandgraphs package"""
import io
import os
from setuptools import setup, find_packages

# setup(
#     name='plotsandgraphs',
#     version='0.1.0',
#     packages=find_packages(include=['plotsandgraphs', 'plotsandgraphs.*'])
# )

PROJECT_NAME = 'plotsandgraphs'




def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name=PROJECT_NAME,
    version=read(PROJECT_NAME, "VERSION"),
    description="Create plots and graphs for your Machine Learning projects.",
    url="https://github.com/joshuawe/plots_and_graphs",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Joshua Wendland and Fabian Krüger",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["project_name = project_name.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
    license='GNU General Public License v3.0',
    keywords=['plots', 'graphs', 'machine learning', 'data science', 'data visualization', 'data analysis', 'matplotlib'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)