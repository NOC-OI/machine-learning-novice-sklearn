---
title: Setup
---
# Requirements

## Software

You will need a terminal, Python 3.8+, and the ability to create Python virtual environments.

## Packages

You will need the MatPlotLib, Pandas, Numpy and OpenCV packages. 

# Setup

Create a new directory for the workshop, then launch a terminal in it:

~~~
mkdir workshop-ml
cd workshop-ml
~~~
{: .language-bash}

## Creating a new Virtual Environment
We'll install the prerequisites in a virtual environment, to prevent them from cluttering up your Python environment and causing conflicts.
First, create a new directory and ent

To create a new virtual environment for the project, open the terminal and type:

~~~
python3 -m venv venv
~~~
{: .language-bash}

> If you're on Linux and this doesn't work, try installing `python3-venv` using your package manager, e.g. `sudo apt-get install python3-venv`.
{: .info}

## Installing your prerequisites

Activate your virtual environment, and install the prerequisites:

~~~
source venv/bin/activate
pip install numpy pandas matplotlib opencv-python
~~~
{: .language-bash}

You'll need to re-activate the virtual environment to use it during the session.

{% include links.md %}
