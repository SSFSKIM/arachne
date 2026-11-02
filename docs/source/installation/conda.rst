Conda
=====

Required Dependencies
---------------------

There are several package managers that work with conda packages,
namely `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`__,
`mamba <https://mamba.readthedocs.io>`__ and `pixi <https://pixi.sh>`__.

Arachne is not packaged for conda. After installing your package manager, you can create a new
environment holding Python and Cairo, and install Arachne into it from its repository, by running


.. tab-set::

   .. tab-item:: conda / mamba

      .. code-block:: bash

         # if you want to use mamba, just replace conda below with mamba
         conda create -n my-arachne-environment -c conda-forge python=3.14 pip pycairo
         conda activate my-arachne-environment
         pip install git+https://github.com/SSFSKIM/arachne

   .. tab-item:: pixi

      .. code-block:: bash

         pixi init
         pixi add python=3.14 pycairo
         pixi add --pypi "arachne-engine @ git+https://github.com/SSFSKIM/arachne"



Optional Dependencies
---------------------

In order to make use of Manim's interface to LaTeX to, for example, render
equations, LaTeX has to be installed as well. Note that this is an optional
dependency: if you don't intend to use LaTeX, you don't have to install it.

Recommendations on how to install LaTeX on different operating systems
can be found :doc:`in our local installation guide </installation/uv>`.


Working with Manim
------------------

At this point, you should have a working installation of Manim, head
over to our :doc:`Quickstart Tutorial <../tutorials/quickstart>` to learn
how to make your own *Manimations*!
