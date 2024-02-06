.. redirect-from:: /users/installing

############
Installation
############

==============================
Installing an official release
==============================

FLocalX releases are available as wheel packages for macOS, Windows and
Linux on PyPI. Install it using
``pip``:

.. code-block:: sh

  python -m pip install -U pip
  python -m pip install -U flocalx


=========================
Third-party distributions
=========================

Various third-parties provide FLocalX for their environments.

Conda packages
==============

Not yet available

.. _install_from_source:

======================
Installing from source
======================

If you are interested in contributing to FLocalX development,
running the latest source code, or just like to build everything
yourself, it is not difficult to build FLocalX from source.

First you need to install the :ref:`dependencies`.

The easiest way to get the latest development version to start contributing
is to go to the git `repository <https://github.com/Kaysera/flocalx>`_
and run::

  git clone https://github.com/Kaysera/flocalx.git

or::

  git clone git@github.com:Kaysera/flocalx.git

If you're developing, it's better to do it in editable mode. The reason why
is that pytest's test discovery only works for FLocalX
if installation is done this way. Also, editable mode allows your code changes
to be instantly propagated to your library code without reinstalling (though
you will have to restart your python process / kernel)::

  cd flocalx
  python -m pip install -e .

If you're not developing, it can be installed from the source directory with
a simple (just replace the last step)::

  python -m pip install .

Then, if you want to update your FLocalX at any time, just do::

  git pull