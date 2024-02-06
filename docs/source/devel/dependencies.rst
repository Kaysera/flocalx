.. _dependencies:

============
Dependencies
============

Runtime dependencies
====================

Mandatory dependencies
----------------------

When installing through a package manager like ``pip`` or ``conda``, the
mandatory dependencies are automatically installed. This list is mainly for
reference.

numpy>=1.19.3
scikit-learn


* `Python <https://www.python.org/downloads/>`_ (>= 3.9)
* `NumPy <https://numpy.org>`_ (>= 1.19)
* `scikit-learn <https://scikit-learn.org/stable/>`_


.. _test-dependencies:

Additional dependencies for testing
===================================
This section lists the additional software required for
:ref:`running the tests <testing>`.

Required:

- pytest_ (>=3.6)

Optional:

- pytest-cov_ (>=2.3.1) to collect coverage information
- pytest-flake8_ to test coding standards using flake8_
- pytest-xdist_ to run tests in parallel

.. _pytest: http://doc.pytest.org/en/latest/
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest/
.. _pytest-flake8: https://pypi.org/project/pytest-flake8/
.. _pytest-xdist: https://pypi.org/project/pytest-xdist/
.. _flake8: https://pypi.org/project/flake8/


.. _doc-dependencies:

Additional dependencies for building documentation
==================================================

Python packages
---------------
The additional Python packages required to build the
:ref:`documentation <documenting-flocalx>` are listed in
:file:`/docs/requirements.txt` and can be installed using ::

    pip install -r docs/requirements.txt

The content of :file:`requirements.txt` is also shown below:

   .. include:: ../../requirements.txt
      :literal: