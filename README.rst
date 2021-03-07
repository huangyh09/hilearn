|PyPI| |Docs| |Build Status|

.. |PyPI| image:: https://img.shields.io/pypi/v/hilearn.svg
    :target: https://pypi.org/project/hilearn
.. |Docs| image:: https://readthedocs.org/projects/hilearn/badge/?version=latest
   :target: https://hilearn.readthedocs.io
.. |Build Status| image:: https://travis-ci.org/huangyh09/hilearn.svg?branch=master
   :target: https://travis-ci.org/huangyh09/hilearn
   
HiLearn
=======

A small library of machine learning models and utility & plotting functions:

1. a set of utility functions, e.g., wrap function for cross-validation on 
   regression and classification models

2. a set of small models, e.g., mixture of linear regression model

3. a set of plotting functions, e.g., `corr_plot`, `ROC_curve`, `PR_curve`


How to install?
---------------

Easy install with *pip* by ``pip install hilearn`` for released version or the 
latest version on github (less stable though)

.. code-block:: bash

  pip install -U git+https://github.com/huangyh09/hilearn

If you don't have the root permission, add ``--user``.


Documentation
-------------

See the documentation_ for how to use, e.g., `cross-validation`_ and 
`plotting functions`_.

.. _documentation: https://hilearn.readthedocs.io
.. _`cross-validation`: https://hilearn.readthedocs.io/en/latest/cross_validation.html
.. _`plotting functions`: https://hilearn.readthedocs.io/en/latest/plotting.html
