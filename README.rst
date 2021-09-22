|image1| |Downloads|

artlearn
========

artlearn is a set of algorithms for Adoptive resonance theory.

This contains these techniques.

-  ART1
-  ART2
-  ART2A
-  Bayesian ART
-  Fuzzy ART
-  SFAM

and my original,

-  L2ART

Dependencies
------------

The required dependencies to use artlearn are

-  scikit-learn >= 0.22.2
-  numpy >= 1.19.5
-  scipy >= 1.4.1

You also need Matplotlib >= 3.2.2 to run the demo and pytest >= 6.2.5 to
run the tests.

install
-------

.. code:: bash

   pip install artlearn

USAGE
-----

We have posted a usage example in the demo folder.

For example…

Fuzzy ART
~~~~~~~~~

.. code:: python

   from artlearn import FuzzyART


   clf = FuzzyART(max_iter=100, max_class=100, rho=0.72, alpha=1e-5, beta=0.1)
   clf.fit(X)

   labels = clf.labels_

Bayesian ART
~~~~~~~~~~~~

.. code:: python

   from artlearn import BayesianART


   clf = BayesianART(max_iter=3, max_class=100, rho=0.01, sigma=0.05, max_hyper_volume=0.07)
   clf.fit(X)

   labels = clf.labels_

SFAM
~~~~

.. code:: python

   from artlearn import SFAM


   clf = SFAM(max_iter=100, max_class=100, rho=0.9, alpha=1e-5, beta=0.1)
   clf.fit(X, y)

   y_pred = clf.predict(X)

License
-------

This code is licensed under MIT License.

Test
----

.. code:: bash

   python setup.py test

.. |image1| image:: https://img.shields.io/badge/dynamic/json.svg?label=version&colorB=5f9ea0&query=$.version&uri=https://raw.githubusercontent.com/ground0state/artlearn/master/package.json&style=plastic
.. |Downloads| image:: https://pepy.tech/badge/artlearn
   :target: https://pepy.tech/project/artlearn
