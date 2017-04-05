#!/usr/bin/env python

from distutils.core import setup

setup(name='MYLIBPCV',
        version='0.3',
        author='Huy LB',
        url='https://github.com/jesolem/PCV',
        packages=['MYLIBPCV', 'MYLIBPCV.tools'],
        requires=['NumPy', 'Matplotlib', 'SciPy'],
        )
