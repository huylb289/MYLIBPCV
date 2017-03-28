#!/usr/bin/env python

from distutils.core import setup

setup(name='MYLIBPCV',
        version='0.1',
        author='Huy LB',
        url='https://github.com/jesolem/PCV',
        packages=['PCV', 'PCV.classifiers', 'PCV.clustering', 'PCV.geometry', 
                'PCV.imagesearch', 'PCV.localdescriptors', 'PCV.tools'],
        requires=['NumPy', 'Matplotlib', 'SciPy'],
        )
