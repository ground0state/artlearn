from setuptools import find_packages, setup


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name='artlearn',
    license='MIT',
    description='Adoptive Resonance Theory in Python with scikit-learn like API.',
    author='Masafumi Abeta',
    author_email='ground0state@gmail.com',
    url='https://github.com/ground0state/artlearn',
    packages=find_packages('src', exclude=['demo', 'test']),
    package_dir={'': 'src'},
    install_requires=_requires_from_file('requirements.txt'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3'
    ],

)
