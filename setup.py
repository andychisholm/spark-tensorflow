from setuptools import setup, find_packages

__version__ = '0.1.0'
__pkg_name__ = 'spark-tensorflow'

setup(
    name = __pkg_name__,
    version = __version__,
    description = 'Toolkit for running distributed tensorflow models on Spark',
    author='Andrew Chisholm',
    packages = find_packages(),
    license = 'MIT',
    url = 'https://github.com/wikilinks/spark-tensorflow',
    scripts = [
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires = [
        "numpy"
    ],
    test_suite = __pkg_name__ + '.test'
)
