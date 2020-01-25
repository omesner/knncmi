from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='knncmi',
      version='0.0.1',
      description='Estimate conditional mutual information for discrete and continuous data',
      url='http://github.com/omesner/knncmi',
      author='Octavio Mesner',
      author_email='omesner@gmail.com',
      long_description=long_description,
      long_description_content_type="text/markdown",
      license='GNU GPLv3',
      packages= find_packages(),
      python_requires='>=3.6',
      zip_safe=False)
