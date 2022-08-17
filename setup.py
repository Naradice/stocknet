from setuptools import setup, find_packages

install_requires = [
    'torch',
    'gym==0.22',
    'numpy',
    'pfrl',
    'pandas',
    'requests'#vantage, es
]

setup(name='stocknet',
      version='0.0.1',
      packages=find_packages(),
      install_requires=install_requires  # And any other dependencies foo needs
)