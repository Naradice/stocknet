from setuptools import setup, find_packages

install_requires = [
]

setup(name='trade_storategy',
      version='0.0.1',
      packages=find_packages(),
      data_files=['./trade_storategy/settings.json'],
      install_requires=install_requires,
      include_package_data=True
)