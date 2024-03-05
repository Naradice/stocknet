from setuptools import setup, find_packages

install_requires = ["torch", "gym==0.22", "numpy", "pfrl", "pandas", "requests"]

setup(name="stocknet", version="0.0.1", packages=find_packages(), install_requires=install_requires)
