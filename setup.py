""" setup script """
import setuptools

exec(open("xwmt/version.py").read())

setuptools.setup(version=__version__)
