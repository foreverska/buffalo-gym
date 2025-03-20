from setuptools import setup

long_description = open('README.md').read()

setup(
    name="buffalo_gym",
    description="Buffalo Gym environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.3.1",
    author="foreverska",
    install_requires=["gymnasium>=0.26.0", "numpy"],
    keywords="gymnasium, gym",
    license_files = ('license.txt',),
    project_urls={"Github:": "https://github.com/foreverska/buffalo-gym"}
)
