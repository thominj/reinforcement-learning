import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reinforcement_learning",
    version="0.0.1",
    author="James Thomin",
    author_email="james.thomin@gmail.com",
    description="A framework for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thominj/reinforcement-learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pytest',
    ],
    tests_require=[
        'pytest',
    ],
)