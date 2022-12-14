from setuptools import setup

with open("README.md") as fh:
    long_desc = fh.read()

setup(
    name='jvsfunc',
    version='1.0.16',
    description='Julek VapourSynth functions',
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url='https://github.com/dnjulek/jvsfunc',
    license='MIT',
    author='Julek',
    packages=['jvsfunc'],
    package_data={
        'jvsfunc': ['py.typed'],
    },
    install_requires=[
        'vapoursynth>=58',
        'vsrgtools>=0.1.4',
        'vsutil>=0.8.0',
    ],
    zip_safe=False,
    python_requires='>=3.8',
)
