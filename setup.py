from setuptools import setup

setup(
    name='jvsfunc',
    version='1.0.2',
    description='Julek VapourSynth functions',
    url='https://github.com/dnjulek/jvsfunc',
    license='MIT',
    author='Julek',
    packages=['jvsfunc'],
    package_data={
        'jvsfunc': ['py.typed'],
    },
    install_requires=[
        'vapoursynth>=58',
        'lvsfunc',
        'vsutil',
    ],
    zip_safe=False,
    python_requires='>=3.8',
)
