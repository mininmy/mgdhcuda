from setuptools import setup

setup(
    name='gpu_polynomial_gmdh',
    version='0.1',
    description='GPU-accelerated polynomial and GMDH modeling',
    author='Mykhailo Minin',
    author_email='you@example.com',
    py_modules=[
        'gpu_polynomial_module',
        'gpu_gmdh_model',
        'cuda_poly_multiply',
    ],
    install_requires=[
        'cupy',
        'cudf',
        'dask',
        'dask_cuda',
        'scikit-learn',
        'numpy',
    ],
    python_requires='>=3.8',
)
