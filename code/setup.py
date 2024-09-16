from setuptools import find_packages, setup

setup(
    name="spatial_validation",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "gpflow==2.9.0 ",
        "h5netcdf==1.2.0",
        "joblib==1.3.2",
        "json-tricks==3.17.3",
        "kaleido==0.2.1",
        "lightgbm==4.3.0",
        "matplotlib==3.8.1",
        "netcdf4==1.6.2",
        "numpy==1.23.5",
        "pandas==2.1.3",
        "plotly==5.18.0",
        "requests==2.31.0",
        "reverse_geocoder==1.5.1",
        "scikit-learn==1.3.2",
        "scipy==1.11.3",
        "tensorflow==2.15.0",
        "tensorflow-probability==0.23.0",
        "xarray==2023.11.0",
    ],
)
