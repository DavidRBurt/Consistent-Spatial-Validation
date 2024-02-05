from setuptools import find_packages, setup

setup(
    name="spatial_validation",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow",
        "tensorflow-probability",
        "gpflow",
        "scikit-learn",
        "matplotlib",
        "json-tricks",
        "xarray",
        "requests",
        "joblib",
        "h5netcdf",
        "netcdf4",
        "reverse_geocoder",
        "plotly",
        "kaledio",
    ],
)
