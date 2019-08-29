import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="acse_9_irp_wafflescore",
    version="1.0.4",
    author="Nitchakul Pipitvej",
    author_email="np2618@ic.ac.uk",
    description="ML for Automatic Facies Classification ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msc-acse/acse-9-independent-research-project-wafflescore",
    py_modules = ['SOMsHelpers', 'MiscHelpers', 'minisom', 'fuzzy_clustering', 'FCMHelpers', 'dataPreprocessing', 'HDBScanHelpers'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
