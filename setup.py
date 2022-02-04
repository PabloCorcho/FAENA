import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FAENA-PabloCorcho",
    version="0.0.1",
    author="PabloCorcho",
    author_email="corchocaballeropablo@gmail.com",
    description="integral Field Analysis for Extended gAlaxies (FAENA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/PabloCorcho/FAENA",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    include_package_data=True,
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
