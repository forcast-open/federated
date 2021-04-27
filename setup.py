import setuptools

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ffl",                    
    version="0.3.4",                      
    author="Forcast OpenSource",                    
    description="Forcast Federated Learning",
    long_description=long_description,   
    long_description_content_type="text/markdown",
    packages=['forcast_federated_learning'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                     
    url = "https://github.com/forcast-open",
    python_requires='>=3.6',              
    package_dir={'forcast_federated_learning':'forcast_federated_learning'},    
    install_requires= parse_requirements('requirements.txt')                   
    
)