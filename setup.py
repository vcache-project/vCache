from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "numpy>=1.26.4",
    "langchain>=0.3.23",
    "langchain-community>=0.3.21",
    "langchain-openai>=0.3.14",
    "langchain-anthropic>=0.3.12",
    "langchain-huggingface>=0.1.2",
    "langchain-google-genai>=2.1.3",
    "torch>=2.6.0",
    "openai>=1.75.0",
    "faiss-cpu>=1.10.0",
    "hnswlib>=0.8.0",
    "chromadb>=1.0.5",
    "sentence-transformers>=4.1.0",
    "accelerate>=1.6.0",
    "typing-extensions>=4.13.2",
    "torchvision>=0.22.0",
    "statsmodels>=0.14.4",
    "ijson>=3.4.0",
    "pytest>=8.3.5",
    "datasets>=3.6.0",
    "pandas>=2.3.0",
]

setup(
    name="vcache",
    version="0.1.0",
    author="Luis Gaspar Schroeder, Aditya Desai, Alejandro Cuadron, Kyle Chu, Shu Liu, Mark Zhao, Stephan Krusche, Alfons Kemper, Matei Zaharia, Joseph E. Gonzalez",
    author_email="luis.gaspar.schroeder@gmail.com",
    description="Reliable and Efficient Semantic Prompt Caching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vcache-project/vCache",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.10,<4.0",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
