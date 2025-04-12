from setuptools import setup, find_packages

setup(
    name="vectorq",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "langchain>=0.3.23,<1.0.0",
        "langchain-community==0.3.21",
        "typing-extensions==4.12.2",
        "vllm==0.7.3",
        "torch==2.5.1",
        "openai>=1.70.0",
        "faiss-cpu==1.10.0",
        "hnswlib==0.8.0",
        "chromadb==1.0.0",
        "sentence-transformers==4.0.2",
        "accelerate==1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.4",
            "black==23.12.1",
            "isort==5.13.2",
        ],
        "benchmarks": [
            "matplotlib>=3.7.0",
            "ijson>=3.2.0",
            "numpy>=1.26.4",
            "seaborn>=0.12.0",
            "scipy>=1.10.0",
            "adjustText>=0.8",
            "scikit-learn>=1.2.0",
        ],
    },
    python_requires=">=3.10",
)
