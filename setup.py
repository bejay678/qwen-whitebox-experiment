from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qwen-whitebox",
    version="1.0.0",
    author="bejay678",
    author_email="533220@qq.com",
    description="White-boxing memory modules of Qwen2.5-0.5B-Instruct",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bejay678/qwen-whitebox-experiment",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
            "torch>=2.0.0",
        ],
        "cpu": [
            "faiss-cpu>=1.7.0",
            "torch>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "qwen-whitebox-train=scripts.train_adapter:main",
            "qwen-whitebox-build-index=scripts.build_faiss:main",
            "qwen-whitebox-demo=scripts.editable_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "adapter": ["c_implementation/*.c", "c_implementation/*.so", "*.pt"],
    },
)