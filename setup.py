#!/usr/bin/env python3
"""
Multi-Camera Anomaly Detection System
====================================

A professional-grade AI system for real-time anomaly detection
in multi-camera surveillance environments.

Features:
- Real-time anomaly detection with YOLO + Deep Learning
- Multi-camera fusion with intelligent scoring
- Mobile and web interfaces
- Continuous learning from user feedback
- 95%+ accuracy target with sub-second inference

Author: Pro AI Engineer
Version: 1.0.0
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multi-camera-anomaly-detection",
    version="1.0.0",
    author="Pro AI Engineer",
    author_email="engineer@anomalydetection.ai",
    description="Professional multi-camera anomaly detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pubu99/Anomaly-Detection-CCTV-FYP---M6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "tensorrt": ["tensorrt>=8.0.0"],
        "triton": ["tritonclient[all]>=2.34.0"],
    },
    entry_points={
        "console_scripts": [
            "anomaly-train=training.train:main",
            "anomaly-inference=inference.real_time_inference:main",
            "anomaly-api=backend.main:start_server",
        ],
    },
)