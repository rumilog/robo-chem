"""
RoboChem - Autonomous Robotic Chemist

A VLM-as-Orchestrator system for wet-chemistry manipulation.
"""

from setuptools import setup, find_packages

setup(
    name="robochem",
    version="0.1.0",
    description="Autonomous Robotic Chemist - VLM orchestrated chemistry manipulation",
    author="Barati Farimani Lab, CMU",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "openai>=1.0.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "vision": [
            "open3d>=0.15.0",
            "segment-anything",  # SAM
        ],
        "robot": [
            # "frankapy",  # Install separately
            # "robomail",  # Install from local
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "robochem-run=scripts.run_experiment:main",
        ],
    },
)
