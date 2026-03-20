from setuptools import setup, find_packages

setup(
    name="cuezero",
    version="0.1.0",
    description="Reinforcement learning system for billiards using neural networks and MCTS",
    author="",
    author_email="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.0",
        "numpy",
        "pyyaml"
    ],
    entry_points={
        "console_scripts": [
            "cuezero-train=scripts.train:main",
            "cuezero-selfplay=scripts.selfplay:main",
            "cuezero-evaluate=scripts.evaluate:main"
        ]
    }
)