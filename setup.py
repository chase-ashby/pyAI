from setuptools import setup, find_packages

# Read requirements 
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

print (requirements)

setup(
    name="pyAI",
    version="0.1",
    description="Python AI/ML Package",
    packages=find_packages(),
    author="Chase Ashby",
    install_requires=requirements,
)