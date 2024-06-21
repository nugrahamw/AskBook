from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME ="AskBooks"
AUTHOR_USER_NAME = "NugrahaMarga"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ['streamlit', 'numpy']

setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small package for Books Recommendation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NugrahaMarga/AskBooks",
    author_email="m227d4ky1778@bangkit.academy",
    packages=[SRC_REPO],
    python_requires=">=3.8",
    install_requires=LIST_OF_REQUIREMENTS
)