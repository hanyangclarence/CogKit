from setuptools import setup, find_packages

setup(
    name='cogkit',
    # Version will be dynamically set by hatch-vcs (see pyproject.toml)
    # version='0.1.0', # Remove this line as it will be set dynamically
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='ZhipuAI',
    author_email='opensource@zhipuai.cn',
    url='https://github.com/THUDM/CogKit',
    packages=find_packages('src'), # Tell setuptools to look in the src directory.
    package_dir={'': 'src'}, # Tell setuptools that packages are under src
    include_package_data=True,
    install_requires=[
        "click~=8.1",
        "diffusers @ git+https://github.com/huggingface/diffusers.git",
        "imageio-ffmpeg~=0.6.0",
        "imageio~=2.37",
        "peft~=0.14.0",
        "pydantic~=2.10",
        "sentencepiece==0.2.0",
        "transformers~=4.49",
        "wandb~=0.19.8",
        "fastapi[standard]~=0.115.11",
        "fastapi_cli~=0.0.7",
        "openai~=1.67",
        "pydantic_settings~=2.8.1",
        "python-dotenv~=1.0",
    ],
    extras_require={
        'finetune': [
            "datasets~=3.4",
            "deepspeed~=0.16.4",
            "av~=14.2.0",
        ],
        'dev': [
            "mypy~=1.15",
            "ruff~=0.11.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'cogkit = cogkit.cli:cli',
        ],
    },
    python_requires='>=3.10',
    license_files = ["LICENSE"],
    keywords = [],
)