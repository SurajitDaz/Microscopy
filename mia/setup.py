from setuptools import setup, find_packages

setup(
    name='mia',
    description='A package for Microscopy Image Analysis Toolkit.',
    author='Surajit Das',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements.txt
    entry_points={
        'console_scripts': [
            'mia=mia.__main__:main',
        ],
    },
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.7',
)
