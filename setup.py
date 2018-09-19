from setuptools import setup

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(
    name='pystacknet',
    version='0.0.1',

    author='Marios Michailidis',
    author_email='kazanovassoftware@gmail.com',

    packages=['pystacknet',
              'pystacknet.test'],
              
    url='https://github.com/h2oai/pystacknet',
    license='LICENSE.txt',

    description='StackNet framework for python',
    long_description=read_md('README.md'),

    install_requires=[
        'numpy >= 1.14.0',
        'scipy >= 1.1.0',
        'scikit-learn >= 0.19.1'
    ],

    classifiers=[


        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',  
        'Programming Language :: Python :: 3.6'         
    ],
)