# csth-imaging
Analysis scripts for CS/TH particle counting and colocalization

## Dependencies
Dependencies are listed in /dependencies/requirements.txt. Additional dependencies:
- python 3
- tifffile.py and czifile.py: these scripts from Christopher Gohlke at UCI provide the classes and methods for reading .czi-format imaging files. They are provided in /dependencies.
- javabridge: The javabridge version provided through PyPI/pip DO NOT WORK for python 3. This script uses the development version provided at https://github.com/LeeKamentsky/python-javabridge, which has a bugfix to take care of this issue.
- [pyto_segmenter](https://github.com/nrweir/pyto_segmenter)
