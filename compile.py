from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("hnefatafl",  ["hnefatafl.py"]),
    Extension("hnefatafl_train",  ["hnefatafl_train.py"])
]
setup(
    name = 'Hnefatafl DRL',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
