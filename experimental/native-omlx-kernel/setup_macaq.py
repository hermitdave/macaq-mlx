"""Build only the experimental MACaQ native kernel, in place.

Run with the oMLX bundled interpreter and its site-packages on PYTHONPATH:
  APP=/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3
  SITE=/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages
  PYTHONPATH=. :$SITE $APP setup_macaq.py build_ext --inplace
"""

from setuptools import setup

from mlx import extension

ROOT = "omlx/custom_kernels/macaq"

setup(
    name="omlx-macaq-native",
    version="0.1.0",
    ext_modules=[
        extension.MetalExtension(
            "omlx.custom_kernels.macaq._ext",
            sources=[f"{ROOT}/csrc/bindings.cpp", f"{ROOT}/csrc/macaq.cpp"],
            include_dirs=[f"{ROOT}/csrc", "omlx/custom_kernels/include"],
            metal_libraries=[
                extension.MetalLibrary(
                    "omlx_macaq_kernels",
                    sources=[f"{ROOT}/csrc/macaq.metal"],
                )
            ],
        )
    ],
    cmdclass={"build_ext": extension.BuildExtension.with_options(generate_stubs=False)},
)
