import sys
from cx_Freeze import setup, Executable
import os

# Dependencies are automatically detected, but it might need fine tuning.
# "packages": ["os"] is used as example only
# build_exe_options = {"packages": ["os"], "excludes": ["tkinter"]}

# base="Win32GUI" should be used only for Windows GUI app
base = None
if sys.platform == "win32":
    base = "Win32GUI"

base_c = None

python_dir = r'C:\Users\Administrator\AppData\Local\Programs\Python\Python39'

includes = ['torch._VF', 'torchvision', 'torch.distributions.constraints', 'numpy.core._multiarray_umath',
            'torch.onnx.symbolic_opset7', 'torch.onnx.symbolic_opset8', 'torch.onnx.symbolic_opset12']
excludes = []
packages = []
includefiles = []
build_exe_options = {'build_exe': {'include_files': includefiles, 'includes': includes}}

cr = """Copyright (c) 2021 Cristhian Romero

        All rights reserved."""

exe = Executable("main.py", target_name='ObjectCam.exe', base=base, copyright=cr)
exe_1 = Executable("mac.py", target_name='KeyGen.exe', base=base_c, icon="icon_2.ico", copyright=cr)

setup(
    name='ObjectCam',
    version="0.1",
    license=cr,
    description='ObjectCam',
    options=build_exe_options,
    executables=[exe, exe_1]
)
