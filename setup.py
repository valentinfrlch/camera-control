import os
import site
import sys
import logging

sys.setrecursionlimit(10000)

from setuptools import setup

site_packages = site.getsitepackages()[0]
torch_lib = os.path.join(site_packages, "torch", "lib")
torchvision_lib = os.path.join(site_packages, "torchvision", "lib")

def existing(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        logging.warning("Skipping missing frameworks: %s", missing)
    return [p for p in paths if os.path.exists(p)]

frameworks = existing(
    [
        os.path.join(torch_lib, "libtorch.dylib"),
        os.path.join(torch_lib, "libtorch_cpu.dylib"),
        os.path.join(torch_lib, "libc10.dylib"),
        os.path.join(torchvision_lib, "libtorchvision.dylib"),
    ]
)

APP = ["webserver.py"]
DATA_FILES = ["templates", "static", "yolov8n.pt", "sitecustomize.py"]
OPTIONS = {
    "iconfile": "assets/appIcon.icns",
    "packages": ["flask", "cv2", "rawpy", "torch", "torchvision", "ultralytics"],
    "includes": ["torchvision._C", "torchvision._meta_registrations"],
    "excludes": ["PyInstaller"],
    "argv_emulation": False,
    "plist": {"NSHighResolutionCapable": True},
    "strip": False,
    "frameworks": frameworks,
}

setup(app=APP, data_files=DATA_FILES, options={"py2app": OPTIONS})
