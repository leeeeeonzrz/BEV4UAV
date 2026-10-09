# Copyright (c) OpenMMLab. All rights reserved.
# Modified for the BEV4UAV source release: packaging metadata and dependencies.
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
version = {}
exec((ROOT / "mmdet3d/version.py").read_text(encoding="utf-8"), version)


def requirements(filename):
    result = []
    for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            result.extend(requirements(line[3:].strip()))
        elif line:
            result.append(line)
    return result


setup(
    # Keep the distribution/import identity expected by OpenMMLab integrations.
    name="mmdet3d",
    version=version["__version__"] + "+bev4uav",
    description="BEV4UAV: UAV LiDAR-camera 3D detection with modified MMDetection3D",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="BEV4UAV and MMDetection3D contributors",
    url="https://github.com/leeeeeonzrz/BEV4UAV",
    license="Apache-2.0",
    python_requires=">=3.8",
    packages=find_packages(include=["mmdet3d", "mmdet3d.*", "projects", "projects.*"]),
    include_package_data=True,
    install_requires=requirements("requirements/runtime.txt"),
    extras_require={
        "mim": requirements("requirements/mminstall.txt"),
        "tests": requirements("requirements/tests.txt"),
        "optional": requirements("requirements/optional.txt"),
    },
    zip_safe=False,
)
