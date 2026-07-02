"""Minimal cv2 stub.

The vendored WireMask-BBO ``utils.py`` does ``import cv2`` but only uses
``cv2.imwrite`` inside ``write_placement_and_overlap`` (a placement-image dump
that is NOT on the EA / HPWL-evaluation path). This stub satisfies the import so
the EA can run without installing opencv. If you want the visualization, install
opencv-python and delete this file.
"""


def imwrite(*args, **kwargs):
    return False
