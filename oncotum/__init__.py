"""
This is the base entry point of OncoTUM. OncoSTR splits in a module 'tumor_segmentation' that holds all necessary
functions for the user, a 'utils.py' file, where sub-ordinate functions are gathered and a sub-package 'models',
where neural networks are defined.

Modules:
    tumor_segmentation:  Its the control file for all functionalities for the user.
    utils:               Herein, helper functions are hold, in order to keep the other file clean for the user.
Sup-package:
    models:              Neural networks are defined here.
"""
from .tumor_segmentation import TumorSegmentation
