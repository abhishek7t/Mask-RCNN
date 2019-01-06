import os

import pdb


import shutil


# def safe_mkdir(path):
#         """ Create a directory if there isn't one already. """
#         try:
#             os.mkdir(path)
#         except OSError:
#             pass

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        shutil.rmtree(path)
        safe_mkdir(path)
        pass