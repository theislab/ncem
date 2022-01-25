"""
Paths to cache directories used throughout the tests.
"""

import os

DIR_TEMP = os.path.join(os.path.dirname(__file__), "temp")
DIR_TEMP_DATA = os.path.join(DIR_TEMP, "data")

DATA_PATH_ZHANG = os.path.join(DIR_TEMP_DATA, "zhang")
DATA_PATH_JAROSCH = os.path.join(DIR_TEMP_DATA, "busch")
DATA_PATH_HARTMANN = os.path.join(DIR_TEMP_DATA, "hartmann")
DATA_PATH_SCHUERCH = os.path.join(DIR_TEMP_DATA, "schuerch")
DATA_PATH_LU = os.path.join(DIR_TEMP_DATA, "lu")
DATA_PATH_DESTVI = os.path.join(DIR_TEMP_DATA, "destvi_lymphnode/")
