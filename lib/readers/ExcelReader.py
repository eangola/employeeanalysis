#!/usr/bin/env python
"""
Created : 25-03-2019
Last Modified : Mon 25 Mar 2019 06:41:25 PM EDT
Created By : Enrique D. Angola
"""

import pandas as pd

class ExcelReader():
    """
    Reads excel data file

    Parameters
    ----------


    Returns
    -------


    Examples
    --------
    >>>

    """

    def __init__(self,filename):
        self.filename = filename
        self.empty = None
        self.null = None
        self.data = None

    def get_data(self,sheetname):

        self.data = pd.read_excel(self.filename,sheet_name=sheetname)
        
        self.null = self._get_null()
        self.empty = self._get_empty()

    def _get_null(self):

        return np.where(pd.isnull(self.data))

    def _get_empty(self):

        return np.where(self.data.applymap(lambda x: x == ''))
