#!/usr/bin/env python
"""
Created : 25-03-2019
Last Modified : Mon 25 Mar 2019 08:35:20 PM EDT
Created By : Enrique D. Angola
"""

import pandas as pd
import numpy as np

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

        data = pd.read_excel(self.filename,sheet_name=sheetname)
        self.data = data.drop_duplicates('EmployeeID',keep='last')
        self.null = self._get_null()
        self.empty = self._get_empty()

    def _get_null(self):

        return np.where(pd.isnull(self.data))

    def _get_empty(self):

        return np.where(self.data.applymap(lambda x: x == ''))

    def _drop_duplicates(data):

        self.data = data.drop_duplicates('EmployeeID',keep='last')

    def bin_data(self,binBy=None,groups=None):
        key = 'binned_'+binBy
        self.data[key] = pd.cut(self.data[binBy],groups)

