#!/usr/bin/env python
"""
Created : 25-03-2019
Last Modified : Thu 28 Mar 2019 02:56:29 PM EDT
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

    def get_data(self,sheetname,primaryKey = 'EmployeeID'):

        data = pd.read_excel(self.filename,sheet_name=sheetname)
        self.data = data.drop_duplicates(primaryKey,keep='last')
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

    def merge_data(self,dataframe=None,on=None,originalkey = None):
        
        if originalkey:
            self.data = self.data.rename(index=str,columns={originalkey:on})

        self.data = self.data.merge(dataframe,on=on)
        self.null = self._get_null()
        self.empty = self._get_empty()


