'''
Строим векторную модель, используем tf-idf в качестве веса для каждого терма.
'''
import json
import re
from math import log10
from collections import UserList, Counter
import numpy as np
import csv


class GitList(UserList):
    """List for GitHub Issues"""

    def __init__(self, initlist=None):
        super().__init__(initlist)
        if initlist:
            self.__initialazer()

    def __initialazer(self):
        '''Find code snippent inside string of GitHub Markdown'''
        cond1 = '```.*?\r?\n(.+?)\r?\n```'
        # return the caret (possible), new line — new line sequence in GitHub markdown.
        # Pattern: 4 whitespaces or tab, after that, any sets of any symbols, including none,
        # except one before the newline character, plus this symbol also.
        # Repeat the Pattern as many times as possible.
        cond2 = '(?:(?: {4}|\t).*(?:\r?\n)*)+'
        pattern_code_1 = re.compile(cond1, flags=re.DOTALL)
        pattern_code_2 = re.compile(cond2)

        for issue in self.data:
            issue['doc'] = issue['body']

            issue['code'] = []
            # for p in [pattern_code_1]:
            for p in (pattern_code_1, pattern_code_2):
                issue['code'].extend(p.findall(issue['doc']))
                issue['doc'] = p.sub('', issue['doc'])

    def getbynumber(self, n):
        '''Return the issue (list element) with given key "number" value.'''
        for i, issue in enumerate(self.data):
            if issue['number'] == n:
                return self.data[i]
        raise ValueError('No such issue with number {}'.format(n))

    def __getitem__(self, i):
        '''
        if y is int or slice: x[y] <==> x.__getitem__(y)
        if y is string: x['y'] <==> x.getbynumber(y)
        '''
        if type(i) is str:
            return self.getbynumber(int(i))
        else:
            return self.data.__getitem__(i)

# help(GitList)


# %% ----------------------------------------------------------------
if __name__ == '__main__':
    pass
