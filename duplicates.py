'''
Строим векторную модель, используем tf-idf в качестве веса для каждого терма.
'''
import json
import re
import string
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
        # pattern for links
        url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        pattern = re.compile(url_regex)

        # pattern for links in markdown ambient
        # http:// or https:// followed by anything but a closing paren
        short_url_regex = r'http[s]?://[^)]+'
        # HACK: if use url_regex here, re.findall function just hanging up the python
        # interpreter for some strings (for example, issue #3924 in
        # 'googleapis/google-cloud-java')
        pattern_md = re.compile(r'\[.*]\(\s*({0})\s*\)'.format(short_url_regex))

        # pattern for code
        pattern_code = re.compile(r'([a-zA-Z]+[0-9\$\.\:]+[a-zA-Z]+)')

        for issue in self.data:
            # find all links, but for links in markdown ambient also takes last
            # round bracket and punctuation after it
            urls = pattern.findall(issue['body'])
            # find all links in markdown ambient
            md_urls = pattern_md.findall(issue['body'])

            # change in 'urls' all markdown links to the correct ones
            # WARNING: we assume here that each link appears only once in the text
            for mdu in md_urls:
                for n, url in enumerate(urls):
                    if url.find(mdu) != -1:
                        urls[n] = mdu
                        break
            issue['urls_in_body'] = urls

            # delete all urls from the issue body
            issue['tokens'] = issue['body']  # copy string
            for url in urls:
                issue['tokens'] = issue['tokens'].replace(url, '')

            # issue['tokens'] = issue['tokens'].lower()

            issue['code'] = pattern_code.findall(issue['tokens'])
            issue['tokens'] = pattern_code.split(issue['tokens'])
            for n, token in enumerate(issue['tokens']):
                if token not in issue['code']:
                    issue['tokens'][n] = re.sub('[^a-zA-Z]', ' ', token).lower()

            issue['tokens'] = ' '.join(issue['tokens']).split()


    def getbynumber(self, n):
        '''Return the issue (list element) with given key "number" value.'''
        for i, issue in enumerate(self.data):
            if issue['number'] == n:
                return self.data[i]
        raise ValueError('no such issue with number {}'.format(n))

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

    name = 'googleapis-google-cloud-java'
    jsonfile = name + '-issues.json'

    with open(jsonfile, 'r', encoding='utf8') as file:
        r = GitList(json.load(file))

    # print(r['3626']['body'])
    # print(r['3295']['body'])

    # r['3626']['tokens']
    # r['3295']['tokens']

    # %% ------------------------------------------------------------------
    '''Find all empty issues and remove them from r'''
    empty_number = []
    for issue in reversed(r):
        if len(issue['tokens']) == 0:
            empty_number.append(issue['number'])
            r.remove(issue)

    # %% ------------------------------------------------------------------
    # df -- document frequency
    # tf -- text frequency for each document
    words = []
    for issue in r:
        words.extend(issue['tokens'])
        issue['tf'] = Counter(issue['tokens'])
    df = Counter(words)

    # %% ------------------------------------------------------------------
    # We sort out words that occur only once because they do not make any
    # impact into the scalar product.
    words = list(df)
    # words = [k for k,v in df.items() if v > 1]
    words.sort()
    # df = {k: df[k] for k in words}

    # %% ------------------------------------------------------------------
    # array of vectors
    av = np.empty((len(r), len(words)))
    idf = np.array([log10(len(r) / df[k]) for k in words])

    for i, issue in enumerate(r):
        av[i] = np.array([issue['tf'][k] if k in issue['tf'] else 0 for k in words])
    av = av * idf

    # calculate norm of each vector
    normav = np.sqrt(np.einsum('ij, ij-> i', av, av))

    # Delete all issues, that consists only from 1 unice word. Their norm is zero.
    z = np.where(normav == 0)
    av = np.delete(av, z, axis=0)
    for i in z[0]:
        del r[i]
    normav = np.delete(normav, z)

    # scalar product
    sc = np.zeros((len(r), len(r)))
    for i, vec in enumerate(av):
        sc[i, i+1:] = np.einsum('ij, j-> i', av[i+1:], vec) / (normav[i] * normav[i+1:])
    cond = np.argwhere(sc > 0.8)

    # (number 1, number 2, state 1, state 2, degree of similarity)
    sim = []
    for i,j in cond:
        sim.append((r[i]['number'], r[j]['number'],
                    r[i]['state'], r[j]['state'], float('%.2f' % (sc[i,j]))))
    sim.sort(key=lambda i: i[-1], reverse=True)

    for i in sim:
        if 'open' in i:
            print(i)

    # %% ------------------------------------------------------------------
    csvfile = name + '-similar.csv'
    with open(csvfile, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(('Issue#1', 'Issue#2', 'State#1', 'State#2', 'Degree of similarity'))
        for line in sim:
            writer.writerow(line)
