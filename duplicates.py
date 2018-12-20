'''
Строим векторную модель, используем tf-idf в качестве веса для каждого терма.
'''
import json
import re
from math import log10
from collections import UserList, Counter
import numpy as np
import csv
# import spacy
# nlp = spacy.load('en')


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

            issue['doc'] = issue['body']
            issue['code'] = []
            for p in (pattern_code_1, pattern_code_2):
                issue['code'].extend(p.findall(issue['doc']))
                issue['doc'] = p.sub('', issue['doc'])

            # issue['doc'] = nlp(issue['doc'])

            issue['tokens'] = re.sub('[^a-zA-Z]', ' ', issue['doc']).lower().split()

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

    name = 'googleapis-google-cloud-java'
    jsonfile = name + '-issues.json'

    with open(jsonfile, 'r', encoding='utf8') as file:
        r = GitList(json.load(file))

    # %% ----------------------------------------------------------------
    '''Find all empty issues and remove them from r'''
    empty_number = []
    for issue in reversed(r):
        if len(issue['tokens']) == 0:
            empty_number.append(issue['number'])
            r.remove(issue)

    # df -- document frequency
    # tf -- text frequency for each document
    words = []
    for issue in r:
        words.extend(issue['tokens'])
        issue['tf'] = Counter(issue['tokens'])
    df = Counter(words)

    # We sort out words that occur only once because they do not make any
    # impact into the scalar product.
    words = list(df)
    # words = [k for k,v in df.items() if v > 1]
    words.sort()

    # %% ------------------------------------------------------------------
    '''Calculate simmilarity'''
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
    sc = np.around(sc, decimals=2)

    # %% ------------------------------------------------------------------
    '''Grouping'''
    sim = []
    for n in np.around(np.arange(80, 101) * 0.01, decimals=2):
        cond = np.argwhere(sc == n)
        cond = [[i,j] for i,j in cond]

        sim.append([n])

        while len(cond) != 0:
            sim.append(cond.pop(0))
            i = 0
            find_them_all = []
            which_delete = []
            while len(cond) != 0:
                one, two = cond[i]
                remember = None
                if one in sim[-1] and two not in sim[-1]:
                    remember = two
                    find_them_all.append(True)
                    which_delete.append(i)
                elif one not in sim[-1] and two in sim[-1]:
                    remember = one
                    find_them_all.append(True)
                    which_delete.append(i)
                elif one in sim[-1] and two in sim[-1]:
                    which_delete.append(i)

                if len(find_them_all) == len(sim[-1]):
                    sim[-1].append(remember)
                    for j in reversed(which_delete):
                        del cond[j]
                    i = 0
                    find_them_all = []
                    which_delete = []
                    continue

                if i == len(cond)-1:
                    break
                else:
                    i += 1

    i = 0
    p = sim.pop(i)[0]
    while i != len(sim):
        if len(sim[i]) == 1:
            p = sim.pop(i)[0]
            continue
        sim[i].insert(0, p)
        i += 1

    sim.reverse()

    p = [i[0] for i in sim]
    for i, line in enumerate(sim):
        sim[i] = [r[j]['number'] for j in line[1:]]

    for i in range(len(sim)):
        sim[i].insert(0, p[i])

    # %% ------------------------------------------------------------------
    '''Writting to CSV'''
    csvfile = name + '-similar.csv'
    with open(csvfile, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(('Degree of similarity', ''))
        writer.writerow(('Issues', ''))
        writer.writerow(('States', ''))
        writer.writerow('')

        for line in sim:
            writer.writerow( [line[0]] )
            writer.writerow(line[1:])
            state = [r.getbynumber(i)['state'] for i in line[1:]]
            writer.writerow(state)
            writer.writerow('')

    # %% ------------------------------------------------------------------
    # print(r['3626']['body'])
    # print(r['3295']['body'])

    # r['3626']['tokens']
    # r['3295']['tokens']
