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


def extract_markdown_links(text):
    r'''
    Find and extract all markdown links (links in markdown ambient).
    Takes str and return list of urls and str without links.
    Input:
        'Random text \[GitHub\]\(http://github.com\)...'
    Output:
        ['http://github.com'], 'Random text GitHub...'
    '''
    # pattern for links in markdown ambient
    # http:// or https:// followed by anything but a closing paren
    short_url_regex = r'http[s]?://[^)]+'
    # HACK: if use url_regex here, re.findall function just hanging up the python interpreter
    # for some strings (for example, issue #3924 in # 'googleapis/google-cloud-java')
    pattern_md_url = r'\[(?P<NAME>.*)]\(\s*({0})\s*\)'.format(short_url_regex)

    urls = [j for i,j in re.findall(pattern_md_url, text)]
    text = re.sub(pattern_md_url, r'\g<NAME>', text)
    return urls, text


def extract_links(text):
    '''
    Find and extract all links from the string.
    Return list of links and str without links.

    Input:
        'Random text https://github.com continue text'
    Output:
        \[https://github.com\], 'Random text  continue text'
    '''
    # Pattern for links. Doesn't take a dot if there is any space character,
    # new line of end of the string after it.
    pattern_url = r'[ ]?(http[s]?://(?:[a-zA-Z]|[0-9]|[!*#$-\-\/-_\(\)]|\.(?!\s|$))+)'
    # pattern_url = r'[ ]?(http[s]?://(?:[a-zA-Z]|[0-9]|[!*#$-\-\/-_\(\)]|(?:%[0-9a-fA-F]{2})|\.(?!\s|$))+)'
    # pattern_url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*#\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # pattern_url = url_regex
    urls = re.findall(pattern_url, text)
    text = re.sub(pattern_url, '', text)
    return urls, text


def extract_github_code_snippets(text):
    '''
    Extract into list all GitHub code snippets from the string.
    '''
    # Find code snippents inside string of GitHub Markdown. Use with re.DOTALL flag
    pattern1 = '```.*?\r?\n(.+?)\r?\n```'
    # Pattern: 4 whitespaces or tab, after that, any sets of any symbols, including none,
    # after that any number of new line wymbols. Repeat the Pattern as many times as possible.
    pattern2 = '(?:(?: {4}|\t).*(?:\r?\n)*)+'

    code = []
    code.extend( re.findall(pattern1, text, flags=re.DOTALL) )
    text = re.sub(pattern1,'', text)

    code.extend( re.findall(pattern2, text) )
    text = re.sub(pattern2, '', text)

    return code, text


class GitList(UserList):
    """List for GitHub Issues"""

    def __init__(self, initlist=None):
        super().__init__(initlist)
        if initlist:
            self.__initialazer()

    def __initialazer(self):
        for issue in self.data:
            issue['doc'] = issue['body']
            issue['urls'], issue['doc'] = extract_markdown_links(issue['doc'])
            urls, issue['doc'] = extract_links(issue['doc'])
            issue['urls'].extend(urls)
            issue['code'], issue['doc'] = extract_github_code_snippets(issue['doc'])

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
