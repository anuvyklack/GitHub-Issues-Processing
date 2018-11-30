'''
Строим векторную модель, используем tf-idf в качестве веса для каждого терма.
'''
import json
import re
import string
import nltk
import numpy as np
from math import log10
nltk.download('punkt')

jsonfile = 'googleapis-google-cloud-java-issues.json'  # format is username/repo

with open(jsonfile, 'r', encoding='utf8') as file:
    r = json.load(file)
len(r)


def getbynumber(n, r):
    for i, issue in enumerate(r):
        if issue['number'] == n:
            break
    return i


# %% ------------------------------------------------------------------
# pattern for links
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
pattern = re.compile(url_regex)

# pattern for links in markdown ambient
# http:// or https:// followed by anything but a closing paren
short_url_regex = 'http[s]?://[^)]+'
# HACK: if use url_regex here, re.findall function just hanging up the python
# interpreter for some strings (for example, issue #3924 in
# 'googleapis/google-cloud-java')
pattern_md = re.compile('\[.*]\(\s*({0})\s*\)'.format(short_url_regex))

# translation table to removing special symbols and punctuation for
# str.translate() method
special_char = '\n\a\b\f\r\t\v'
# trantab = ''.maketrans(special_char, ' ' * len(special_char), string.punctuation)
trantab = ''.maketrans(special_char + string.punctuation,
                       ' ' * (len(special_char) + len(string.punctuation)))

for issue in r:
    # find all links, but for links in markdown ambient also takes last
    # round bracket and punctuation after it
    urls = pattern.findall(issue['body'])
    # find all links in markdown ambient
    md_urls = pattern_md.findall(issue['body'])

    # change in 'urls' all markdown links to the correct ones
    # WARNING: we assume here that each link appears in the text only once
    for mdu in md_urls:
        for n, url in enumerate(urls):
            if url.find(mdu) != -1:
                urls[n] = mdu
                break
    issue['urls_in_body'] = urls

    # delete all urls from the issue body
    issue['tokens'] = issue['body']
    for url in urls:
        issue['tokens'] = issue['tokens'].replace(url, '')
    # remove all special symbols, punctuation and convert to lowercase
    issue['tokens'] = issue['tokens'].translate(trantab).lower()
    # substitute multiple whitespace with single whitespace

    # issue['tokens'] = re.sub(' +', ' ', issue['tokens'])
    issue['tokens'] = nltk.word_tokenize(issue['tokens'])

# %% ------------------------------------------------------------------
'''Find all empty issues and remove the from r'''
empty_number = []
for issue in reversed(r):
    if len(issue['tokens']) == 0:
        empty_number.append(issue['number'])
        r.remove(issue)

# %% ------------------------------------------------------------------
df = {}  # document frequency
for issue in r:
    if issue.get('tf') is None: issue['tf'] = {}
    tokens = {t for t in issue['tokens'] if not t.isdigit()}
    for token in tokens:
        issue['tf'][token] = issue['tokens'].count(token)
        if token not in df:
            df[token] = 1
        else:
            df[token] += 1

# We sort out words that occur only once because they do not make any
# impact into the scalar product.
# TODO: there are appers issues consists only from one unic word.
# For them norm of the vetor is zero.
# words = [k for k,v in df.items() if v > 1]
words = [k for k,v in df.items()]
words.sort()
df = {k: df[k] for k in words}

# most_common = {k: v for k,v in sorted(df.items(), key=lambda d: d[1], reverse=True)}
# less_common = {k: v for k,v in sorted(df.items(), key=lambda d: d[1])}

# %% ------------------------------------------------------------------
# array of vectors
av = np.empty((len(r), len(words)))
idf = np.array([log10(len(r) / df[k]) for k in words])

for i, issue in enumerate(r):
    av[i] = np.array([issue['tf'][k] if k in issue['tf'] else 0 for k in words])
av = av * idf

# calculate norm of each vector
normav = np.sqrt(np.einsum('ij, ij-> i', av, av))

# scalar product
sc = np.zeros((len(r), len(r)))
for i, vec in enumerate(av):
    sc[i, i+1:] = np.einsum('ij, j-> i', av[i+1:], vec) / (normav[i] * normav[i+1:])
cond = np.argwhere(sc > 0.8)

# number, number, probability
sim = []
for i,j in cond:
    sim.append((r[i]['number'], r[j]['number'],
                r[i]['state'], r[j]['state'], float('%.2f' % (sc[i,j]))))
sim.sort(key=lambda i: i[-1], reverse=True)

for i in sim:
    print(i)

# print(r[getbynumber(sim[-1][0], r)]['body'])
# print(r[getbynumber(sim[-1][1], r)]['body'])
