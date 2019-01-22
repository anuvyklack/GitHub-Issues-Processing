"""
Exports Issues from a specified repository to a JSON file
Uses basic authentication (Github username + password) to retrieve Issues
from a repository that username has access to. Supports Github API v3.
"""
import os
import json
import requests

# If you don't use login and password, GitHub's API v3 allows only 60
# requests per hour from one IP address.
# WARNING: if you use two-factor authentication you need to create personal
# access token, and use it in the password field.
# https://github.com/settings/tokens
GITHUB_USER = 'anuvyklack'
try:
    # try to use Environment Variables if exist
    GITHUB_PASSWORD = os.environ['GITHUB_PERSONAL_ACCESS_TOKEN']
except KeyError:
    GITHUB_PASSWORD = ''
REPO = 'googleapis/google-cloud-java'  # format is username/repo
# REPO = 'googleapis/google-cloud-python'  # format is username/repo
issue_state = 'all'  # possible state values: open, close, all
ISSUES_FOR_REPO_URL = 'https://api.github.com/repos/{}/issues'.format(REPO)
AUTH = (GITHUB_USER, GITHUB_PASSWORD)


def edit_issues(request, result):
    '''Edit issues to remain only necessary fields.'''
    # Which fields we want to remain
    cond = ('html_url', 'number', 'title', 'labels', 'state', 'body')
    if r.status_code != 200:
        raise Exception(request.status_code)

    # On issues request GitHub's API v3 returns both issues and pull requests
    # in the response. We sort out pull requests by the pull_request key.
    issues = [i for i in request.json() if 'pull_request' not in i]

    for issue in issues:
        keys = list(issue.keys())
        for key in keys:
            if key not in cond:
                del issue[key]
        issue['labels'] = [i['name'] for i in issue['labels']]
    result.extend(issues)
    return result


result = []
r = requests.get(ISSUES_FOR_REPO_URL, auth=AUTH, params={'state':issue_state})
r.url
result = edit_issues(r, result)

# more pages? examine the 'link' header returned
if 'link' in r.headers:
    pages = dict([(rel[6:-1], url[url.index('<')+1:-1]) for url, rel in
                  [link.split(';') for link in r.headers['link'].split(',')]])

    while 'last' in pages and 'next' in pages:
        r = requests.get(pages['next'], auth=AUTH)
        result = edit_issues(r, result)
        if pages['next'] == pages['last']:
            break
        pages = dict([(rel[6:-1], url[url.index('<')+1:-1]) for url, rel in
                      [link.split(';') for link in r.headers['link'].split(',')]])

print(len(result))

jsonfile = '{}-issues.json'.format(REPO.replace('/', '-'))
with open(jsonfile, 'w') as file:
    json.dump(result, file, indent=4)
