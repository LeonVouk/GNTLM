import json
import requests

from bs4 import BeautifulSoup


INITIAL_URL = "https://en.wikipedia.org/wiki/Category:Living_people"
HREF_LINKS = []

def get_page_links(url=INITIAL_URL):
    page_links = []
    response = requests.get(
        url=url,
    )
    html_page = response.content
    soup = BeautifulSoup(html_page, "html.parser")
    href_list = [link for link in soup.findAll('a') if link.get('href')]
    for link in href_list:
        if 'https://en.wikipedia.org/wiki/Category:Living_people' in link.get('href') \
                and link.get('href') != 'https://en.wikipedia.org/wiki/Category:Living_people':
            page_links.append(link.get('href'))
    return page_links


def scrape(url=INITIAL_URL, first=True):
    print(f'Scraping {url}')
    response = requests.get(
        url=url,
    )
    html_page = response.content
    soup = BeautifulSoup(html_page, "html.parser")
    href_list = [link for link in soup.findAll('a') if link.get('href')]
    n_lower = 0
    for n, _l in enumerate(href_list):
        if _l.get('href') == 'https://en.wikipedia.org/wiki/Category:Living_people?from=Zt':
            n_lower = n + 1
            break
    list_of_possible_people = [l.get('href') for l in href_list[n_lower:]]
    for pp in list_of_possible_people:
        if pp.startswith('/wiki/') \
                and 'category' not in pp.lower() \
                and 'wikipedia' not in pp.lower():
            HREF_LINKS.append(pp)
    if first:
        page_links = get_page_links()
        print(page_links)
        for hlink in page_links:
            scrape(url=hlink, first=False)


if __name__ == '__main__':
    scrape()
    modified_list = [item.replace('/wiki/', '') for item in HREF_LINKS]
    with open('data/people_data/people_slugs.json', 'w') as f:
        json.dump(modified_list, f, indent=4)
