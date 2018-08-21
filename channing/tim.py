import requests
from bs4 import BeautifulSoup
import re
def get_words(url = 'https://www.nytimes.com/2016/11/02/world/americas/canada-patrick-lagace-montreal-police.html'):
    r = requests.get(url)
    assert r.status_code == 200
    soup = BeautifulSoup(r.text, 'lxml')
    body = soup.find_all('p', {"class": "css-1i0edl6 e2kc3sl0"})
    body = ' '.join(map(str, body))
    body = re.sub('\<[^\>]*\>',' ', body)
    return body.split()
