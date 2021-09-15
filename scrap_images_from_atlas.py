import requests
from bs4 import BeautifulSoup
import re
import urllib2

def find_number(text, c):
    return re.findall(r'%s(\d+)' % c, text)[0]

for i in range(1,502):
  url = '//www.atlasdermatologico.com.br/disease.jsf?diseaseId='+str(i)

  response = requests.get('http:' + url)

  soup = BeautifulSoup(response.text, "html.parser")

  images = soup.find_all('img')

  for image in images: 
    #print(image)
    src = image['src']
    num=find_number(src, 'imageId=')
    cat = image['alt']
    photo='http://www.atlasdermatologico.com.br/img?imageId='+num
    urllib2.request.urlretrieve(photo, '/content/drive/MyDrive/Atlas-images/'+cat+str(num)+".jpeg")
