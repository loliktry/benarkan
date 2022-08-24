from concurrent.futures.process import _threads_wakeups
from operator import countOf
import requests
from bs4  import BeautifulSoup
from nltk import tokenize
from firebase import firebase
import re, string

firebase = firebase.FirebaseApplication('https://berita-benar-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

url = 'https://www.kompas.com/'
html        = requests.get(url)
soup        = BeautifulSoup(html.content, 'lxml')

populer     = soup.find('div', {'class', 'most__wrap clearfix'})
isi_berita  = populer.find_all('div', {'class', 'most__list clearfix'})
list_judul_berita = []
for each in isi_berita:
    nomor = each.find('div',{'class','most__count'}).text
    judul_berita = each.find('h4',{'class','most__title'}).text
    list_judul_berita.append(judul_berita) 
    link_berita = each.a.get('href')+"?page=all"
    print(nomor)
    print(judul_berita)
    print(link_berita)
    print('')
    
links = []
for each in isi_berita:
    links.append(each.a.get('href'))

print("Jumlah berita adalah {}".format(len(links)))   
i = 0

def bersihkan(kalimat_berita):
    no_punct = ""
    for char in kalimat_berita:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def word_drop(text):
    remove = string.punctuation
    remove = remove.replace(".", "")
    remove = remove.replace('"', "")
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    #text = re.sub("\\W"," ",text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(remove), '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)    
    return text




for link in links:
    i       = i + 1
    page    = requests.get(link)
    halaman_berita    = BeautifulSoup(page.content, 'lxml')
    Isi_Berita = halaman_berita.find('div',{'class','read__content'}).text

    berita = tokenize.sent_tokenize(Isi_Berita)
    beritanya = []
    for kalimat_berita in berita:
        if "KOMPAS.com - " in kalimat_berita: kalimat_berita = kalimat_berita.replace("KOMPAS.com - ","")
        if kalimat_berita.lower().find("baca juga:") >= 0 :
            kalimat_berita = ''         
                
        if kalimat_berita.lower().find('dapatkan update berita pilihan') >= 0 :
            kalimat_berita = ''
            break

        #kalimat_berita = cleanedthings(kalimat_berita)
        beritanya.append(kalimat_berita)
        #print (kalimat_berita)
    
    kalimat_berita_bersih = word_drop(' '.join(beritanya))
    
    #cleanedthings(kalimat_berita_bersih)
    if "  " in  kalimat_berita_bersih:  kalimat_berita_bersih =  kalimat_berita_bersih.replace("  "," ") 
    
    firebase.post('/Berita', {'Judul':list_judul_berita[i-1],'Konten':kalimat_berita_bersih})
    print('======= berita ke-{} ======='.format(i))
    print(kalimat_berita_bersih)



    