##### MODULI DA IMPORTARE #####

#import di url lib
try: 
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
    
#import Beatiful Soap
import bs4
from bs4 import BeautifulSoup
import requests

#import nltk modules
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import word_tokenize



import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from __future__ import print_function

#import re
import re

#varibili che mi servono
listblog = []
global t1text 
global t2text
global l1text
global l2text

global descsforcluster 
global keysforcluster 
global keyspre

descsforcluster = []
keysforcluster = []
keyspre = []


#pulisco la lista se Ë la prima volta che la uso
del listblog [:]


In [7]:
#istanzio una classe feed per salvare i risulati
class feed:
    listimp = []
    pass
#creo due classi che permettono di creare du feed rss separati
FeedFirst = feed()
FeedSecond = feed()


In [8]:
#Trovo i tag pi˘ frequenti
def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())


In [9]:
#pulisco la lista se non Ë nulla
def cleanlist(a):
    if not a:
        a
    else:
        del a[:]
    return a


In [10]:
#Stringa Rss da acquisire 
def getrss(strsrc):
    indirizzo = "http://www.feedbucket.com/?src=" + strsrc
    ricorda("Ho cercato il seguente indirizzo su feedbucket: http://www.feedbucket.com/?src=" + strsrc )
    return indirizzo


In [11]:
#da bs4tag (varibile di beautiful soup) a stringa
def frombs4tagtostring(a):
    paragraphs = []
    for x in a:
        paragraphs.append(str(x))
    str1 = ''.join(str(e) for e in paragraphs)
    #print("str1" + str1)
    return str1


In [12]:
#pulisco ulteriormente l html dato beautiful soup
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>|\n|[+‚+?ù]|EDT?')
    cleantext = re.sub(cleanr,'', raw_html)
    return cleantext


In [13]:
#creo una lista di parole chiave basandomi dal testoscelto
##!!! importante, non mi interessa particolarmente che questa operazione abbia un alto
###indice di similarit‡, perchË questa operazione 
def implist(feed):
    l = []
    q = 0
    for i in feed.NNP.get('NNP') :
        q = q + 1
        #scelgo i 4 sostantivi pi˘ comuni
        if q < 5:
            l.append(i)    
    q = 0
    for i in feed.NNS.get('NNS') :
        q = q + 1
        if q < 5:
            l.append(i)      
    q = 0
    for i in feed.JJ.get('JJ') :
        q = q + 1
        if q < 2:
            l.append(i)

    l = [x for x,_ in l]
    return l
    


In [14]:
#salve su di una lista il sito web
def saveblog(a):
    listblog.append(a)
    ricorda("Questa Ë la lista di siti gi‡ cercati:" + str(listblog))
    return


In [15]:
# spacchetto il dizionario nei siti e gli articoli
def get_keys(dictit):
    #salvo in variabile keys e nella variabile descs il dicit in modo da avere due liste una delle keys e una delle descs
    keys = list(dictit.keys())
    ricorda("Ecco gli indirizzi di blog papabili: ")
    ricorda(' '.join(keys))
    return keys
def get_cleandescs(dictit):
    descs = list(dictit.values())
    descs = cleandate(descs)
    return descs


In [16]:
# funzione che elimina tutti i giorni contratti della settimana e dei mesi, prima della tokenizzazione

def clean_date(query):
        stopwords = ['Mon,','Tue,','Wed,','Thu,','Fri,','Sat,','Sun,', 'Jan', 'Feb', 'Mar', 'Apr', 
                                                 'May', 'Jun', 'Jul', 'Aug','Sep','Oct','Nov','Dec']
        querywords = query.split()
        resultwords  = [word for word in querywords if word not in stopwords]
        result = ' '.join(resultwords)
        return result
#ciclo per pulire dalle date   
def cleandate(descs):
    list1 = []
    for desc in descs:
        d = clean_date(desc)
        list1.append(d)
    return list1


In [17]:
#richiedo la pagina html da internet
def getpage(a):
    strrss = getrss(a)
    p = requests.get(strrss)
    return p


In [18]:
#print
def ricorda(t):
    print(" "+t+"\n")
    return


In [19]:
#funzione che permette di controllare se il link di google non sia nei blog gi‡ trovati
#questo mi permette di non avere un futuro indice di correlazione = 1
def savedblog(a) :
    c = 1 ##vai
    for i in listblog:
        if i == a:
            c = 0 ## Ë uguale ad un altro nella lista
            ricorda("Da savedblog: Ho trovato un sito gi‡ analizzato")    
    return c


In [20]:
#vedo se feedbucket ritorna dei feedrss e quindi il link Ë un blog 
def isablog(p):
    a = 1
    if not p:
        a = 1 
    elif (p[0:6] == "Unable"):
        a = 0
        ricorda("Questo sito non ha rssfeed")
    return a


In [21]:
#pulisco l'indirzzo del sito
def yesisasite(a):
    cleanr = re.compile('http.|//|:|/.*')
    cleantext = re.sub(cleanr,'', a)
    return cleantext


In [22]:
#metodo che salva i due articoli presentati
def defineRSS(a,feed):
    ricorda("Sto definendo l'ultimo titolo e l'ultima descrizione del sito:" + a)
    page = getpage(a)
    soup = BeautifulSoup(page.text, "lxml")
    feed.link = a
    d = soup.findAll("div", { "class" : "item-contents" })
    feed.desc = cleanhtml(frombs4tagtostring(d[0]))
    t = soup.findAll("a", { "class" : "title" })
    feed.title = cleanhtml(frombs4tagtostring(t[0]))
    dcleaned = cleanhtml(frombs4tagtostring(d))
    #tokenizzi il testo e assegno i tag
    e = nltk.pos_tag(nltk.word_tokenize(dcleaned))
    #salvo i risultati di alcuni tag trovati
    feed.NNP = findtags('NNP', e)
    feed.NNS = findtags('NNS', e)
    feed.VB = findtags('VB', e)
    feed.JJ = findtags('JJ', e)
    #salvo il link del blog in modo da non averlo pi˘ nei risultati
    saveblog(a)
        #svuoto la lista delle parole chiave trovate //se volessi creare una semi intelligenza artificiale
        #dovrei non vuotarla
    feed.listimp = cleanlist(feed.listimp)
    feed.listimp = implist(feed)
    ricorda("Ecco la lista di parole chiave del sito" + a + ":" + str(feed.listimp))
            
    
    return


In [1]:
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer


def findthetwo(descs, keys, feed):
    stemmer = nltk.stem.porter.PorterStemmer()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

    def stem_tokens(tokens):
        return [stemmer.stem(item) for item in tokens]

    '''remove punctuation, lowercase, stem'''
    def normalize(text):
        return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

    def cosine_sim(text1, text2):
        tfidf = vectorizer.fit_transform([text1, text2])
        return ((tfidf * tfidf.T).A)[0,1]
    
    l = []   
    try:
        #se nella lista di keys trovata non esiste la seconda, chiudi il programma
        b=keys[1]
    except ValueError:
        print("E' imbarazzante, ma non ho trovato abbastanza siti")
        root.destroy
        sys.exit
    else:
        #trovo e salvo i due indici di similarit‡ pi˘ alti 
        best = cosine_sim(feed.desc, descs[0])
        bk = keys[0]
        second = cosine_sim(feed.desc, descs[1])
        sk = keys[1]
        i = 0 
        #trovo l'indice pi˘ alto
        for desc in descs:
            a = cosine_sim(feed.desc, desc)
            if keys[i] != sk:
                if max(a, best) == a:
                    bk = keys[i]
                    best = a
            i = i + 1
        #trovo il secondo indice pi˘ alto
        i = 0
        for desc in descs:
            a = cosine_sim(feed.desc, desc)
            if keys[i] != bk:
                if max(a, second) == a:
                    second = a
                    sk = keys[i]
            i = i + 1


        ricorda("Il blog:"+bk+" ha ottenuto il pi˘ alto indice " + str(best) + " , invece il blog: " + sk +  
                                                                " ha il secondo indice pi˘ alto ossia: " + str(second) )
        l.append(bk)
        l.append(sk)
    
    return l


In [93]:
#main attivato dalla pressione del pulsante
def gospidygo(a, feed):
    # trovo tutti i blog alternativi 
    dictit = findanalternative(feed)
    #di questi trovo i link
    keys = get_keys(dictit)
    ricorda("Ecco le liste di chiavi:" + str(keys))
    #salvo gli ultimi 10 articoli dei blog
    descs = get_cleandescs(dictit)
    
    #salvo le variabili trovate in variabili globali per future analisi
    savekd(keys, keysforcluster )
    savekd (descs, descsforcluster)
    savekd(keys, keyspre)
    
    #faccio il cluster di descs...
    clustorizethat(keysforcluster, descsforcluster)
    #trovo i due link con indice di correlazione pi˘ alti
    listsites = findthetwo(descs, keys, feed)
    #cancello dall'insieme di tutte le chiavi trovati in precenza le due stampate
    #in tk inteler
    ###!! questa funzione permette di stampare risultati anche di blog trovati in precedenza
    ### risolve alcuni problemi di passaggi di variabili vuote ed inoltre
    #### permette di avere una semi-intelligenza artificale in grado di 
    ##### ripercorrere passi precedenti (articoli di blog precedenti)
    deletelistsitesinkeyspre(listsites)
    
    #aggiorna la tkintler con i due link 
    createtwolabel(feed,listsites)
    
    return


In [94]:
#elimina dall'insieme di tutti i blog papabili quelli stampati in tkintler
def deletelistsitesinkeyspre(l2):
    global keyspre
    l1 = keyspre
    l3 = [x for x in l1 if x not in l2]
    keyspre = l3
    return


In [95]:
#funzione per salvare le keys per clusterizzare 
def savekd(tosave, saved):
    for i in tosave:
        saved.append(i)
    return saved


In [96]:
#trovo i blog papabili e i loro ultimi 10 articoli
def findanalternative(feed):
    lista_links = []
    dictit = {}
    #lista degli aggettivi, sostantivi del feed scelta dall'utente diventa stringa
    strfind = ' '.join(feed.listimp)
    #cerco su google i primi 10 risultati con le parole chiave salvate nella lista
    page = requests.get("https://www.google.it/search?q="+ strfind)
    ricorda(str("ecco cosa ho cercato: https://www.google.it/search?q= " + strfind))
    soup = BeautifulSoup(page.text, "lxml") #.get_text()
    #trovo tutti i link della pagina google
    links = soup.findAll('a')

    #prendo solo i link
    for link in links:
        if link['href'].startswith('/url?q='):
            lista_links.append(link['href'].replace('/url?q=', ''))

    #dei link trovati controllo che siano blog e nel caso salvo i suoi 10 articoli
    for link in lista_links :
        #pulisco il link
        link = yesisasite(link)
        #richiedo la pagina web da feedbucket
        page = getpage(link)
        soup = BeautifulSoup(page.text, "lxml")
        p = soup.p.string
        #controllo che abbia i feedrss
        if isablog(p) == 1 :
            #vedo se non Ë gi‡ stato salvato
            #if not link:
                if savedblog(link):
                    #richiedo i feed rss
                    d = soup.findAll("div", { "class" : "item-contents" })
                    t = soup.findAll("a", { "class" : "title" })
                    ricorda("Ho trovato un link interessante si tratta di:" + link)        
                    d = cleanhtml(frombs4tagtostring(d))
                    #salvo nel dizionario
                    dictit[link] = d
    
    ### ricontrollo il tutto con altri risultati di google, faccio ciÚ in quanto spesso 
    #### non trovo abbastanza blog...
    
    morepages = requests.get("https://www.google.it/search?q="+ strfind +"&start=10")
    soup = BeautifulSoup(morepages.text, "lxml") #.get_text()
    #trovo tutti i link della pagina google
    links = soup.findAll('a')

    for link in links:
        if link['href'].startswith('/url?q='):
            lista_links.append(link['href'].replace('/url?q=', ''))

    
    for link in lista_links :
        #pulisco il link
        link = yesisasite(link)
        #richiedo la pagina web da feedbucket
        page = getpage(link)
        soup = BeautifulSoup(page.text, "lxml")
        p = soup.p.string
        #controllo che abbia i feedrss
        if isablog(p) == 1 :
            #vedo se non Ë gi‡ stato salvato
            #if not link:
                if savedblog(link):
                    #richiedo i feed rss
                    d = soup.findAll("div", { "class" : "item-contents" })
                    t = soup.findAll("a", { "class" : "title" })
                    ricorda("Ho trovato un link interessante si tratta di:" + link)        
                    d = cleanhtml(frombs4tagtostring(d))
                    #salvo nel dizionario
                    dictit[link] = d
                    
    #restituisco il dizionario [linkblog][10articoli]
    return dictit


In [97]:
#aggiorno la pagina tkintler 
def createtwolabel(feed,listsites):
    
    if feed == FeedFirst:
        defineRSS(listsites[0], feed)
        t1text.set(feed.title)
        l1text.set(feed.desc)
        defineRSS(listsites[1], FeedSecond)
        t2text.set(FeedSecond.title)
        l2text.set(FeedSecond.desc)
    else:
        defineRSS(listsites[0], feed)
        t2text.set(feed.title)
        l2text.set(feed.desc)
        defineRSS(listsites[1], FeedFirst)
        t1text.set(FeedFirst.title)
        l1text.set(FeedFirst.desc)
    
    return
    


In [64]:
#funzione creata per calcolare la clusterizzazione delle descrizioni
def clustorizethat(keys, descs):
    #utilizzo le due liste primarie che sono keys - i siti internet papabili - e le descs - corpus creato 
    #dagli ultimi 10 articoli di ogni key
    
    # Creo una variabile chiamata stopwords usando il modulo di NlTK contiene tutte le parole come "at, the" ecc..
    stopwords = nltk.corpus.stopwords.words('english')
    
    # Creo una variabile che contiene lo stemmer in grado di ridurmi le derivazioni nella radice di provenienza - stemma -
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    
    # Creo due funzioni--
    # tokenize_and_stem: mi permette di ritornarmi una lista di Stemmi dato un corpus
    # tokenize_only: permette di tokenizzare il testo

    def tokenize_and_stem(text):
        #tokenizzo grazie alla funzione di NLTK 
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filtro il testo e tolgo tutti i punti, numeri ecc...
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        #creo la lista di stemmi e la ritorno
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems


    def tokenize_only(text):
        #tokenizzo grazie alla funzione di NLTK 
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filtro il testo e tolgo tutti i punti, numeri ecc...
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        #ritorno il testo tokenizzato 
        return filtered_tokens
    
    #Utilizzo le due funzioni create prima per creare due dizionari: il primo contente gli stemmi
    # e il secondo contenente le parole tokenizzate
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in descs:
        #per ogni termine in descs:  tokenize/stem
        allwords_stemmed = tokenize_and_stem(i) 
        #creo il vocabolario contente gli stemmi
        totalvocab_stemmed.extend(allwords_stemmed)
    
        allwords_tokenized = tokenize_only(i)
        #creo il vocabolario contente i token
        totalvocab_tokenized.extend(allwords_tokenized)
        
    #Uso queste due liste per creare un Dataframe (Grazie al modulo Pandas)
    #Il dataframe avr‡ nella prima colonna (index) gli stemmi e nella seconda le parole tokenizzate con stemma ideantico
    # in modo da avere ex 1) ran ---> ran, runs, running ecc.. 
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    ricorda('Ci sono: ' + str(vocab_frame.shape[0]) + ' item nel vocabolario stemmi/token')
    ricorda("<--->" + vocab_frame.head())
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Qui creo grazie alla funzione tf-idf vectorize, una matrice di corrispondeza tra gli stemmi e 
    # la presenza degli stessi nelle varie descrizioni. 
    # quindi asse y = stemmi asse x = docuementi e dall'incrocio dei due si conta il numero dello stemma,
    # nel tal documento
    # Ogni stemma viene quindi valutato a seconda del suo "peso/presenza" nei vari documenti

    #questa Ë la funzione principale, indico il peso che ogni stemma deve avere per essere "considerato"
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                     min_df=0.2, stop_words='english',
                                     use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    
    #creo la matrice dalle descrioni date in pasto
    tfidf_matrix = tfidf_vectorizer.fit_transform(descs) 
    #strtfidfshape = ''.join(tfidf_matrix.shape)
    strtfidfshape = str(tfidf_matrix.shape)
    ricorda("Ecco quanto Ë grande la matrice tfidf:" + strtfidfshape )
    
    
    #Ora creo un vocabolario selezionando i termini con pi˘ peso dalla matrice
    terms = tfidf_vectorizer.get_feature_names()
    #Ora colcolo la similarit‡ alla distanza definita dal coseno
    #Si calcola grazie alla formula 1 - coseno di similarit‡ di ciascun documento
    #Il coseno di similarit‡ Ë calcolato partendo dalla matrice tfidf e puÚ essere usato 
    #per generare una misura di similarit‡ tra ogni documento e il corpus 
    
    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1 - cosine_similarity(tfidf_matrix)    
        
    #K-means clustering
    #Ora userÚ la tecnica k means per clustorizzare il testo e rendere pi˘ visibile le somiglianze tra le 
    #varie descrioni
    #Scelgo di creare tre Cluster
    #Ogni osservazione (testo) Ë assegnato ad un cluster. Ogni osservazione Ë calcolata mediante 
    #la somma dei minimi (controlla meglio)
    # sotto radice quadrata ed successivamente ogni osservazione Ë asseganta come centroide
    # Questo calcolo viene reiterato costantemente fino a trovare il risultato migliore

    from sklearn.cluster import KMeans

    num_clusters = 2

    km = KMeans(n_clusters=num_clusters)

    %time km.fit(tfidf_matrix)

    clusters = km.labels_.tolist()

    # Ora creo un dizionario contente gli indirizzi all'indice e i cluster delle descrizioni 
    #corrispondenti (convertito poi con Pandas Dataframe)
    
    dictcluster = { 'title': keys, 'Ultimi 10 articoli': descs, 'cluster': clusters }

    frame = pd.DataFrame(dictcluster, index = [clusters] , columns = ['title', 'cluster'])
    #numero di link clusterizzati da 0 a 1
    
    ricorda("Ecco i termini pi˘ rilevanti per cluster:")
    print()
    #riordina il cluster a seconda dei centroidi pi˘ rilevanti
    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

    for i in range(num_clusters):
        print("Cluster numero %d: ha i seguenti termini pi˘ rilevanti:" % i, end='')
        
         #trovo i primi 6 termini per cluster e stampo
        for ind in order_centroids[i, :6]:
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
            print()
        
        print("Cluster %d contiene i seguenti indirizzi:" % i, end='')
        #for title in frame.ix[i]['title'].values.tolist():
        for title in frame.ix[i]['title'].values.tolist():
            print(' %s,' % title, end='')
            print()
            print()
            
    return
    


In [ ]:
####### Da far avviare per il programma


import tkinter as tk
from tkinter import * 
root = Tk()
root.geometry("1000x600")
root.title("Choose One")
t1text = StringVar()
t2text = StringVar()
l1text = StringVar()
l2text = StringVar()

listblog=[]
defineRSS("www.apple.com", FeedFirst)
defineRSS("www.egmcartech.com", FeedSecond)


#t2text.set(FeedSecond.title)
t1text.set(FeedFirst.title)
t2text.set(FeedSecond.title)
l1text.set(FeedFirst.desc)
l2text.set(FeedSecond.desc)

b1 = Button(root, text= "Scelta 1", font=("Roboto", 13), width=32,command= lambda: (gospidygo(FeedFirst.link, FeedFirst))).pack()
t1 = Label(root, textvariable = t1text , justify=LEFT, wraplengt=1000, font=("Roboto", 13)).pack()
l1 = Label(root, textvariable= l1text , justify=LEFT, wraplengt=1000, font=("Roboto", 13)).pack()
# ---><>><>>><<<>> #
b2 = Button(root, text= "Scelta 2", font=("Roboto", 13), width=32,command= lambda: (gospidygo(FeedSecond.link, FeedSecond))).pack()
t2 = Label(root, textvariable = t2text , justify=LEFT, wraplengt=1000, font=("Roboto", 13)).pack()
l2 = Label(root, textvariable = l2text , justify=LEFT, wraplengt=1000, font=("Roboto", 13)).pack()
ricorda("Il programma Ë stato avviato!")
root.mainloop()
