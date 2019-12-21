"load dataset and remove duplicated rows"
import matplotlib.pyplot as plt
import pandas as pd
from pylab import figure, axes, pie, title, savefig

df = pd.read_excel('patent_output2003_2009.xlsx')
df = df.drop_duplicates(keep = 'first')


###patent abstract text-preprocessing
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
#stop =['wherein','methods','comprise','ca','fas','specific','thereby','dl','l','entirely','finally','preferred','also','ehv','first','viable','limiting','s']
stop_b = stopwords.words('english')
#stops = stop+stop_b
stops = stop_b
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
import numpy
from gensim.matutils import hellinger
from pprint import pprint  # pretty-printer
from collections import defaultdict
import gensim
from gensim import corpora
from pprint import pprint
from gensim.parsing.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

"patent-abstract"
df_dtm = df[['patent number','application_year', 'Abstract']]
df_dtm = df_dtm.sort_values(['application_year'], ascending = True)
df_dtm = df_dtm.reset_index(drop=True)

abstract = []
for i in range(22341):
    abstract.append(df_dtm["Abstract"][i])

test = []
for i in range(22341):
    if type(abstract[i]) != str:
        test.append(i)

for j in test:
    abstract[j] = "empty"
print("it's done so far")

stop_2 = pd.read_csv("stopword_list.csv")
stop_sec = stop_2[stop_2['1']==1.0]
stop_temp = ['within','functionally','effective','irrespective','complementary','treating','live','upon','retarding','agents','little','preventing','risks','significant','predominately','suitable','produces','included','agent','guides','upon','adhesive','complement','employed','way','formulated','especially','within','administered','responses','moderate','describ','promote','contain','naturally','active','medium','least','predetermined','connected','covered','heavy','via',
            'introducting','another','present','least','forms','expressing','method','structure','exhibit','novel','either','use','activity','relea','achieving','associated','daily','generated','essentially','exceeding','loss','long','concurrent','utility','administering','state','subject','normally','causes','low','non','subsequent','affords','level','offer','make','two','presented','aforementioned','units','would','undergo','purpose','groups','event','adjacent','partial','time','segment','take','current','located','drawbacks','therefor','growth','development','prior','multi','promoting','status','entry','six','respect','induce','reacting','mutually','target','states','system','different','peculiar','charges','radical','contact','serial','and','therby''thereto','also','actual','without','with','domain','subject','administered','formed','prevents','cores','basic','physically','give','charge','developing','complete','third','damaged','sparingly','internal','implement','wound','occasionally','concerns','herein','inter','harden','transporter','shredded','higher','supplying','binding','selectively','respectively','accordance','uses','respectively','yield','releas','expressed','lines','yet','coding','proper','disrupts','particularly','biological','preservation','effects','identify','follows','activated','hard','together','likelihood','separately','functional','group','term','size','entity','producing']
stop_sec =list(stop_sec['aaa']) + stop_temp
stops = set(stops + stop_sec)

for i in range(22341):
    filtered_words = [word for word in abstract[i].split() if word not in stops]
    #filtered_words = [word for word in abstract[i].split() if word not in stop_sec]

    filtered_words = gensim.corpora.textcorpus.remove_short(filtered_words, minsize=3)
    abstract[i] = " ".join(filtered_words)

    abstract[i] = abstract[i].lower()
    abstract[i] = re.sub(r'[^\x00-\x7f]',r' ',abstract[i])
    abstract[i] = gensim.corpora.textcorpus.strip_multiple_whitespaces(abstract[i])

    abstract[i] = gensim.parsing.preprocessing.strip_punctuation2(abstract[i])

    abstract[i] = gensim.parsing.preprocessing.strip_numeric(abstract[i])

    abstract[i] = gensim.corpora.textcorpus.strip_multiple_whitespaces(abstract[i])
    abstract[i] = gensim.parsing.preprocessing.strip_short(abstract[i], minsize=3)
    #abstract[i] = [word for word in abstract[i].split() if word not in stops]
    #abstract[i] = " ".join(filtered_words)
    p = PorterStemmer()
    q = SnowballStemmer("english")
    abstract[i] = p.stem(abstract[i])
    abstract[i] = q.stem(abstract[i])

    query = abstract[i].split()
    resultwords = [t for t in query if t.lower() not in stops]
    result = " ".join(resultwords)
    abstract[i] = result
    query = abstract[i].split()
    resultwords = [t for t in query if len(t) >=3]
    result = " ".join(resultwords)
    abstract[i] = result
print('so far so good')
texts = [[text for text in doc.split()] for doc in abstract]
dictionary = corpora.Dictionary(texts)
print('so far so good')

from gensim.utils import simple_preprocess

tokenized_list = [simple_preprocess(doc) for doc in abstract]
mydict = corpora.Dictionary()
mycorpus = [mydict.doc2bow(doc, allow_update=True) for doc in tokenized_list]
print('good so far')


df_dtm['token'] = tokenized_list
dict = pd.DataFrame(dictionary.keys(), dictionary.values())
whole = []
for i in range(df_dtm.shape[0]):
    whole = whole + df_dtm['token'][i]
print('done')
from collections import Counter
key = Counter(whole)
dict_keyword = pd.DataFrame.from_dict(key, orient='index')

dict_keyword.columns=['cnt']
dict_keyword['cnt']
dict_keyword=dict_keyword.loc[(dict_keyword['cnt'] >= 210) ]
keyword = list(dict_keyword.index.values)
keyword.remove('methods')
keyword.remove('vegf')
keyword.remove('gram')
keyword.remove('chronic')
keyword.remove('variant')

len(keyword)

def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)

tokn = tokenized_list
for i in range(22341):
    tokn[i] = Intersection(keyword, tokn[i])

df_dtm['keyword_tokn'] = tokn


##word2vec analysis
import nltk
from gensim.models.word2vec import Word2Vec
#% % time
model = Word2Vec(df_dtm['keyword_tokn'], size=600, window = 10, sample = 1e-5, iter=500, sg=1)

print('word2vec done')

#model.wv.most_similar("tumor")
len(model.wv.vocab)
model.save('word2vec.model')

from gensim.models import Word2Vec
import matplotlib.pyplot as plt

def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize=(8 ,6))
    plt.scatter(xs, ys, marker = 'o')
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))

word_vectors = model.wv
vocabs            = word_vectors.vocab.keys()
word_vectors_list = [word_vectors[v] for v in vocabs]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xys = pca.fit_transform(word_vectors_list[:15])
xs = xys[:,0]
ys = xys[:,1]

plot_2d_graph(vocabs, xs, ys)

###cnn
