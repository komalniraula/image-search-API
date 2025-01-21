from flask import Flask, jsonify
import re
import requests, uuid, json
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from keybert import KeyBERT
from nltk.corpus import stopwords
import itertools
import concurrent.futures
from flask import request
from langdetect import detect

app = Flask(__name__)

model_freepik = SentenceTransformer('paraphrase-MiniLM-L6-v2')
kw_model = KeyBERT()

def text(user_text):
    nepali_text = user_text.replace("TT", "teacher").replace("SS", "student")
    final_text = re.sub("[\(\[].*?[\)\]]", "", nepali_text).split("<<", 1)[0]
    return final_text

def translator(input_edited):
    # API key and endpoint
    key = " "
    endpoint = "https://api.cognitive.microsofttranslator.com/"

    # location, also known as region.
    # required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
    location = "eastasia"

    path = '/translate'
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0',
        'from': 'ne',
        'to': ['en']
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        # location required if you're using a multi-service or regional (not global) resource.
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    
    # You can pass more than one object in body.
    body = [{
        'text': input_edited
    }]
    
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]

def translated_extraction(translated):
    
    text_dic = translated["translations"]
    text = text_dic[0]
    translated_text = text["text"]
    
    return translated_text

def keyword_extraction(full_text):    
    sentence = nltk.sent_tokenize(full_text)
    freepik_search = []
    google_search = []
    nouns = []
    proper_noun_main = []
    proper_noun = []
    key_word = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
    #print(key_word)
    #print('-------------------')
    words = nltk.word_tokenize(full_text)
    words = [word for word in words if word not in set(stopwords.words('english'))]            
    tagged = nltk.pos_tag(words)
    for (word, tag) in tagged:
        if tag == 'NN':
            if tag != 'NNP' or tag != 'NNPS':
                nouns.append(word)
        elif tag == 'NNP' or tag == 'NNPS':
            proper_noun_main.append(word)
    #print('nouns :', nouns)
    #print('proper nouns main:', proper_noun_main)
    proper_noun = [x.lower() for x in proper_noun_main]
    
    if type(key_word[0]) == list:
        key_word = list(itertools.chain.from_iterable(key_word))

    #print('proper nouns :', proper_noun)
    for w in key_word:
        wr, cos = w
        if cos > 0.5:
            freepik_search.append(wr)
        else:
            sp = wr.split()
            for k in sp:
                if k in nouns:
                    if len(sp) == 2:
                        first_word = sp[0]
                        second_word = sp[1]
                        if first_word in nouns:
                            if second_word in nouns:
                                if wr not in freepik_search:
                                    freepik_search.append(wr)
                            elif second_word not in nouns:
                                if first_word not in freepik_search:
                                    freepik_search.append(first_word)
                        elif second_word in nouns:
                            if second_word not in freepik_search:
                                freepik_search.append(second_word)

                    elif wr not in freepik_search:
                        freepik_search.append(wr)
                elif k in proper_noun:
                    if len(sp) == 2:
                        first_word = sp[0]
                        second_word = sp[1]
                        if first_word in proper_noun:
                            if second_word in proper_noun:
                                if wr not in google_search:
                                    first_word_cap = first_word.capitalize()
                                    second_word_cap = second_word.capitalize()
                                    if first_word_cap in proper_noun_main and second_word_cap in proper_noun_main:
                                        google_search.append(first_word)
                                        google_search.append(second_word)
                                    #elif first_word_cap not in proper_noun_main and second_word_cap in proper_noun_main:
                                    #   google_search.append(first_word)
                                    #  google_search.append(second_word)
                                    elif first_word_cap in proper_noun_main and second_word_cap not in proper_noun_main:
                                        google_search.append(wr)
                            elif second_word not in proper_noun:
                                if first_word not in google_search:
                                    google_search.append(first_word)
                        elif second_word in proper_noun:
                            if second_word not in google_search:
                                google_search.append(second_word)            
                    elif wr not in google_search:
                        google_search.append(wr)
            #sentence = sentence.replace('.', "")
            #sentence_word_dict[sentence] = search_words                #print(fin)
        #print('||||||||||||||||||||||')

        #print('freepik search: ', freepik_search)
    google_search = list(set(google_search))
    #res.append(x) for x in test_list if x not in res
    #print('google search: ', google_search)
    #print(total_sentences)
    return freepik_search

def find_keywords(user_input):
    global translated_text
    input_edited = text(user_input)
        
    lang = detect(input_edited)

    if lang == 'en' or lang == 'so':
        translated_text = input_edited
        final = keyword_extraction(translated_text)

    else:
        translated_text = translated_extraction(translator(input_edited))
        
        words = re.findall('[a-z]+', input_edited, flags=re.IGNORECASE)

        words = [word for word in words if word not in set(stopwords.words('english'))]  
        
        freepik_search = keyword_extraction(translated_text)
        final = freepik_search + words 
    
    key_words = list(set(final)) 
    #print(key_words)
    return key_words

def get_freepik_imglist(image_title):
    results = {}
    
    for i in range(3):
        url = 'https://www.freepik.com/search?format=search&page={pag}&query={key}&type=vector'.format(pag = i + 1, key = image_title)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.text, 'html.parser')
        for div in soup.find_all('p', {'class':"cleaned-filters"}):
            return results

        for div in soup.find_all('a', {'class':"showcase__link"}):
            img = div.find('img', alt=True)
            title = img['alt']
            title = title.lower()
            link = div['href']

            every_word = link.split('.')

            if 'freepik' in every_word:
                try:
                    lin = link.split("vector/",1)[1]

                except:
                    lin = link.split("ai-image/",1)[1]


                #print(lin)
                final_til = title + ' ' + lin[:-4].replace('-', ' ').lower()

                results[final_til] = link
    
    return results

def embeddings(title_list, full_text):
    embeddings = model_freepik.encode(title_list)
    embeddings_for_main = model_freepik.encode(full_text)
    embeddings_list = {}
    for title, embedding in zip(title_list, embeddings):
        embeddings_list[title] = embedding
    return embeddings_for_main, embeddings_list

def get_cosine(embeddings_for_main, embeddings_list):
    cosine_value = {}
    for emb_key in embeddings_list:
        c = cosine_similarity([embeddings_for_main], [embeddings_list[emb_key]])
        k = float(c[0])
        if k >= 0.5:
            cosine_value[emb_key] = k
    
    if len(cosine_value) == 0:
        for emb_key in embeddings_list:
            c = cosine_similarity([embeddings_for_main], [embeddings_list[emb_key]])
            k = float(c[0])
            if len(cosine_value) != 2:
                cosine_value[emb_key] = k
                cosine_value = {k: v for k, v in sorted(cosine_value.items(), emb_key=lambda item: item[1], reverse = True)}
            else:
                return cosine_value
        
    cosine_value = {k: v for k, v in sorted(cosine_value.items(), emb_key=lambda item: item[1], reverse = True)}
    return cosine_value

def get_image_link(cosine_value, results):
    image_link = []
    
    for csn_key in cosine_value:
        image_link.append(results[csn_key])
        used_images.append(csn_key)

    return image_link

def img_source(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, 'html.parser')
    for x in soup.findAll('div', {'class':"detail__gallery detail__gallery--vector alignc"}):
        img = x.find('img', alt=True)
        source[url] = img['src']
    return 0

def img_title(image_title):
    
    results = get_freepik_imglist(image_title)
    title_list = list(results.keys())
    
    embeddings_for_main, embedding_list = embeddings(title_list, translated_text)
    cosine_value = get_cosine(embeddings_for_main, embedding_list)
    
    image_link = get_image_link(cosine_value, results)
    if len(image_link) != 0:
        with concurrent.futures.ThreadPoolExecutor() as executor: 
            executor.map(img_source, image_link)
    
    return 0

def img_freepik(freepik_search):
    #complete_image_path = {}

    with concurrent.futures.ThreadPoolExecutor() as executor: 
        executor.map(img_title, freepik_search)
        
    return 0


@app.route('/freepikImage', methods=["GET", "POST"])
def freepikImage():
    
    if request.method == 'POST':
        global source, used_images
        source = {}
        used_images = []


        content = request.json
        
        user_input = content['sentence']
        
        words = find_keywords(user_input)  

        img_freepik(words)

        return jsonify({"value": source})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
