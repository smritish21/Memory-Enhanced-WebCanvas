import json
from openai import OpenAI
import os
import urllib.parse
import re
import psycopg2
#from chroma_db import answer_query_ch
import csv
from datetime import datetime
#import chromadb
from urllib.parse import urlparse, parse_qs
#from rapidfuzz import fuzz
from evaluate.database_wc import get_db_connection,check_existing_url
from evaluate.rag_query import decide_website, get_embedding

#chroma_client = chromadb.PersistentClient(path="/Users/smritisharma/Downloads/Thesis/WebCanvas/evaluate/chroma_data")

client = OpenAI()

# Known "dynamic" query terms (can be extended)
KNOWN_DYNAMIC_KEYS = [
    'q', 'query', 'search', 'keyword', 'term', 'kw',
    'location', 'from', 'to', 'date', 'category', 'brand', 'name',
    'txt', 'txtSearch', 'destination', 'origin', 'ref_'
]

# Threshold for fuzzy similarity (0–100)
FUZZY_MATCH_THRESHOLD = 80

def is_dynamic_key(key):
    key = key.lower()
    for known in KNOWN_DYNAMIC_KEYS:
        if fuzz.partial_ratio(key, known) >= FUZZY_MATCH_THRESHOLD:
            return True
    return False

def is_dynamic_value(value):
    # Dynamic if value looks like a date or readable keyword
    import re
    if re.search(r'\d{4}-\d{2}-\d{2}', value):  # date-like
        return True
    if re.match(r'^[a-zA-Z\s\+\%]+$', value):   # human-readable
        return True
    return False

def classify_url(url):
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    #print(f'Parsed url {parsed} query params {query_params}')
    if not query_params:
        return "static"

    for key, values in query_params.items():
        value = values[0] if values else ""
        if not (is_dynamic_key(key) or is_dynamic_value(value)):
            return "static"
    return "dynamic"


#os.environ["OPENAI_API_KEY"] ='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'
#client = OpenAI()

url = "https://www.flightaware.com/"
prompt = f'''You are provided with an URL. You need to identify how to genralize the URL such that it can be used with any query. 
            A query is usually present in the parameter of the URL. Identify if the URL can be genralized and return the 
            generalized URL,in the json format :-
                generalizable:_True or False
                genralized_url: genralized url
            This is the {url} url to check.'''
'''
response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "assistant", "content": prompt}],
        temperature=0.0
    )
message = response.choices[0].message.content
print(message)
'''


def generalize_url(url):
    parsed_url = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_url.query)

    generalized_params = {}
    for key, values in query_params.items():
        value = values[0] if values else ''

        # Heuristics to determine if it should be generalized
        if value.strip() == '':
            generalized_params[key] = f'{{{key}}}'  # empty but meaningful param
        elif any(char.isalpha() for char in value):  # likely user input
            generalized_params[key] = f'{{{key}}}'
        else:
            # Keep as-is or generalize based on your needs
            generalized_params[key] = f'{{{key}}}'  # or just value, e.g. ConceptID

    # Rebuild the query string
    generalized_query_string = urllib.parse.urlencode(generalized_params, doseq=True)

    # Construct generalized URL
    generalized_url = urllib.parse.urlunparse((
        parsed_url.scheme, parsed_url.netloc, parsed_url.path,
        parsed_url.params, generalized_query_string, parsed_url.fragment
    ))

    return generalized_url




def classify_param(key, value):
    if value == "":
        return "dynamic"
    if len(value) > 30 and value.isdigit():
        return "static"
    if re.search(r'(\w+:\w+)', value):  # has colon-separated pairs
        return "dynamic"
    if re.search(r'\d{4}-\d{2}-\d{2}', value):  # date
        return "dynamic"
    if key.lower() in ['from', 'to', 'budget', 'q', 'search', 'location']:
        return "dynamic"
    if any(char.isalpha() for char in value) and not value.isupper():
        return "dynamic"
    return "static"

def analyze_url(url):
    print(url)
    parsed = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed.query)

    results = {}
    for key, values in query_params.items():
        value = values[0] if values else ""
        classification = classify_param(key, value)
        results[key] = {
            "value": value,
            "type": classification
        }

    return results

def extract_website_name(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc  
    
    
    return domain

#print(analyze_url('https://www.gamestop.com/video-games/playstation-5'))

def add_url_to_db(data):
    data = {k: (None if v is None else v) for k, v in data.items()}
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS url_parameters (
        task TEXT,
        base_url TEXT,
        url_name TEXT,
        param_name TEXT,
        param_value TEXT
    )
    """)
    with conn.cursor() as cur:
        for i, step in enumerate(data["step_list"]):
            add = True
            step_url = step["step_url"]
            param = analyze_url(step_url)
            if_url_exists = check_existing_url(step_url)
            print(param)
            if not (if_url_exists):
                for k,v in param.items():
                    if v['type'] == "static":
                        print('Skipping static url')
                        add = False
                if add:
                    #website = decide_website(data["task_name"])
                    website = extract_website_name(step_url)
                    url_name = extract_website_name(step_url)
                    dynamic_params = {k: v['value'] for k, v in param.items() if v['type'] == "dynamic"}
                    for param_name, param_value in dynamic_params.items():
                        embeddings = get_embedding(data["task_name"])
                        query = """
                                    INSERT INTO url_parameters (task, base_url,url_name,website, param_name, param_value,embeddings) VALUES (%s, %s,%s, %s, %s, %s,%s)
                                """
                        print(query)
                        cursor.execute(query,(data["task_name"], step_url,url_name,website,param_name,param_value,embeddings))
    conn.commit()  
    conn.close()

def add_url_to_chroma_db(res_file):
    with open(res_file) as file:
        lines = csv.reader(file)

        documents = []
        metadatas = []
        ids = []
        id = 1

        for i, line in enumerate(lines):
            param = analyze_url(line[1])
            url_name = extract_website_name(line[1])
            dynamic_params = {k: v['value'] for k, v in param.items() if v['type'] == "dynamic"}
            #print(f'Task :- {line[0]}, Url:- {line[1]}, Extracted params:- {param}, Url Name:- {url_name}')
            for param_name, param_value in dynamic_params.items():
                data = f'Task: {line[0]},base_url: {line[1]},url_name: {url_name},param_name: {param_name},param_value: {param_value}'
                documents.append(data)
                metadatas.append({"item_id": line[0]})
                ids.append(str(id))
                id+=1
    
    collection = chroma_client.get_or_create_collection(
        name="webcanvas_url_collection",
        metadata={
            "hnsw:space": "cosine",
            "created": str(datetime.now())
        })

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )




def gen_url(doc,task):
    prompt = f'You are provided with the base url and the parameters of the URL for a given task. {doc} Can you generate a url based on these information for {task} task? Provide just the generated url and not the explaination. The result should be in the json format with just the key value pair and no json appended at the start:- URL: generalized url'
    response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "assistant", "content": prompt}],
            temperature=0.0
        )
    message = response.choices[0].message.content
    cleaned_response = message.strip().strip('`').strip()
    #print(cleaned_response)
    res = json.loads(cleaned_response)
    if 'url' in message.lower():
        return res['URL']


