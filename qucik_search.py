import ast
import csv
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import json
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer,util,CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch
#from evaluate.rag_query import decide_website,get_embedding_for_db,get_embedding

#from logs import logger

os.environ["OPENAI_API_KEY"] =''
client = OpenAI()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def get_db_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="webagent",
        host="localhost",
        port="5433"
    )


def add_path_to_db(task,path):
    conn = get_db_connection()
    print('Connected to DB')
    try:
        with conn.cursor() as cur:
            query = "select * from quick_search where task =%s"
            cur.execute(query, (task,))
            results = cur.fetchall()
            if not results:
                if(isinstance(path,list)):                
                    for paths in path:
                        query = """
                                INSERT INTO quick_search (task, probable_path)
                                VALUES (%s, %s);
                            """
                        print(paths)
                        cur.execute(query, (task,paths))
                    conn.commit()  
                    print("Data inserted successfully!")  
                else:
                    print('Path type :- ',type(path))

    except Exception as e:
        print("Error occurred:", e)
        conn.rollback()

    finally:
        conn.close()


def decide_paths(task):
    prompt = 'Based on the given task, can you create short paths in which the tasks can be solved. Dont keep the paths very consise, repeat the next path with sompe part of the previous path for context. Add path number before each path denoting the order of the path. Replace product/items to be searched with "product" keyword instaed of actual name of the item. For example:- for a task: "Rate the movie Thor on IMDB", you can divide the task into following paths:- 1. Search for IMDB website. 2. After searching for the IMDB website, search for the movie on IMDB. 3. After searching for the movie, rate the movie on IMDB and so on. Here, we replaced the movie name with "the movie". example 2: for task "Find blue Jacket on Amazon" you can divide the task into following paths:- 1. Search for Amazon website. 2. After searching for amzon website,find product on the website. Please output only an array of such paths and no other pretext or posttext for the array. Output format:- ["1.path1","2.path2"...]'
    response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "assistant", "content": prompt+task}],
            temperature=0.0
        )
    print('LLM Called ----')
    message = response.choices[0].message.content
    cleaned_response = message.strip().strip('`').strip()
    cleaned_response = ast.literal_eval(cleaned_response)
    return cleaned_response

def check_path(data):
    conn = get_db_connection()
    print('Connected to DB')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with conn.cursor() as cur:
        for i,line in enumerate(data):
            task = line[0]
            th = line[2]
            query = "select * from quick_search where task =%s"
            cur.execute(query, (task,))
            results = cur.fetchall()
            if not results:
                return False
            else:
                query = "select probable_path from quick_search where task =%s"
                cur.execute(query, (task,))
                p_path = cur.fetchall()
                steps = [rows[0] for rows in p_path]
                if p_path:
                    step_embeddings = model.encode(steps, convert_to_tensor=True)
                    thought_embedding = model.encode(th, convert_to_tensor=True)
                    scores = util.cos_sim(thought_embedding, step_embeddings)[0]
                    k = min(3,scores.shape[0])
                    top_in = torch.topk(torch.tensor(scores),k).indices.tolist()
                    candidate = [steps[i] for i in top_in]
                    pairs= [(th,steps) for steps in candidate]
                    rerank = reranker.predict(pairs)
                    rernkd = sorted(zip(candidate,rerank),key=lambda x: x[1],reverse=True)
                    finl_step = rernkd[:1]
                    best_step = finl_step[0][0]
                    query = """
                                INSERT INTO quick_search (task,probable_path,actual_th,base_url,status,action_type,action_value)
                                VALUES (%s,%s,%s,%s,%s,%s,%s);
                                """
                    cur.execute(query, (task,best_step,th,line[1],line[3],line[4],line[5]))
                    conn.commit() 
                    print('Data Inserted')

def check_path_cos(data):
    conn = get_db_connection()
    print('Connected to DB')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with conn.cursor() as cur:
        for i,line in enumerate(data):
            task = line[0]
            th = line[2]
            query = "select * from quick_search_cos where task =%s"
            cur.execute(query, (task,))
            results = cur.fetchall()
            if not results:
                return False
            else:
                query = "select probable_path from quick_search_cos where task =%s"
                cur.execute(query, (task,))
                p_path = cur.fetchall()
                steps = [rows[0] for rows in p_path]
                if p_path:
                    step_embeddings = model.encode(steps, convert_to_tensor=True)
                    thought_embedding = model.encode(th, convert_to_tensor=True)
                    scores = util.cos_sim(thought_embedding, step_embeddings)[0]
                    best_idx = torch.argmax(scores).item()
                    best_score = scores[best_idx].item()
                    best_step = steps[best_idx]
                    query = """
                                INSERT INTO quick_search_cos (task,probable_path,actual_th,base_url,status,action_type,action_value)
                                VALUES (%s,%s,%s,%s,%s,%s,%s);
                                """
                    cur.execute(query, (task,best_step,th,line[1],line[3],line[4],line[5]))
                    conn.commit()  

def match_tasks(task):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    encoded_task = model.encode(task, convert_to_tensor=True)
    query = "select task from quick_search where task =%s"
    cur.execute(query, (task,))
    p_path = cur.fetchall()


task='Compare all membership tier benefits in qatarairways'
#task = 'Find the highest-rated mover in Honolulu to shift a vehicle and large appliances and who has virtual discussion options available in yelp'
#path =decide_paths(task)
#add_path_to_db(task,path)
th=['Since I have already clicked the filter button to sort by lowest price, I need to check the results to find the spider-man toys sorted by price.','I need to click the filter button to sort the spider-man toys by the lowest price.','Now I need to sort the results by lowest price to find the best deals on spider-man toys for kids.','Now I need to submit the search for spider-man toys for kids to see the results.','I need to search for spider-man toys for kids on the Kohls website.','Now, I need to go directly to the Kohl''s website to search for spider-man toys for kids.','Now I need to input "spider-man toys for kids" into the search bar to find the relevant products on the Kohls website.']
'''
with open('evaluate/complete_training_data.csv') as file:
    lines = csv.reader(file)
    check_path(lines)
'''
    
def add_path_from_csv(file_path): #'evaluate/training_task.csv'
    with open(file_path) as file:
        lines = csv.reader(file)
        for i,line in enumerate(lines):
            task = line[0]
            path =decide_paths(task)
            add_path_to_db(task,path)

def prep_data(task):
    conn = get_db_connection()
    print('Connected to DB')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    data=[]
    with conn.cursor() as cur:
        query = "select task from quick_search where task =%s"
        cur.execute(query, (task,))
        results = cur.fetchall()
        if not results:
            return False
        else:
            query = "select distinct probable_path from quick_search where task =%s order by probable_path asc"
            cur.execute(query, (results[0][0],))
            path_results = cur.fetchall()
            if not results:
                return False
            else:
                for path in path_results:
                    query = "select actual_th,base_url,status,action_type,action_value from quick_search where task =%s and probable_path = %s"
                    cur.execute(query, (results[0][0],path[0],))
                    path_results = cur.fetchall()
                    if not path_results:
                        continue
                    else:
                        data.append(path[0] + 'History of thoughts and action taken under this path:-'+str(path_results))
                return data

def check_path_llm(task,prev_trace,data):
    prompt = f'''You are an assistant who not only helps to browse and operate web pages to achieve certain goals,
      but also needs to explore the information on the page to answer the questions raised by the target task. 
      You are given a task and the steps needed to solve the task in order. These are the steps previously used by you 
      to solve the task, followed by your thought process and the url you visited. Based on the information,
      select the step number you can directly jump to solve the task.
      Return an integer number denoting the step number or null if you think you can not jump to any of the steps. 
      Task :- {task},
      Steps with corresponding thought and url:- {data},
      Here is what you did in the previous steps:- {prev_trace}
        Please ensure the accuracy of your output, as we will execute subsequent steps based on the integer you provide.
        
        **Output Requirements**:
        - Ensure your output strictly only contain step number or null
      '''
    prompt_n = f'''
        You are assisting a web agent that is solving a task by following a sequence of steps.
        Each step represents a meaningful milestone in completing the task.

        You are given:
        1. The task description
        2. A list of previously successful steps for this task, each with:
        - step number
        - intent / thought
        - action type
        - URL visited
        3. The agent's current execution trace

        Your goal is to determine whether the agent can safely SKIP ahead and resume execution
        from one of the listed steps.

        A step is a valid jump target if:
        - The agent's current intent matches or subsumes the intent of the step
        - AND the step does not depend on missing prior information
        - AND the agent can reasonably continue execution from that step without loss of correctness

        You should prefer selecting a step when there is a clear match.
        Return null ONLY if no step is a reasonable continuation point.

        Task:
        {task}

        Previously successful steps:
        {data}

        Current execution trace:
        {prev_trace}

        Decision rules:
        - Return the step number (integer) of the BEST step to jump to
        - If multiple steps are possible, choose the earliest valid step
        - Return null only if none of the steps can be safely reused

        Output format (strict):
        - Output ONLY a single integer (e.g., 3) OR null
        '''
    print('Step Prompt:----',prompt_n)

    response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "assistant", "content": prompt_n}]
        )
    print('LLM Called ----')
    message = response.choices[0].message.content
    cleaned_response = message.strip().strip('`').strip()
    #cleaned_response = ast.literal_eval(cleaned_response)
    return cleaned_response

def add_old_data_to_qs(file_path): #'evaluate/training_task.csv'
    with open(file_path) as file:
        lines = csv.reader(file)
        check_path(lines)

def extract_step_info(task,step_num):
    conn = get_db_connection()
    url = []
    print('Connected to DB, extracting step info')
    with conn.cursor() as cur:
        query = f"select * from quick_search where task ='{task}' and probable_path like '%{step_num}%'"
        cur.execute(query)
        results = cur.fetchall()
    for items in results:
        i = 0
        for val in items:
            if(i==3):
                url.append(val)
            i = i+1
    return results,url

#add_old_data_to_qs('evaluate/complete_training_data.csv')
#prep_data function will return path and thoughts, send the data to check_path_llm function and verify the flow
#data = prep_data('Show me the schedule for the orange line in mbta')
#print(check_path_llm(task,data))