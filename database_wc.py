import ast
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import json
from openai import OpenAI
import os
from evaluate.rag_query import decide_website,get_embedding_for_db,get_embedding
import hashlib
from logs import logger

client = OpenAI()

def get_db_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="webagent",
        host="localhost",
        port="5433"
    )


def insert_to_db(data):
    data = {k: (None if v is None else v) for k, v in data.items()}
    conn = get_db_connection()
    print('Connected to DB')
    try:
        with conn.cursor() as cur:
            for i, step in enumerate(data["step_list"]):
                status = 'start'
                task_name = data["task_name"]
                #thought = optimize_thought(step["dict_result"]["description"]["thought"])
                thought = step["dict_result"]["description"]["thought"]
                action_type = step["dict_result"]["action_type"]
                action_value = step["dict_result"]["value"]
                step_url = step["step_url"]
                dom_hash = hashlib.sha256(step["observation"].encode('utf-8')).hexdigest()
                if i > 0:
                    cur_score = step["score"]
                    prev_step = data["step_list"][i - 1]
                    prev_score = prev_step["score"]
                    cur_score = cur_score.split('/')
                    cur_val = int(cur_score[0])
                    prev_score =  prev_score.split('/')
                    prev_val =  int(prev_score[0])
                    if cur_val > prev_val:
                        status = 'success'
                    else:
                        status = 'fail'
                embeddings = get_embedding_for_db(task_name, thought, status)
                query = """
                    INSERT INTO webcanvas (task,base_url, thought,  status,action_type,action_value,embedding,dom_hash)
                    VALUES (%s, %s, %s, %s, %s, %s,%s,%s) ON CONFLICT (thought) DO NOTHING;
                """
                #print(query, (task_name, step_url, thought, status,action_type,action_value,data,task_name,step_url))
                cur.execute(query, (task_name, step_url, thought, status,action_type,action_value,embeddings,dom_hash))
                #print(f"Inserted step with URL: {step_url}")

            conn.commit()  
            print("Data inserted successfully!")

    except Exception as e:
        print("Error occurred:", e)
        conn.rollback()

    finally:
        conn.close()


def insert_prev_trace_to_db(data):
    data = {k: (None if v is None else v) for k, v in data.items()}
    conn = get_db_connection()
    print('Connected to DB')
    try:
        with conn.cursor() as cur:
            for i, step in enumerate(data["step_list"]):
                status = 'start'
                task_name = data["task_name"]
                len_list = len(data["step_list"])
                prev_trace = data["step_list"][len_list-1]
                if i > 0:
                    cur_score = step["score"]
                    prev_step = data["step_list"][i - 1]
                    prev_score = prev_step["score"]
                    cur_score = cur_score.split('/')
                    cur_val = int(cur_score[0])
                    prev_score =  prev_score.split('/')
                    prev_val =  int(prev_score[0])
                    if cur_val > prev_val:
                        status = 'success'
                    else:
                        status = 'fail'
                data = ast.literal_eval(prev_trace["previous_trace"])
                for th in data:
                    thought = th['thought']
                    action = th['action']
                    query = """
                        INSERT INTO webcanvas_prev (task, thought, action, status)
                        VALUES (%s, %s, %s, %s);
                    """
                    cur.execute(query, (task_name, thought, action,status))
                break

            conn.commit()  
            print("Data inserted successfully!")

    except Exception as e:
        print("Error occurred:", e)
        conn.rollback()

    finally:
        conn.close()

def insert_db_hash(task,observation):
    conn = get_db_connection()
    print('Connected to DB')
    dom_hash = hashlib.sha256(observation.encode('utf-8')).hexdigest()
    try:
        cur = conn.cursor()
        query = """
                        INSERT INTO webcanvas (task,dom_hash)
                        SELECT %s, %s;
                    """
        cur.execute(query, (task, dom_hash))
        print(f"Inserted in dom_hash table with task: {task}")

        conn.commit()  
        print("Data inserted successfully!")
    except Exception as e:
        print("Error occurred:", e)
        conn.rollback()


def optimize_thought(th):
    prompt = f"This is a thought process to solve a task :- {th}. Can you consie it as such that no important context is lost?"
    response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "assistant", "content": prompt}],
            temperature=0.0
        )
    message = response.choices[0].message.content
    return message

#print(optimize_thought("To compare available plans for the AeroAPI on FlightAware, I need to visit the FlightAware website and look for the section that details the AeroAPI plans."))
def query_db_emb(task):
    logger.info("----Querring DB----")
    conn = get_db_connection()
    cur = conn.cursor()
    print('Connected to DB')

    new_em = np.array(get_embedding(task), dtype=np.float32)

    cur.execute("""
        SELECT thought
        FROM webcanvas
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
    """, (new_em.tolist(),))

    # Fetch all results
    results = cur.fetchall()
    #print("Closest match:", results)
    cur.close()
    conn.close()
    if not results:
        return False
    # Print the results
    else:
        return results

print(query_db_emb('Show list of popular businesses in Cleveland on yellowpages.') )
def query_db(task):
    logger.info("----Querring DB----")
    conn = get_db_connection()
    cur = conn.cursor()
    print('Connected to DB')


    query = f"SELECT thought FROM webcanvas WHERE task = %s and status = 'success';"

    cur.execute(query, (task,))

    # Fetch all results
    results = cur.fetchall()

    cur.close()
    conn.close()
    if not results:
        return False
    # Print the results
    else:
        return results
    

def query_db_website(website):

    conn = get_db_connection()
    cur = conn.cursor()
    print('Connected to DB')


    query = f"select * from webcanvas where task like '%{website}%';"

    cur.execute(query)
    print('Query to DB:- ', query)
    # Fetch all results
    results = cur.fetchall()

    cur.close()
    conn.close()
    if not results:
        return False
    # Print the results
    else:
        return results
    
def query_db_url(task):

    conn = get_db_connection()
    cur = conn.cursor()
    print('Connected to DB')
    #website = decide_website(task)

    new_em = get_embedding(task)

    cur.execute("""
        SELECT base_url,url_name,website, param_name, param_value
        FROM url_parameters
        ORDER BY embeddings <=> %s::vector
        LIMIT 3;
    """, (new_em,))

    # Fetch all results
    results = cur.fetchall()

    cur.close()
    conn.close()
    if not results:
        return False
    # Print the results
    else:
        return results

def check_existing_url(url):

    conn = get_db_connection()
    cur = conn.cursor()
    print('Connected to DB')

    cur.execute("""
        SELECT *
        FROM url_parameters
        WHERE base_url=%s ;
    """, (url,))

    # Fetch all results
    results = cur.fetchall()

    cur.close()
    conn.close()
    if not results:
        return False
    # Print the results
    else:
        return results


def query_db_dom_hash(dom_hash,task):

    conn = get_db_connection()
    cur = conn.cursor()
    print('Connected to DB')

    cur.execute("""
        SELECT task, thought,  status,action_type,action_value
        FROM webcanvas 
        WHERE dom_hash = %s and task = %s;
    """, (dom_hash,task,))

    # Fetch all results
    results = cur.fetchall()

    cur.close()
    conn.close()
    if not results:
        return False
    # Print the results
    else:
        return results

#conversation = json.load(open('/Users/smritisharma/Downloads/Thesis/WebCanvas/batch_tasks_results/example/json_result/6_25.json', 'r'))
#nsert_to_db(conversation)
#dbmemory = query_db("Compare available plans for the AeroAPI on flightaware")
#print(dbmemory)
def format_prompt(dbmemory):
    #prompt = 'The following are history of tasks you have performed on the website. There also the thought process and observations you had while solving these tasks. Please use them to efficiently solve the new task. The momory provided is in a dictionary format. The key of the dictionary is the task and the values are the observation for the task.\n'
    prompt ='Data '
    dic= {}
    for item in dbmemory:
        cur_task = item[-2]
        if cur_task in dic:
            dic[cur_task].append(item[-1])
        else:
            dic[cur_task] = [item[-1]]
    final_promt = prompt + str(dic)
    data = []
    for task_name, step_url, thought, status in dbmemory:
        data.append(f"Task: {task_name},URL visited: {step_url},Thought: {thought},Status: {status}")
    print(data)

def fetch_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM webcanvas;")
    rows = cursor.fetchall()  
    cursor.close()
    conn.close()
    return rows

#format_prompt(dbmemory)
#data={"task_name": "Compare available plans for the AeroAPI on flightaware", "id": 1, "reference_task_length": 4, "status": "finished", "LLM_error_rate": "0.0", "step_list": [{"step_index": 0, "dict_result": {"id": "", "action_type": "goto", "value": "https://flightaware.com/", "description": {"thought": "To compare the available plans for AeroAPI on FlightAware, I need to visit the FlightAware website to access the relevant information.", "action": "I'm navigating to the FlightAware website to find information about the AeroAPI plans."}, "error_message": "", "planning_token_count": [1135, 79]}, "current_trace": {"thought": "To compare the available plans for AeroAPI on FlightAware, I need to visit the FlightAware website to access the relevant information.", "action": "I'm navigating to the FlightAware website to find information about the AeroAPI plans.", "reflection": ""}, "selector": None, "execute_action": {"action_type": 2, "element_id": 0, "url": "https://flightaware.com/", "fill_text": "", "element_name": ""}, "element_value": "", "text_content": "", "score": "0 / 3", "match_func_result": [], "step_reward": {}, "step_url": "https://www.flightaware.com/", "error_message": "", "previous_trace": "[{'thought': 'To compare the available plans for AeroAPI on FlightAware, I need to visit the FlightAware website to access the relevant information.', 'action': \"I'm navigating to the FlightAware website to find information about the AeroAPI plans.\", 'reflection': ''}]"}, {"step_index": 1, "dict_result": {"id": 3, "action_type": "click", "value": "link 'AeroAPI'", "description": {"thought": "To compare the available plans for AeroAPI, I need to click on the AeroAPI link to access the relevant details about the plans.", "action": "Now I'm on the FlightAware website. I'm clicking the link with element_id [3] to access information about the AeroAPI plans."}, "error_message": "", "planning_token_count": [3022, 97]}, "current_trace": {"thought": "To compare the available plans for AeroAPI, I need to click on the AeroAPI link to access the relevant details about the plans.", "action": "Now I'm on the FlightAware website. I'm clicking the link with element_id [3] to access information about the AeroAPI plans.", "reflection": ""}, "selector": "#header-nav-list > li:nth-child(1) > details > ul > li > div.bg-primary.flex.flex-row.gap-10.px-10.pt-5 > nav:nth-child(1) > ul > li:nth-child(1) > a", "execute_action": {"action_type": 1, "element_id": 1012, "url": "", "fill_text": "", "element_name": "link"}, "element_value": "AeroAPI", "text_content": "", "score": "1 / 3", "match_func_result": [{"url_included_match": "flightaware."}], "step_reward": {}, "step_url": "https://www.flightaware.com/commercial/aeroapi/", "error_message": "", "previous_trace": "[{'thought': 'To compare the available plans for AeroAPI on FlightAware, I need to visit the FlightAware website to access the relevant information.', 'action': \"I'm navigating to the FlightAware website to find information about the AeroAPI plans.\", 'reflection': ''}, {'thought': 'To compare the available plans for AeroAPI, I need to click on the AeroAPI link to access the relevant details about the plans.', 'action': \"Now I'm on the FlightAware website. I'm clicking the link with element_id [3] to access information about the AeroAPI plans.\", 'reflection': ''}]"}, {"step_index": 2, "dict_result": {"id": 247, "action_type": "click", "value": "link 'Compare Tiers'", "description": {"thought": "To compare the available plans for the AeroAPI on FlightAware, I need to click on the 'Compare Tiers' link to access the detailed information about the different plans.", "action": "Now I'm on the FlightAware AeroAPI page. I'm clicking the link with element_id [247] to compare the available plans for AeroAPI."}, "error_message": "", "planning_token_count": [5269, 107]}, "current_trace": {"thought": "To compare the available plans for the AeroAPI on FlightAware, I need to click on the 'Compare Tiers' link to access the detailed information about the different plans.", "action": "Now I'm on the FlightAware AeroAPI page. I'm clicking the link with element_id [247] to compare the available plans for AeroAPI.", "reflection": ""}, "selector": "#per-query-fee-section > div > div.button-container > a", "execute_action": {"action_type": 1, "element_id": 451, "url": "", "fill_text": "", "element_name": "link"}, "element_value": "Compare Tiers", "text_content": "", "score": "2 / 3", "match_func_result": [{"url_included_match": "flightaware."}, {"url_included_match": "/aeroap"}], "step_reward": {}, "step_url": "https://www.flightaware.com/commercial/aeroapi/#compare-plans-section", "error_message": "", "previous_trace": "[{'thought': 'To compare the available plans for AeroAPI on FlightAware, I need to visit the FlightAware website to access the relevant information.', 'action': \"I'm navigating to the FlightAware website to find information about the AeroAPI plans.\", 'reflection': ''}, {'thought': 'To compare the available plans for AeroAPI, I need to click on the AeroAPI link to access the relevant details about the plans.', 'action': \"Now I'm on the FlightAware website. I'm clicking the link with element_id [3] to access information about the AeroAPI plans.\", 'reflection': ''}, {'thought': \"To compare the available plans for the AeroAPI on FlightAware, I need to click on the 'Compare Tiers' link to access the detailed information about the different plans.\", 'action': \"Now I'm on the FlightAware AeroAPI page. I'm clicking the link with element_id [247] to compare the available plans for AeroAPI.\", 'reflection': ''}]"}, {"step_index": 3, "dict_result": {"id": 247, "action_type": "click", "value": "link 'Compare Tiers'", "description": {"thought": "I have already accessed the page for comparing the available plans for AeroAPI on FlightAware. Now I need to click on the 'Compare Tiers' link to view the details of the different plans.", "action": "I'm on the AeroAPI page on FlightAware, and I'm clicking the link with element_id [247] to view the different plans available for AeroAPI."}, "error_message": "", "planning_token_count": [5398, 114]}, "current_trace": {"thought": "I have already accessed the page for comparing the available plans for AeroAPI on FlightAware. Now I need to click on the 'Compare Tiers' link to view the details of the different plans.", "action": "I'm on the AeroAPI page on FlightAware, and I'm clicking the link with element_id [247] to view the different plans available for AeroAPI.", "reflection": ""}, "selector": "#per-query-fee-section > div > div.button-container > a", "execute_action": {"action_type": 1, "element_id": 451, "url": "", "fill_text": "", "element_name": "link"}, "element_value": "Compare Tiers", "text_content": "", "score": "3 / 3", "match_func_result": [{"url_included_match": "flightaware."}, {"url_included_match": "/aeroap"}, {"url_included_match": "#compare-plans-section"}], "step_reward": {}, "step_url": "https://www.flightaware.com/commercial/aeroapi/#compare-plans-section", "error_message": "", "previous_trace": "[{'thought': 'To compare the available plans for AeroAPI on FlightAware, I need to visit the FlightAware website to access the relevant information.', 'action': \"I'm navigating to the FlightAware website to find information about the AeroAPI plans.\", 'reflection': ''}, {'thought': 'To compare the available plans for AeroAPI, I need to click on the AeroAPI link to access the relevant details about the plans.', 'action': \"Now I'm on the FlightAware website. I'm clicking the link with element_id [3] to access information about the AeroAPI plans.\", 'reflection': ''}, {'thought': \"To compare the available plans for the AeroAPI on FlightAware, I need to click on the 'Compare Tiers' link to access the detailed information about the different plans.\", 'action': \"Now I'm on the FlightAware AeroAPI page. I'm clicking the link with element_id [247] to compare the available plans for AeroAPI.\", 'reflection': ''}, {'thought': \"I have already accessed the page for comparing the available plans for AeroAPI on FlightAware. Now I need to click on the 'Compare Tiers' link to view the details of the different plans.\", 'action': \"I'm on the AeroAPI page on FlightAware, and I'm clicking the link with element_id [247] to view the different plans available for AeroAPI.\", 'reflection': ''}]"}], "evaluate_steps": [{"match_function": "url_included_match", "key": "", "reference_answer": "flightaware.", "score": 1}, {"match_function": "url_included_match", "key": "", "reference_answer": "/aeroap", "score": 1}, {"match_function": "url_included_match", "key": "", "reference_answer": "#compare-plans-section", "score": 1}]}
#f = open('batch_tasks_results/example/json_result/0_40.json')
#apple_dta= json.load(f)
#insert_to_db(apple_dta)
#print(query_db("Compare available plans for the AeroAPI on flightaware",'webcanvas'))
