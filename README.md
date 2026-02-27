
## Getting Started


### Commands on to run locally

1. Activate conda environment.
 ```sh
    conda activate webcanvas
   ```
2. Add your Google CX keys. [How to set up google search](https://developers.google.com/custom-search/v1/overview?hl=zh-cn).
   ```sh
    export GOOGLE_API_KEY=your_api_key
    export GOOGLE_CX=your_custom_search_engine_id
   ```
3. Add your OpenAPI keys
   ```sh
   export OPEN_API_KEY=your_api_key
   ```
4. Update dataset in settings.toml file
     Use the train and test tasks from repo.
5. Run evaluate.
   ```sh
   python evaluate.py
   ```
   
### DB setup

1. Locally pgAdmin 4 application can be used to view data
     webcanvas table has all the details for each task
     url_parameters table is used by generalize_url file to create new URLs
     quick_search table has step data for each tasks. It is taken from webcanvas table + the steps data
2. For new setup, install postgresql and pgvector and update login prarameters on database_wc.py, generalize_url.py file.
### For new setup

1. Clone WebCanvas implementation and complete its setup
     https://github.com/iMeanAI/WebCanvas/blob/main/README.md
2. Replace planning.py file under Plan folder with the planning_updated.py file
3. Replace prompt_constructor.py file under Prompt folder with the prompt_constructor_updated.py file
4. Add quick_search.py to the Prompt folder
5. Add database_wc.py and generalize_url.py under evaluate folder.
6. Run the local commands as mentioned above.
