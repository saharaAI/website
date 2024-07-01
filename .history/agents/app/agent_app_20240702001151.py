import streamlit as st
import os
import re
from datetime import datetime
import json
from litellm import completion
import random
import streamlit as st
import os
import re
from datetime import datetime
import json
from litellm import completion
import random
import zipfile
import io

# Set environment variables for API keys
#st.set_page_config(layout='wide', page_title='AI Task Orchestrator', page_icon='🤖')
# Cacher les éléments de Streamlit
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
KEYS = ["AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU", "AIzaSyDbzt8ZGVd3P15MMuIUh8wz1lzT5jRLWlc"]
# Sidebar for API key inputs
st.sidebar.header("API Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
gemini_api_keys = random.choice(KEYS)
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

os.environ["GEMINI_API_KEY"] = random.choice(KEYS)

# Define the models
ORCHESTRATOR_MODEL = "gemini/gemini-1.5-flash-latest"
SUB_AGENT_MODEL = "gemini/gemini-1.5-flash-latest"
REFINER_MODEL = "gemini/gemini-1.5-pro-latest"

def gpt_orchestrator(objective, file_content=None, previous_results=None, use_search=False):
    st.write("Calling Orchestrator for your objective")
    previous_results_text = "\n".join(previous_results) if previous_results else "None"
    if file_content:
        st.code(file_content, language="python")
    
    messages = [
        {"role": "system", "content": "You are a detailed and meticulous assistant. Your primary goal is to break down complex objectives into manageable sub-tasks, provide thorough reasoning, and ensure code correctness. Always explain your thought process step-by-step and validate any code for errors, improvements, and adherence to best practices."},
        {"role": "user", "content": f"Based on the following objective{' and file content' if file_content else ''}, and the previous sub-task results (if any), please break down the objective into the next sub-task, and create a concise and detailed prompt for a subagent so it can execute that task. IMPORTANT!!! when dealing with code tasks make sure you check the code for errors and provide fixes and support as part of the next sub-task. If you find any bugs or have suggestions for better code, please include them in the next sub-task prompt. Please assess if the objective has been fully achieved. If the previous sub-task results comprehensively address all aspects of the objective, include the phrase 'The task is complete:' at the beginning of your response. If the objective is not yet fully achieved, break it down into the next sub-task and create a concise and detailed prompt for a subagent to execute that task.:\n\nObjective: {objective}" + ('\nFile content:\n' + file_content if file_content else '') + f"\n\nPrevious sub-task results:\n{previous_results_text}"}
    ]

    if use_search:
        messages.append({"role": "user", "content": "Please also generate a JSON object containing a single 'search_query' key, which represents a question that, when asked online, would yield important information for solving the subtask. The question should be specific and targeted to elicit the most relevant and helpful resources. Format your JSON like this, with no additional text before or after:\n{\"search_query\": \"<question>\"}\n"})

    response = completion(model=ORCHESTRATOR_MODEL, messages=messages)
    response_text = response['choices'][0]['message']['content']

    st.write("Orchestrator Response:")
    st.write(response_text)

    search_query = None
    if use_search:
        json_match = re.search(r'{.*}', response_text, re.DOTALL)
        if json_match:
            json_string = json_match.group()
            try:
                search_query = json.loads(json_string)["search_query"]
                st.write(f"Search Query: {search_query}")
                response_text = response_text.replace(json_string, "").strip()
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON: {e}")
                st.warning("Skipping search query extraction.")
        else:
            search_query = None

    return response_text, file_content, search_query

def gpt_sub_agent(prompt, search_query=None, previous_gpt_tasks=None, use_search=False, continuation=False):
    if previous_gpt_tasks is None:
        previous_gpt_tasks = []

    continuation_prompt = "Continuing from the previous answer, please complete the response."
    system_message = (
        "You are an expert assistant. Your goal is to execute tasks accurately, provide detailed explanations of your reasoning, "
        "and ensure the correctness and quality of any code. Always explain your thought process and validate your output thoroughly.\n\n"
        "Previous tasks:\n" + "\n".join(f"Task: {task['task']}\nResult: {task['result']}" for task in previous_gpt_tasks)
    )
    if continuation:
        prompt = continuation_prompt

    qna_response = None
    if search_query and use_search:
        st.write(f"Search query: {search_query}")
        st.warning("Search functionality is not implemented in this Streamlit version.")

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    if qna_response:
        messages.append({"role": "user", "content": f"\nSearch Results:\n{qna_response}"})

    response = completion(model=SUB_AGENT_MODEL, messages=messages)
    response_text = response['choices'][0]['message']['content']

    st.write("Sub-agent Result:")
    st.write(response_text)

    if len(response_text) >= 4000:
        st.warning("Output may be truncated. Attempting to continue the response.")
        continuation_response_text = gpt_sub_agent(prompt, search_query, previous_gpt_tasks, use_search, continuation=True)
        response_text += continuation_response_text

    return response_text
def anthropic_refine(objective, sub_task_results, filename, projectname, continuation=False):
    st.write("Calling Opus to provide the refined final output for your objective:")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Objective: " + objective + "\n\nSub-task results:\n" + "\n".join(sub_task_results) + "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. When working on code projects, ONLY AND ONLY IF THE PROJECT IS CLEARLY A CODING ONE please provide the following:\n1. Project Name: Create a concise and appropriate project name that fits the project based on what it's creating. The project name should be no more than 20 characters long.\n2. Folder Structure: Provide the folder structure as a valid JSON object, where each key represents a folder or file, and nested keys represent subfolders. Use null values for files. Ensure the JSON is properly formatted without any syntax errors. Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, separating items with commas as necessary.\nWrap the JSON object in <folder_structure> tags.\n3. Code Files: For each code file, include ONLY the file name NEVER EVER USE THE FILE PATH OR ANY OTHER FORMATTING YOU ONLY USE THE FOLLOWING format 'Filename: <filename>' followed by the code block enclosed in triple backticks, with the language identifier after the opening backticks, like this:\n\n```python\n<code>\n```"}
            ]
        }
    ]

    response = completion(model=REFINER_MODEL, messages=messages)
    response_text = response['choices'][0]['message']['content']
    
    st.write("Final Output:")
    st.write(response_text)

    if len(response_text) >= 4000 and not continuation:
        st.warning("Output may be truncated. Attempting to continue the response.")
        continuation_response_text = anthropic_refine(objective, sub_task_results + [response_text], filename, projectname, continuation=True)
        response_text += "\n" + continuation_response_text

    return response_text

def create_folder_structure(project_name, folder_structure, code_blocks):
    st.write(f"Creating project folder: {project_name}")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        create_folders_and_files_recursive(zip_file, project_name, folder_structure, code_blocks)
    return zip_buffer

def create_folders_and_files_recursive(zip_file, current_path, structure, code_blocks):
    for key, value in structure.items():
        path = os.path.join(current_path, key)
        if isinstance(value, dict):
            st.write(f"Created folder: {path}")
            create_folders_and_files_recursive(zip_file, path, value, code_blocks)
        else:
            code_content = next((code for file, code in code_blocks if file == key), None)
            if code_content:
                st.write(f"Created file: {path}")
                st.code(code_content, language="python")
                zip_file.writestr(path, code_content)
            else:
                st.warning(f"Code content not found for file: {key}")

def main():
    st.title("🤖 AI Task Orchestrator - Sahara Analytics")

    objective = st.text_area("Enter your objective:")
    file_content = st.text_area("Enter file content (optional):")
    use_search = st.checkbox("Use search")

    if st.button("Start Task"):
        task_exchanges = []
        gpt_tasks = []

        with st.spinner("Processing your task..."):
            while True:
                previous_results = [result for _, result in task_exchanges]
                if not task_exchanges:
                    gpt_result, file_content_for_gpt, search_query = gpt_orchestrator(objective, file_content, previous_results, use_search)
                else:
                    gpt_result, _, search_query = gpt_orchestrator(objective, previous_results=previous_results, use_search=use_search)

                if "The task is complete:" in gpt_result:
                    final_output = gpt_result.replace("The task is complete:", "").strip()
                    break
                else:
                    sub_task_prompt = gpt_result
                    if file_content_for_gpt and not gpt_tasks:
                        sub_task_prompt = f"{sub_task_prompt}\n\nFile content:\n{file_content_for_gpt}"
                    sub_task_result = gpt_sub_agent(sub_task_prompt, search_query, gpt_tasks, use_search)
                    gpt_tasks.append({"task": sub_task_prompt, "result": sub_task_result})
                    task_exchanges.append((sub_task_prompt, sub_task_result))
                    file_content_for_gpt = None

            sub_task_results = [f"Orchestrator Prompt: {prompt}\nSub-agent Result: {result}" for prompt, result in task_exchanges]

            sanitized_objective = re.sub(r'\W+', '_', objective)
            timestamp = datetime.now().strftime("%H-%M-%S")
            refined_output = anthropic_refine(objective, sub_task_results, timestamp, sanitized_objective)

            project_name_match = re.search(r'Project Name: (.*)', refined_output)
            project_name = project_name_match.group(1).strip() if project_name_match else sanitized_objective

            folder_structure_match = re.search(r'<folder_structure>(.*?)</folder_structure>', refined_output, re.DOTALL)
            folder_structure = {}
            if folder_structure_match:
                json_string = folder_structure_match.group(1).strip()
                try:
                    folder_structure = json.loads(json_string)
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing JSON: {e}")
                    st.error(f"Invalid JSON string: {json_string}")

            code_blocks = re.findall(r'Filename: (\S+)\s*```[\w]*\n(.*?)\n```', refined_output, re.DOTALL)
            zip_buffer = create_folder_structure(project_name, folder_structure, code_blocks)

        st.success("Task completed successfully!")

        # Offer the zip file for download
        st.download_button(
            label="Download Project Files",
            data=zip_buffer.getvalue(),
            file_name=f"{project_name}.zip",
            mime="application/zip"
        )

        # Display the full refined output
        st.subheader("Refined Output")
        st.text_area("", value=refined_output, height=300)

if __name__ == "__main__":
    main()
