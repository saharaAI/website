import os
from datetime import datetime
import json
from colorama import init, Fore, Style
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
from tavily import TavilyClient
import pygments.util
import base64
from PIL import Image
import io
import re
from anthropic import Anthropic
import google.generativeai as genai

class MultimodalAIChat:
    def __init__(self, anthropic_api_key, tavily_api_key, gemini_api_key):
        # Initialize colorama
        init()

        # Color constants
        self.USER_COLOR = Fore.WHITE
        self.AI_COLOR = Fore.BLUE
        self.TOOL_COLOR = Fore.YELLOW
        self.RESULT_COLOR = Fore.GREEN

        # Constants
        self.CONTINUATION_EXIT_PHRASE = "AUTOMODE_COMPLETE"
        self.MAX_CONTINUATION_ITERATIONS = 25

        # Initialize clients
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        self.tavily = TavilyClient(api_key=tavily_api_key)
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')

        # Set up the conversation memory
        self.conversation_history = []

        # automode flag
        self.automode = False

        # Default model
        self.default_model = "gemini"

        # System prompt (for Claude, can be adapted for Gemini)
        self.system_prompt = """
        You are an AI assistant powered by advanced language models. You are an exceptional software developer with vast knowledge across multiple programming languages, frameworks, and best practices. Your capabilities include:

        1. Creating project structures, including folders and files
        2. Writing clean, efficient, and well-documented code
        3. Debugging complex issues and providing detailed explanations
        4. Offering architectural insights and design patterns
        5. Staying up-to-date with the latest technologies and industry trends
        6. Reading and analyzing existing files in the project directory
        7. Listing files in the root directory of the project
        8. Performing web searches to get up-to-date information or additional context
        9. When you use search make sure you use the best query to get the most accurate and up-to-date information
        10. IMPORTANT!! You NEVER remove existing code if it doesn't require to be changed or removed, never use comments like # ... (keep existing code) ... or # ... (rest of the code) ... etc, you only add new code or remove it or EDIT IT.
        11. Analyzing images provided by the user
        When an image is provided, carefully analyze its contents and incorporate your observations into your responses.

        When asked to create a project:
        - Always start by creating a root folder for the project.
        - Then, create the necessary subdirectories and files within that root folder.
        - Organize the project structure logically and follow best practices for the specific type of project being created.
        - Use the provided tools to create folders and files as needed.

        When asked to make edits or improvements:
        - Use the read_file tool to examine the contents of existing files.
        - Analyze the code and suggest improvements or make necessary edits.
        - Use the write_to_file tool to implement changes.

        Be sure to consider the type of project (e.g., Python, JavaScript, web application) when determining the appropriate structure and files to include.

        You can now read files, list the contents of the root folder where this script is being run, and perform web searches. Use these capabilities when:
        - The user asks for edits or improvements to existing files
        - You need to understand the current state of the project
        - You believe reading a file or listing directory contents will be beneficial to accomplish the user's goal
        - You need up-to-date information or additional context to answer a question accurately

        When you need current information or feel that a search could provide a better answer, use the tavily_search tool. This tool performs a web search and returns a concise answer along with relevant sources.

        Always strive to provide the most accurate, helpful, and detailed responses possible. If you're unsure about something, admit it and consider using the search tool to find the most current information.

        {automode_status}

        When in automode:
        1. Set clear, achievable goals for yourself based on the user's request
        2. Work through these goals one by one, using the available tools as needed
        3. REMEMBER!! You can Read files, write code, LIST the files, and even SEARCH and make edits, use these tools as necessary to accomplish each goal
        4. ALWAYS READ A FILE BEFORE EDITING IT IF YOU ARE MISSING CONTENT. Provide regular updates on your progress
        5. IMPORTANT RULE!! When you know your goals are completed, DO NOT CONTINUE IN POINTLESS BACK AND FORTH CONVERSATIONS with yourself, if you think we achieved the results established to the original request say "AUTOMODE_COMPLETE" in your response to exit the loop!
        6. ULTRA IMPORTANT! You have access to this {iteration_info} amount of iterations you have left to complete the request, you can use this information to make decisions and to provide updates on your progress knowing the amount of responses you have left to complete the request.
        Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within <thinking></thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.
        """

        self.tools = [
            {
                "name": "create_folder",
                "description": "Create a new folder at the specified path. Use this when you need to create a new directory in the project structure.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path where the folder should be created"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "create_file",
                "description": "Create a new file at the specified path with optional content. Use this when you need to create a new file in the project structure.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path where the file should be created"
                        },
                        "content": {
                            "type": "string",
                            "description": "The initial content of the file (optional)"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_to_file",
                "description": "Write content to an existing file at the specified path. Use this when you need to add or update content in an existing file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "read_file",
                "description": "Read the contents of a file at the specified path. Use this when you need to examine the contents of an existing file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path of the file to read"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "list_files",
                "description": "List all files and directories in the root folder where the script is running. Use this when you need to see the contents of the current directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path of the folder to list (default: current directory)"
                        }
                    }
                }
            },
            {
                "name": "tavily_search",
                "description": "Perform a web search using Tavily API to get up-to-date information or additional context. Use this when you need current information or feel a search could provide a better answer.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def update_system_prompt(self, current_iteration=None, max_iterations=None):
        automode_status = "You are currently in automode." if self.automode else "You are not in automode."
        iteration_info = ""
        if current_iteration is not None and max_iterations is not None:
            iteration_info = f"You are currently on iteration {current_iteration} out of {max_iterations} in automode."
        return self.system_prompt.format(automode_status=automode_status, iteration_info=iteration_info)

    def print_colored(self, text, color):
        print(f"{color}{text}{Style.RESET_ALL}")

    def print_code(self, code, language):
        try:
            lexer = get_lexer_by_name(language, stripall=True)
            formatted_code = highlight(code, lexer, TerminalFormatter())
            print(formatted_code)
        except pygments.util.ClassNotFound:
            self.print_colored(f"Code (language: {language}):\n{code}", self.AI_COLOR)

    def create_folder(self, path):
        try:
            os.makedirs(path, exist_ok=True)
            return f"Folder created: {path}"
        except Exception as e:
            return f"Error creating folder: {str(e)}"

    def create_file(self, path, content=""):
        try:
            with open(path, 'w') as f:
                f.write(content)
            return f"File created: {path}"
        except Exception as e:
            return f"Error creating file: {str(e)}"

    def write_to_file(self, path, content):
        try:
            with open(path, 'w') as f:
                f.write(content)
            return f"Content written to file: {path}"
        except Exception as e:
            return f"Error writing to file: {str(e)}"

    def read_file(self, path):
        try:
            with open(path, 'r') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def list_files(self, path="."):
        try:
            files = os.listdir(path)
            return "\n".join(files)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def tavily_search(self, query):
        try:
            response = self.tavily.qna_search(query=query, search_depth="advanced")
            return response
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def execute_tool(self, tool_name, tool_input):
        if tool_name == "create_folder":
            return self.create_folder(tool_input["path"])
        elif tool_name == "create_file":
            return self.create_file(tool_input["path"], tool_input.get("content", ""))
        elif tool_name == "write_to_file":
            return self.write_to_file(tool_input["path"], tool_input.get("content", ""))
        elif tool_name == "read_file":
            return self.read_file(tool_input["path"])
        elif tool_name == "list_files":
            return self.list_files(tool_input.get("path", "."))
        elif tool_name == "tavily_search":
            return self.tavily_search(tool_input["query"])
        else:
            return f"Unknown tool: {tool_name}"

    def encode_image_to_base64(self, image_path):
        try:
            with Image.open(image_path) as img:
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.DEFAULT_STRATEGY)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        except Exception as e:
            return f"Error encoding image: {str(e)}"

    def parse_goals(self, response):
        return re.findall(r'Goal \d+: (.+)', response)

    def execute_goals(self, goals):
        for i, goal in enumerate(goals, 1):
            self.print_colored(f"\nExecuting Goal {i}: {goal}", self.TOOL_COLOR)
            response, _ = self.chat_with_ai(f"Continue working on goal: {goal}")
            if self.CONTINUATION_EXIT_PHRASE in response:
                self.automode = False
                self.print_colored("Exiting automode.", self.TOOL_COLOR)
                break
    
    def chat_with_ai(self, user_input, image_path=None, current_iteration=None, max_iterations=None):
        if image_path:
            self.print_colored(f"Processing image at path: {image_path}", self.TOOL_COLOR)
            image_base64 = self.encode_image_to_base64(image_path)

            if image_base64.startswith("Error"):
                self.print_colored(f"Error encoding image: {image_base64}", self.TOOL_COLOR)
                return "I'm sorry, there was an error processing the image. Please try again.", False

        if self.default_model == "claude":
            response, exit_continuation = self.chat_with_claude(user_input, image_path, current_iteration, max_iterations)
        elif self.default_model == "gemini":
            response, exit_continuation = self.chat_with_gemini(user_input, image_path, current_iteration, max_iterations)
        else:
            return "Invalid default model selected.", False
        
    def chat_with_claude(self, user_input, image_path=None, current_iteration=None, max_iterations=None):
        if image_path:
            self.print_colored(f"Processing image at path: {image_path}", self.TOOL_COLOR)
            image_base64 = self.encode_image_to_base64(image_path)
            
            if image_base64.startswith("Error"):
                self.print_colored(f"Error encoding image: {image_base64}", self.TOOL_COLOR)
                return "I'm sorry, there was an error processing the image. Please try again.", False

            image_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": f"User input for image: {user_input}"
                    }
                ]
            }
            self.conversation_history.append(image_message)
            self.print_colored("Image message added to conversation history", self.TOOL_COLOR)
        else:
            self.conversation_history.append({"role": "user", "content": user_input})
        
        messages = [msg for msg in self.conversation_history if msg.get('content')]
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4000,
                system=self.update_system_prompt(current_iteration, max_iterations),
                messages=messages,
                tools=self.tools,
                tool_choice={"type": "auto"}
            )
        except Exception as e:
            self.print_colored(f"Error calling Claude: {str(e)}", self.TOOL_COLOR)


        if image_path:
            self.conversation_history.append({"role": "assistant", "content": image_message})
            self.conversation_history.append({"role": "assistant", "content": response})
        else:
            self.conversation_history.append({"role": "assistant", "content": response})
        return response, True
    

