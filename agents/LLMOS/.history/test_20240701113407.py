import os
from datetime import datetime
import json
from colorama import init, Fore, Style
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
from tavily import TavilyClient
from typing import List, Dict
import pygments.util
import base64
from PIL import Image
import io
import re
from anthropic import Anthropic
import google.generativeai as genai

# --- Configuration ---
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"
TAVILY_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyBkTJsctYOkljL0tx-6Y8NwYCaSz-r0XmU"
MODEL_NAME = "gemini-pro" 
DEFAULT_MODEL = "gemini"  # Choose "claude" or "gemini"

# --- Constants ---
CONTINUATION_EXIT_PHRASE = "AUTOMODE_COMPLETE"
MAX_CONTINUATION_ITERATIONS = 25

# --- Initialize Clients ---
genai.configure(api_key=GEMINI_API_KEY) 

class MultimodalAIChat:
    def __init__(self):
        # Initialize colorama
        init()
        self.USER_COLOR = Fore.WHITE
        self.AI_COLOR = Fore.BLUE
        self.TOOL_COLOR = Fore.YELLOW
        self.RESULT_COLOR = Fore.GREEN

        # Initialize AI clients (only if needed)
        self.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if DEFAULT_MODEL == "claude" else None
        self.gemini_model = genai.GenerativeModel(MODEL_NAME) if DEFAULT_MODEL == "gemini" else None
        self.default_model = DEFAULT_MODEL
        # Initialize other tools
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY) 
        
        # Conversation and Automode State
        self.conversation_history = []
        self.automode = False
        self.context = ""  # For potential context management

        # --- System Prompts (Tailored) ---
        self.claude_system_prompt = """
        You are Claude, an AI assistant. You are an exceptional software 
        developer. You NEVER remove existing code unless explicitly asked. 
        You analyze images and incorporate observations into responses. 
        {automode_status}
        {iteration_info}
        Answer the user's request using the available tools. Analyze the 
        situation within <thinking></thinking> tags. 
        """

        self.gemini_system_prompt = """
        You are an AI assistant and a skilled software developer. Avoid 
        removing existing code without clear instructions. Analyze images and 
        use your observations in responses.
        {automode_status}
        {iteration_info}
        Use the available tools thoughtfully to address the user's requests.
        """

        # --- Tools Definition --- 
        self.tools = [
            {
                "name": "create_folder",
                "description": "Create a folder at the specified path.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Folder path"}
                    },
                    "required": ["path"]
                }
            },
            # ... [Add other tools similarly: create_file, write_to_file, etc.] 
            {
                "name": "tavily_search",
                "description": "Performs a web search and returns an answer.",
                "input_schema": {
                    "type": "object", 
                    "properties": {
                        "query": {"type": "string", "description": "The search query"} 
                    },
                    "required": ["query"] 
                }
            }
        ]

    # --- Utility Functions ---
    def print_colored(self, text, color):
        print(f"{color}{text}{Style.RESET_ALL}")

    def print_code(self, code, language=""): 
        try:
            lexer = get_lexer_by_name(language, stripall=True)
            formatter = TerminalFormatter()
            highlighted_code = highlight(code, lexer, formatter)
            print(highlighted_code)
        except pygments.util.ClassNotFound:
            print(f"Code:\n{code}")  

    def encode_image_to_base64(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.thumbnail((1024, 1024)) # Resize if too large
                img = img.convert('RGB')  # Ensure RGB format
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return f"Error encoding image: {str(e)}" 

    def update_system_prompt(self, current_iteration=None, max_iterations=None):
        automode_status = "You are in automode." if self.automode else ""
        iteration_info = f"Iteration {current_iteration}/{max_iterations}" if self.automode else ""
        if self.default_model == "claude":
            return self.claude_system_prompt.format(automode_status=automode_status, iteration_info=iteration_info)
        else:  # Gemini
            return self.gemini_system_prompt.format(automode_status=automode_status, iteration_info=iteration_info)

    # --- Tool Execution ---
    def execute_tool(self, tool_name, tool_input):
        if tool_name == "create_folder":
            return self.create_folder(tool_input["path"])
        # ... [Add execution logic for other tools]
        elif tool_name == "tavily_search":
            try:
                response = self.tavily.qna_search(query=tool_input["query"], search_depth="advanced")
                return response  # Consider formatting for better presentation
            except Exception as e:
                return f"Error performing search: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"

    # --- AI Interaction Logic ---
    def chat_with_claude(self, user_input, image_path=None, current_iteration=None, max_iterations=None):
        if image_path:
            image_base64 = self.encode_image_to_base64(image_path)
            if "Error" in image_base64:
                return image_base64, False

            image_message = {
                "role": "user",
                "content": [
                    {"type": "image", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": f"User input for image: {user_input}"}
                ]
            }
            self.conversation_history.append(image_message)

        else:
            self.conversation_history.append({"role": "user", "content": user_input})

        try: 
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4000,
                system=self.update_system_prompt(current_iteration, max_iterations), 
                messages=self.conversation_history,
                tools=self.tools,
                tool_choice={"type": "auto"}
            )

            assistant_response = ""
            exit_continuation = False
            for block in response.content:
                if block.type == "text":
                    assistant_response += block.text
                    if CONTINUATION_EXIT_PHRASE in block.text:
                        exit_continuation = True
                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id 
                    
                    tool_result = self.execute_tool(tool_name, tool_input) 
                    self.conversation_history.append({"role": "assistant", "content": tool_result})
                    self.print_colored(f"\nTool Used: {tool_name}", self.TOOL_COLOR)
                    self.print_colored(f"Tool Input: {tool_input}", self.TOOL_COLOR)
                    self.print_colored(f"Tool Result: {tool_result}", self.RESULT_COLOR)

                    # Get the AI's response after using the tool
                    response = self.anthropic_client.messages.create( 
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=4000,
                        system=self.update_system_prompt(current_iteration, max_iterations), 
                        messages=self.conversation_history,
                        tools=self.tools,
                        tool_choice={"type": "auto"} 
                    )
                    for block in response.content:
                        if block.type == "text":
                            assistant_response += block.text
                            if CONTINUATION_EXIT_PHRASE in block.text:
                                exit_continuation = True

            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response, exit_continuation

        except Exception as e:
            self.print_colored(f"Error calling Claude API: {str(e)}", self.TOOL_COLOR)
            return "Error communicating with Claude.", False 

    def chat_with_gemini(self, user_input: str, image_path=None, current_iteration=None, 
                       max_iterations=None):
        """Interacts with Google Gemini.
        """
        if image_path:
            self.print_colored("Gemini doesn't support images yet.", self.TOOL_COLOR)
            return "I can't process images right now.", False

        response = get_gemini_response(
            user_input, 
            self.update_system_prompt(current_iteration, max_iterations), 
            self.context,  # Pass context if using
            self.conversation_history
        )

        self.print_colored(f"\nGemini: {response}", self.AI_COLOR)
        self.conversation_history.append({"role": "assistant", "parts": [response]}) 

        if self.CONTINUATION_EXIT_PHRASE in response:
            return response, True 
        else:
            return response, False

    def process_and_display_response(self, response):
        if response.startswith("Error") or response.startswith("I'm sorry"):
            self.print_colored(response, self.TOOL_COLOR) 
        else: 
            # (Optional) Add more sophisticated response processing here
            #   - Extract code blocks 
            #   - Handle different response formats
            self.print_colored(response, self.AI_COLOR)

    def main(self):
        self.print_colored("Welcome to the Multimodal AI Chat!", self.AI_COLOR)
        self.print_colored("Type 'exit' to quit, 'image' to use an image.", self.AI_COLOR)
        self.print_colored("Use 'automode [iterations]' for autonomous mode.", self.AI_COLOR)
        
        while True:
            user_input = input(f"\n{self.USER_COLOR}You: {Style.RESET_ALL}")
            if user_input.lower() == 'exit':
                break

            if user_input.lower() == 'image':
                image_path = input(f"{self.USER_COLOR}Image path: {Style.RESET_ALL}")
                if os.path.isfile(image_path):
                    user_input = input(f"{self.USER_COLOR}Prompt (with image context): {Style.RESET_ALL}")
                    if self.default_model == "claude":
                        response, _ = self.chat_with_claude(user_input, image_path)
                    else: 
                        response, _ = self.chat_with_gemini(user_input, image_path)
                    self.process_and_display_response(response)
                else:
                    self.print_colored("Invalid image path.", self.AI_COLOR)
            
            elif user_input.lower().startswith('automode'):
                try:
                    parts = user_input.split()
                    max_iterations = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else MAX_CONTINUATION_ITERATIONS
                    self.automode = True
                    self.print_colored(f"Automode ({max_iterations} iterations). Ctrl+C to stop.", self.TOOL_COLOR)
                    user_input = input(f"\n{self.USER_COLOR}You: {Style.RESET_ALL}") 
                    
                    iteration_count = 0
                    while self.automode and iteration_count < max_iterations:
                        if self.default_model == "claude":
                            response, exit_continuation = self.chat_with_claude(user_input, current_iteration=iteration_count+1, max_iterations=max_iterations) 
                        else:
                            response, exit_continuation = self.chat_with_gemini(user_input, current_iteration=iteration_count+1, max_iterations=max_iterations)
                        self.process_and_display_response(response)

                        if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                            self.print_colored("Automode complete.", self.TOOL_COLOR)
                            self.automode = False
                        else:
                            iteration_count += 1
                            user_input = "Continue with the next step." 

                    if iteration_count >= max_iterations:
                        self.print_colored("Max iterations reached.", self.TOOL_COLOR)
                        self.automode = False
                    
                except KeyboardInterrupt:
                    self.print_colored("\nAutomode interrupted.", self.TOOL_COLOR)
                    self.automode = False
                
                self.print_colored("Exited automode.", self.TOOL_COLOR)

            else: # Regular chat 
                if self.default_model == "claude":
                    response, _ = self.chat_with_claude(user_input)
                else:
                    response, _ = self.chat_with_gemini(user_input)
                self.process_and_display_response(response)

if __name__ == "__main__":
    chatbot = MultimodalAIChat()
    chatbot.main() 