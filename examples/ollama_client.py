import ollama
import requests
import json

# --- Configuration ---
# Your FreeCADMCP server's address
MCP_SERVER_URL = "http://localhost:9875" 

# The Ollama model you are using
OLLAMA_MODEL = "llama3" # Or "phi3", "mistral", etc.

# --- 1. Get Tool Definitions from the MCP Server ---
# This part is the same as before. It gets the list of available tools.
try:
    response = requests.get(f"{MCP_SERVER_URL}/tools/schema")
    response.raise_for_status()
    tools = response.json()
    print("✅ Successfully fetched tools from FreeCADMCP server.")
except requests.exceptions.RequestException as e:
    print(f"❌ Failed to fetch tools from server: {e}")
    tools = [] 
    exit()

# --- 2. Main Conversation Loop ---
# The system prompt helps the model understand its role.
messages = [
    {
        "role": "system",
        "content": "You are a helpful FreeCAD assistant. You must use the provided tools to interact with the FreeCAD application based on the user's request. When creating assets, first check the library with get_parts_list, then try to create objects, and always verify your work.",
    }
]

print("🤖 Ollama FreeCAD agent is ready. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    messages.append({"role": "user", "content": user_input})

    # --- 3. Call the Ollama Model with Tools ---
    # The client streams the response to show thinking process.
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        tools=tools,
    )
    
    # Add the AI's response to the conversation history
    messages.append(response['message'])
    
    # --- 4. Execute Tool Calls if the Model Requests It ---
    if response['message'].get('tool_calls'):
        # The model wants to use a tool
        tool_calls = response['message']['tool_calls']
        
        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            function_args = tool_call['function']['arguments']
            
            print(f"🧠 AI wants to call function: {function_name} with args: {function_args}")

            # --- 5. Call the MCP Server's Tool Endpoint ---
            try:
                tool_response = requests.post(
                    f"{MCP_SERVER_URL}/tools/{function_name}",
                    json=function_args
                )
                tool_response.raise_for_status()
                # Get the result from the tool server
                function_result = tool_response.json()

            except requests.exceptions.RequestException as e:
                function_result = {"error": f"Failed to call tool server: {e}"}
            
            print(f"🔧 Tool Result: {function_result}")
            
            # --- 6. Send Tool Result Back to Ollama ---
            # Append the result so the model knows what happened
            messages.append({
                'role': 'tool',
                'content': json.dumps(function_result),
            })
            
            # --- 7. Get the Final AI Response After Using the Tool ---
            final_response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages
            )
            
            print(f"AI: {final_response['message']['content']}")
            messages.append(final_response['message']) # Add final response to history
    else:
        # If no tool call was made, just print the AI's plain text response
        print(f"AI: {response['message']['content']}")

