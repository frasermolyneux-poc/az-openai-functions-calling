import os
import json
import requests
from openai import AzureOpenAI
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv("config.env")

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview"
)

# Define the deployment you want to use for your chat completions API calls

deployment_name = "gpt-4o"

def get_current_intensity(location, fuel):
    """Get the current time for a given location"""
    print(f"get_current_intensity called with location: {location} and {fuel}")  
    location_lower = location.lower()
    fuel_lower = fuel.lower()

    # Call the carbon intensity API
    response = requests.get(f"https://api.carbonintensity.org.uk/regional/{location_lower}")
    
    if response.status_code != 200:
        return {"error": "Failed to fetch data from the API"}
    
    data = response.json()
    
    # Extract relevant data
    try:
        region_data = data['data'][0]
        intensity_data = region_data['data'][0]['intensity']
        generation_mix = region_data['data'][0]['generationmix']
        
        # Find the specific fuel data
        fuel_data = next((item for item in generation_mix if item['fuel'] == fuel_lower), None)
        
        if not fuel_data:
            return {"error": f"Fuel type '{fuel}' not found in the generation mix"}
        
        result = {
            "region": region_data['shortname'],
            "intensity_forecast": intensity_data['forecast'],
            "intensity_index": intensity_data['index'],
            "fuel": fuel_data['fuel'],
            "fuel_percentage": fuel_data['perc']
        }
        
        return json.dumps(result, indent=4)
    
    except (KeyError, IndexError) as e:
        return json.dumps({"error": "Error parsing the API response", "details": str(e)})


def run_conversation():
    # Initial user message
    messages = [{"role": "user", "content": "What's the current gas percentage used in england?"}] # Parallel function call with a single tool/function defined
    #messages = [{"role": "user", "content": "What's the current gas and biomass percentage used in england compared to wales?"}] # Parallel function call with a single tool/function defined

    # Define the function for the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_intensity",
                "description": "Get the current fuel usage in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The region name e.g. england, scotland or wales",
                        },
                        "fuel": {
                            "type": "string",
                            "description": "The fuel type e.g. gas, biomass, nuclear, solar, wind, hydro, coal, imports",
                        },
                    },
                    "required": ["location", "fuel"],
                },
            }
        }
    ]

    # First API call: Ask the model to use the function
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Process the model's response
    response_message = response.choices[0].message
    messages.append(response_message)

    print("Model's response:")  
    print(response_message)  

    # Handle function calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_current_intensity":
                function_args = json.loads(tool_call.function.arguments)
                print(f"Function arguments: {function_args}")  
                time_response = get_current_intensity(
                    location=function_args.get("location"),
                    fuel=function_args.get("fuel")
                )
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "get_current_intensity",
                    "content": time_response,
                })
    else:
        print("No tool calls were made by the model.")  

    # Second API call: Get the final response from the model
    final_response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
    )

    return final_response.choices[0].message.content

# Run the conversation and print the result
print(run_conversation())