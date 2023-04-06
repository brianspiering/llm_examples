"""
Generate text using OpenAI API
"""

import os

import openai as ai


ai.organization = "org-rqM8eCgmlkxwmdWF6j2mzysj"
ai.api_key = os.getenv("OPENAI_API_KEY")

def generate_gpt3_response(prompt: str, max_tokens: int=35) -> str:
    """
    Query OpenAI API to get response.
    """

    response = ai.Completion.create(
        engine='text-davinci-003',  # GPT-3
        temperature=0.5,            
        prompt=prompt,           
        max_tokens=max_tokens,    
        n=1,              
    )

    return [c.text.strip() for c in response["choices"]]

def generate_prompt(name: str, topic) -> str:
    """
    Create prompt that gets sent to OpenAI API.
    """

    prompt = f"""Write an email from {name} about {topic}."""
    
    return prompt

if __name__ == '__main__':

    name = "Brian"
    topic = "Checking in…"
    
    prompt = generate_prompt(name, topic)
    response = generate_gpt3_response(prompt=prompt)
    
    print(response)
