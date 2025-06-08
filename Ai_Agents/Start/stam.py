from google import genai

client = genai.Client(api_key="AIzaSyBMQYqgqg_WjYBNJOe2hE4VS-BOcjv7wsU")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["How does AI work?"]
)
print(response.text)