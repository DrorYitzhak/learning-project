import pandas as pd
import requests


url = "https://api.ipify.org?format=json"
response_j = requests.get(url)
print(response_j)

response_d = response_j.json()
print(response_d)
print(response_d['ip'])

df = pd.read_json(response_j)

"""
Find your computer's IP address
"""