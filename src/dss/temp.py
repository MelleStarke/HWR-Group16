import json

d = {}

with open("./output_dictionary.json", "r") as file:
  d = json.load(file)
  
[print(x) for x in d.items()]