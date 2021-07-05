import json

    
def json_read(path):
    with open(path, 'r') as file:
        data = json.load(file)

    return data