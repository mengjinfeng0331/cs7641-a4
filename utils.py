import json

def write2json(file,data):
    with open(file, 'w') as file:
        file.write(json.dumps(data, indent=4))
        
        