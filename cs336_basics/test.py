import json

with open("test", encoding='utf-8') as f:
    t = f.read()
    txt = json.loads(t)

print(txt)