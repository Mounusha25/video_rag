import certifi, os, sys
from dotenv import load_dotenv
load_dotenv()

from transformers import CLIPProcessor, CLIPModel
import torch

proc = CLIPProcessor.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
mdl = CLIPModel.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
inp = proc(text=['dancing man in black trenchcoat'], return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    f = mdl.get_text_features(**inp)
    f = f / f.norm(p=2, dim=-1, keepdim=True)
vec = f[0].tolist()

from pymongo import MongoClient
c = MongoClient(os.environ['MONGODB_URI'], tlsCAFile=certifi.where())
pipeline = [
    {'$vectorSearch': {'index': 'frame_vectors', 'path': 'embedding', 'queryVector': vec, 'limit': 10, 'numCandidates': 200}},
    {'$addFields': {'score': {'$meta': 'vectorSearchScore'}}}
]
results = list(c['videorag']['frames'].aggregate(pipeline))
if not results:
    print("NO RESULTS - check index name or collection")
for r in results:
    print(f"score={r['score']:.4f}  time={r.get('time')}  title={r.get('title','')[:50]}")
