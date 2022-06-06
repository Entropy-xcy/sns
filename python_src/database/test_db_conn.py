import pymongo
from synpred_db import url

# client = pymongo.MongoClient("mongodb://synpred:Synthesis_Predictor@synpred.duckdns.org/?authSource=synpred") 
client = pymongo.MongoClient(url) 

db = client.synpred

try: 
    print("Design Count:", db.design.count_documents({}))
    print("Connection: Okay")
except:
    print("Connection: Error")
