import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/cache/'
from simplet5 import SimpleT5
import json

model = SimpleT5()
model_name = os.environ['MODEL_NAME']
model.load_model("t5", f"/var/task/{model_name}", use_gpu=False)


def handler(event, context):
    abstract = event['abstract']
    abstract = "summarize: " + abstract
    
    summary = model.predict(abstract)[0]
    return {
        'statusCode': 200,
        'summary': json.dumps(summary, default=str),
        'headers': {
            "Content-Type": "application/json"
        }
    }
