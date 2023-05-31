from urllib.parse import parse_qs
import base64
import json
from simplet5 import SimpleT5
import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/cache/'

model = SimpleT5()
model_name = os.environ['MODEL_NAME']
model.load_model("t5", f"/var/task/{model_name}", use_gpu=False)

def handler(event, context):
    e = event['body']
    e = json.dumps(parse_qs(e))
    e = json.loads(e)
    abstract = e['abstract'][0]
    abstract = "summarize: " + abstract

    summary = model.predict(abstract)[0]
    result = {'summary': summary}
    return {
        'statusCode': 200,
        'body': json.dumps(result),
        'headers': {
            "Content-Type": "application/json",
            'Access-Control-Allow-Origin': '*'
        }
    }
