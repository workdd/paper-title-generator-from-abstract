from simplet5 import SimpleT5
import json
import os

model = SimpleT5()
model.from_pretrained("t5", "t5-base")

model_name = os.environ('MODEL_NAME')
model.load_model("t5", f"/var/task/{model_name}", use_gpu=False)


def handler(event, context):
    abstract = event['abstract']

    summary = model.predict(abstract)[0]
    return {
        'statusCode': 200,
        'summary': json.dumps(summary, default=str),
        'headers': {
            "Content-Type": "application/json"
        }
    }
