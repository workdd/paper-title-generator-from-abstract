from simplet5 import SimpleT5
import json

model = SimpleT5()
model.from_pretrained("t5", "t5-base")
model.load_model(r"T5-epoch1", use_gpu=False)


def handler(event, context):
    abstract = event['abstract']

    summary = model.predict(abstract)

    return {
        'statusCode': 200,
        'summary': json.dumps(summary, default=str),
        'headers': {
            "Content-Type": "application/json"
        }
    }
