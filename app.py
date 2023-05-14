from simplet5 import SimpleT5
import json

model = SimpleT5()
model.from_pretrained("t5", "t5-base")
model.load_model(r"outputs/simplet5-epoch-5-train-loss-1.3866-val-loss-1.7207", use_gpu=False)

# generate
model.predict("summarize:  some text you want to test it on")


def handler(event, context):
    sample_abstracts = test_df.sample(10)

    abstract = event['abstract']

    summary = model.predict(abstract)

    return {
        'statusCode': 200,
        'summary': json.dumps(summary, default=str),
        'headers': {
            "Content-Type": "application/json"
        }
    }
