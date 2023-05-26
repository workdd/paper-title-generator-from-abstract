from simplet5 import SimpleT5
import json

model = SimpleT5()
model.from_pretrained("t5", "t5-base")
model.load_model("models/T5-epoch4", use_gpu=False)


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


event = {
    'abstract': "title: The game of Go has long been viewed as the most challenging of classic games for artificial intelligence owing to its enormous search space and the difficulty of evaluating board positions and moves. Here we introduce a new approach to computer Go that uses ‘value networks’ to evaluate board positions and ‘policy networks’ to select moves. These deep neural networks are trained by a novel combination of supervised learning from human expert games, and reinforcement learning from games of self-play. Without any lookahead search, the neural networks play Go at the level of state-of-the-art Monte Carlo tree search programs that simulate thousands of random games of self-play. We also introduce a new search algorithm that combines Monte Carlo simulation with value and policy networks. Using this search algorithm, our program AlphaGo achieved a 99.8% winning rate against other Go programs, and defeated the human European Go champion by 5 games to 0. This is the first time that a computer program has defeated a human professional player in the full-sized game of Go, a feat previously thought to be at least a decade away."
}

res = handler(event, "")

print(res)
