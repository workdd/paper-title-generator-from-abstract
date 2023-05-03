import re
import json
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from simplet5 import SimpleT5

data_file = r'arxiv-dataset.json'

paper_categories = set(["cs.AI",  # Artificial Intelligence
                        "cs.CV",  # Computer Vision and Pattern Recognition
                        "cs.LG",
                        "cs.CL",
                        "cs.CC",
                        "cs.CE",
                        "cs.CG",
                        "cs.GT",
                        "cs.CY",
                        "cs.CR",
                        "cs.DS",
                        "cs.DB",
                        "cs.DL",
                        "cs.DM",
                        "cs.DC",
                        "cs.ET",
                        "cs.FL",
                        "cs.GL",
                        "cs.GR",
                        "cs.AR",
                        "cs.HC",
                        "cs.IR",
                        "cs.IT",
                        "cs.LO",
                        "cs.LG",
                        "cs.MS",
                        "cs.MA",
                        "cs.MM",
                        "cs.NI",
                        "cs.NE",
                        "cs.NA",
                        "cs.OS",
                        "cs.OH",
                        "cs.PF",
                        "cs.PL",
                        "cs.RO",
                        "cs.SI",
                        "cs.SE",
                        "cs.SD",
                        "cs.SC",
                        "cs.SY"])  # Machine Learning


def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line


def build_dataset(paper_categories):
    titles = []
    abstracts = []
    metadata = get_metadata()
    for paper in tqdm(metadata):
        paper_dict = json.loads(paper)
        category = paper_dict.get('categories')
        if len(paper_categories.intersection(set(category.split(" ")))) > 0:
            try:
                year = int(paper_dict.get('journal-ref')[-4:])
                titles.append(paper_dict.get('title'))
                abstracts.append(paper_dict.get('abstract').replace("\n", ""))
            except:
                pass

    papers = pd.DataFrame({'title': titles, 'abstract': abstracts})
    papers = papers.dropna()
    papers["title"] = papers["title"].apply(lambda x: re.sub('\s+', ' ', x))
    papers["abstract"] = papers["abstract"].apply(lambda x: re.sub('\s+', ' ', x))

    del titles, abstracts
    return papers


papers = build_dataset(paper_categories)
papers = papers[['abstract', 'title']]
papers.columns = ["source_text", "target_text"]

# let's add a prefix to source_text, to uniquely identify kind of task we are performing on the data, in this case --> "summarize"
papers['source_text'] = "summarize: " + papers['source_text']

train_df, test_df = train_test_split(papers, test_size=0.1, random_state=42)

model = SimpleT5()
model.from_pretrained("t5", "t5-base")
model.train(train_df=train_df,
            eval_df=test_df,
            source_max_token_len=512,
            target_max_token_len=128,
            max_epochs=1,
            batch_size=64,
            use_gpu=False,
            dataloader_num_workers=32)

# model.load_model(r"outputs/simplet5-epoch-5-train-loss-1.3866-val-loss-1.7207", use_gpu=False)
#
# # generate
# model.predict("summarize:  some text you want to test it on")
#
# sample_abstracts = test_df.sample(10)
#
# for i, abstract in sample_abstracts.iterrows():
#     print(f"===== Abstract =====")
#     print(abstract['source_text'])
#     summary = model.predict(abstract['source_text'])[0]
#     print(f"\n===== Actual Title =====")
#     print(f"{abstract['target_text']}")
#     print(f"\n===== Generated Title =====")
#     print(f"{summary}")
#     print("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
