import pandas as pd
import json
import pickle
from numpy import linalg as LA

from sklearn.feature_extraction.text import TfidfVectorizer

# with open('training.json', "r") as f:
#     d = json.loads(f.read())

# train = pd.DataFrame.from_dict(d["data"])
# print(train.head(3))

def cosine_similarity(arr1, arr2):
    return np.dot(arr1, arr2)/(LA.norm(arr1)*LA.norm(arr2))

def read_train(filename):
    train = pd.read_json(filename)
    df = pd.DataFrame()
    context_count = 0
    contexts = []
    qa_dfs = []
    for i in range(len(train)):
        curr = train.loc[i, "data"]
        lsts = curr["paragraphs"] # each contains context, qas
        # print(lsts[0])
        # print("=========")
        # print(lsts[1])
        for item in lsts:
            contexts.append(item["context"])
            curr_qas = item["qas"]
            # columns in tmp_df: [u'answers', u'id', u'is_impossible', u'question', u'context']
            tmp_df = pd.DataFrame.from_records(curr_qas) 
            tmp_df["context_idx"] = context_count
            qa_dfs.append(tmp_df)
            context_count += 1
        print(i)
    context_df = pd.DataFrame.from_dict({"context": contexts})
    df = pd.concat(qa_dfs)
    return context_df, df

def save_pickle(filename, obj):
    pickling_on = open(filename + ".pickle","wb")
    pickle.dump(obj, pickling_on)
    pickling_on.close()

train_context, train = read_train("training.json")
save_pickle("train_context", train_context)
save_pickle("train", train)

test_context, test = read_train("test.json")
save_pickle("test_context", test_context)
save_pickle("test", test)

# print("length of training is", len(train))
# print("length of train context is ", len(train_context))
# print(train.head(3))

# print("================================================")

# print("length of test is", len(test))
# print("length of test context is ", len(test_context))
# print(test.head(3))

# add a column with similarity score in df
def compute_sim(context_df, df):
    df["similarity"] = 0
    for i in range(len(context_df)):
        vectorizer = TfidfVectorizer(max_df=0.95, stop_words="english") # build a vectorizer for each context
        curr_context = context_df.loc[i, "context"]
        vectorizer.fit(curr_context)
        context_vec = vectorizer.transform(curr_context)
        qas = df[df["context_idx"] == i].reset_index()
        for j in range(len(qas)):
            q_vec = vectorizer.transform(qas.loc[j, "question"])

        