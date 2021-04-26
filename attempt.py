import pandas as pd
import numpy as np
from os.path import join
from fuzzywuzzy import fuzz
from sklearn.ensemble import RandomForestClassifier

# 1. read data

ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))

def generatePercentPriceDiff(row):
    return (abs(row["price_l"]-row["price_r"]))/((row["price_l"]+row["price_r"])/2)*100

def generateTitleFuzzRatio(row):
    return fuzz.token_set_ratio(row["title_l"], row["title_r"])

# or_modelno since some brands are mislabled and some are nan
def getCandidates(ltable, rtable):
    # ensure brand is str
    ltable["brand"] = ltable["brand"].astype(str)
    rtable["brand"] = rtable["brand"].astype(str)
    ltable["modelno"] = ltable["modelno"].astype(str)
    rtable["modelno"] = rtable["modelno"].astype(str)

    # get all brands
    brands_l = set(ltable["brand"].values)
    brands_r = set(rtable["brand"].values)
    brands = brands_l.union(brands_r)
    modelno_l = set(ltable["modelno"].values)
    modelno_r = set(rtable["modelno"].values)
    models = modelno_l.union(modelno_r)

    # map each brand to left ids and right ids
    brand2ids_l = {b.lower(): [] for b in brands}
    brand2ids_r = {b.lower(): [] for b in brands}
    modelno2ids_l = {m.lower(): [] for m in models}
    modelno2ids_r = {m.lower(): [] for m in models}
    for i, x in ltable.iterrows():
        brand2ids_l[x["brand"].lower()].append(x["id"])
        modelno2ids_l[x["modelno"].lower()].append(x["id"])

    for i, x in rtable.iterrows():
        brand2ids_r[x["brand"].lower()].append(x["id"])
        modelno2ids_r[x["modelno"].lower()].append(x["id"])

    # put id pairs that share the same brand in candidate set
    candset = set()
    for brd in brands:
        l_ids = brand2ids_l[brd]
        r_ids = brand2ids_r[brd]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.add((l_ids[i], r_ids[j]))
    for model in models:
        l_ids = modelno2ids_l[model]
        r_ids = modelno2ids_r[model]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.add((l_ids[i], r_ids[j]))
    return list(candset)


def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR


def generateBrandFuzzRatio(row):
    # do not disqualify any entry with missing brand data
    return fuzz.token_set_ratio(row["brand_l"], row["brand_r"])


def generateModelnoFuzzRatio(row):
    # do not disqualify any entry with missing modelno data
    if (row["modelno_l"] == "nan" and row["modelno_r"] == "nan"):
        return 0
    return fuzz.token_set_ratio(row["modelno_l"], row["modelno_r"])


def generatePercentPriceDiff(row):
    diff = (abs(row["price_l"]-row["price_r"]))/((row["price_l"]+row["price_r"])/2)*100
    if np.isnan(diff):
        return 150
    return diff

def generateTitleFuzzRatio(row):
    return fuzz.token_set_ratio(row["title_l"], row["title_r"])


def generateCategoryFuzzRatio(row):
    return fuzz.token_set_ratio(row["category_l"], row["category_r"])


def engineerFeatures(LR):
    LR["brand_fuzz_ratio"] = LR.apply(lambda row: generateBrandFuzzRatio(row), axis=1)
    LR["modelno_fuzz_ratio"] = LR.apply(lambda row: generateModelnoFuzzRatio(row), axis=1)
    LR["%_price_diff"] = LR.apply(lambda row: generatePercentPriceDiff(row), axis=1)
    LR["title_fuzz_ratio"] = LR.apply(lambda row: generateTitleFuzzRatio(row), axis=1)
    LR["category_fuzz_ratio"] = LR.apply(lambda row: generateCategoryFuzzRatio(row), axis=1)
    features = LR.drop(columns=[
        "title_l",
        "title_r",
        "brand_l",
        "brand_r",
        "modelno_l",
        "modelno_r",
        "category_l",
        "category_r",
        "price_l",
        "price_r"
    ])
    return features.astype(str)


pd.set_option('display.max_columns', None)
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = engineerFeatures(training_df)
training_labels = train.label.values
rf = RandomForestClassifier(class_weight="balanced", random_state=0)
rf.fit(training_features, training_labels)

candidates = getCandidates(ltable, rtable)
LR = pairs2LR(ltable, rtable, candidates)
candidate_features = engineerFeatures(LR)
predictions = rf.predict(candidate_features)

# use training pairs to create baseline of the averages
matching_pairs = LR.loc[predictions == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

# print(len(matching_pairs))

matching_pairs_in_training = training_df.loc[training_labels == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training

pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])

print(pred_df)
print(len(pred_df))
pred_df.to_csv("output.csv", index=False)