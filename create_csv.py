import h5py
import numpy as np
import pandas as pd

dataset = pd.read_csv("dataset.csv")
uuid = list(dataset["company_uuid"])
back = {uuid: i for i, uuid in enumerate(uuid)}

uuid_list = input("Please input the name of the UUID list: ")


with open(uuid_list, "r") as f:
    N = len(f.readlines())

dta = h5py.File("dataset.hdf5", "r")
sims = dta["cos_sims"]
key_sims = dta["key_sims"]
rels = dta["reliabilities"]
with open(uuid_list, "r") as f:
    id = f.readline().strip()
    if id not in uuid:
        print(f"ignoring first line: {id}")
        N -= 1
        id = f.readline().strip()
    inds = []
    uuids = []
    while id != "":
        if id not in back:
            print(f"Error, {id} in list doesn't correspond to a company in the dataset")
        inds.append(back[uuid])
        uuids.append(uuid)
    inds = np.array(inds)
    similarities = sims[inds]
    sims_df = pd.DataFrame(similarities, index=uuids, columns=uuids)
    sims_df.to_csv("cos_sims.csv")
    key_df = pd.DataFrame(key_sims[inds], index=uuids, columns=uuids)
    key_df.to_csv("key_sims.csv")

    reliabilities = rels[inds]
    rels_df = pd.DataFrame(reliabilities, index=uuids, columns=uuids)
    rels_df.to_csv("reliabilities.csv")

dta.close()