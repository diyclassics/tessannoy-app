from pathlib import Path
import random
import pickle

import gdown

import pandas as pd
import numpy as np

import streamlit as st
from streamlit_extras.no_default_selectbox import selectbox

# TODO: Make this install from package rather than local copy
from tessannoy import TessAnnoyIndex


def clear_select():
    # TODO: Not working?
    print("clearing selectbox")
    st.session_state.line = None


st.set_page_config(layout="wide")

st.title("TessAnnoy")
st.subheader("Vector-based verse search for CLTK-Tesserae texts")
st.text("*Only available currently for epic*")

import streamlit as st

# Using object notation
model_select = st.sidebar.selectbox("Model:", ("lg", "trf"))

st.text(f"Using model: {model_select}")


@st.cache_data
def load_data():
    # cf. https://discuss.streamlit.io/t/how-to-download-large-model-files-to-the-sharing-app/7160/5
    data_dest = Path("data")
    data_dest.mkdir(exist_ok=True)

    checkpoints = {
        "data/tess_epic_line_lg_vectors.pkl": "https://drive.google.com/uc?id=1hmlFrlzDcggCEAu_eNiIlACuKRYMyjf1",
        "data/tess_epic_line_trf_vectors.pkl": "https://drive.google.com/uc?id=1EIVDSKKf_-NAgc6Bct2PnlRjY8qq4NRA",
    }

    for checkpoint, remote in checkpoints.items():
        print(f"Checking {checkpoint}...")
        if not Path(checkpoint).exists():
            print(Path(checkpoint).exists())
            print(Path(checkpoint))
            gdown.download(remote, checkpoint, quiet=False)
            pass
        else:
            print(f"Found {checkpoint}.")

    lg_data = pickle.load(open("data/tess_epic_line_lg_vectors.pkl", "rb"))
    trf_data = pickle.load(open("data/tess_epic_line_trf_vectors.pkl", "rb"))
    data = {
        "lg": lg_data,
        "trf": trf_data,
    }
    return data


@st.cache_data
def create_indexes():
    indexes = {
        "lg": "data/tess_epic_line_lg_vectors.pkl",
        "trf": "data/tess_epic_line_trf_vectors.pkl",
    }

    for model, index in indexes.items():
        vectors = pickle.load(open(index, "rb"))
        citations = np.array(list(vectors.keys()))
        texts = np.array([vectors[citation]["text"] for citation in citations])
        vectors = np.array([vectors[citation]["vector"] for citation in citations])

        index = TessAnnoyIndex(vectors, texts, citations)
        index.build()
        index.save(f"data/tessannoy_{model}.ann")


data = load_data()
create_indexes()

vectors = data[model_select]
lg_index = "data/tessannoy_lg.ann"
trf_index = "data/tessannoy_trf.ann"

citations = np.array(list(vectors.keys()))
texts = np.array([vectors[citation]["text"] for citation in citations])
vectors = np.array([vectors[citation]["vector"] for citation in citations])

index = TessAnnoyIndex(vectors, texts, citations)
if model_select == "lg":
    index.load(lg_index)
else:
    index.load(trf_index)

line = selectbox("Select line", citations, key="selectbox")


def get_results(query):
    results = []
    for i, q in enumerate(query, 0):
        results.append(
            [
                i,
                q[0],
                q[1],
                q[2],
            ]
        )
    df = pd.DataFrame(
        results, columns=["Rank", "Citation", "Text", "Cos. Sim."], index=None
    )

    st.dataframe(df, width=720, hide_index=True)


random_button = st.button("Show random verse", on_click=clear_select)

if random_button or line:
    # print(selectbox)
    if random_button:
        random_item = random.randint(0, len(texts) - 1)
        query = index.query(vectors[random_item], k=25)
    else:
        query_idx = np.where(citations == line)[0][0]
        query = index.query(vectors[query_idx], k=25)
    get_results(query)
