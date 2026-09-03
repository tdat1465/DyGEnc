import os 
import json
import pickle
import gc
from copy import deepcopy
from itertools import pairwise, chain

import torch
import networkx as nx
from tqdm import tqdm
from loguru import logger
from os.path import join as spj
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from src.utils.seed import set_seed
from src.datasets.agqa_storage import INDEX_NAME, build_qa_index, iter_qa_json


root_path = os.environ.get("AGQA_ROOT")
LPE_NUM = 4
MAPPING = {}
MODEL_NAME = "mbert"
model = tokenizer = device = text2embedding = None
SG_GLOBAL = dict()  # Split + video ID -> interval keys, never graph tensors.
SAVE_NETWORKX = os.environ.get("DYGENC_SAVE_NETWORKX", "1") == "1"


def initialize_embedding_model():
    # Importing helpers must not download/load a model or open AGQA data. The
    # process running graph preprocessing owns exactly one embedding model.
    global model, tokenizer, device, text2embedding, MAPPING
    if model is not None:
        return
    if root_path is None:
        raise RuntimeError("AGQA_ROOT must be set before graph preprocessing")
    from src.utils.lm_modeling import load_model, load_text2embedding
    logger.info(f"Setting lpe={LPE_NUM}; loading LM={MODEL_NAME}")
    with open(spj(root_path, "data", "ENG.txt"), "rt", encoding="utf-8-sig") as file:
        MAPPING = json.load(file)
    model, tokenizer, device = load_model[MODEL_NAME]()
    text2embedding = load_text2embedding[MODEL_NAME]
    logger.info(device)


def textualize_graph(nx_graph):
    description_nodes = []
    description_edges = []

    for idx, lab_ in dict(nx_graph.nodes(data=True)).items():
        description_nodes.append(f"{idx}: {lab_['label']}")
    description_nodes = ", ".join(description_nodes)

    for edg_ in nx_graph.edges(data=True):
        description_edges.append(f"{edg_[0]} {edg_[2]['label']} {edg_[1]}")
    description_edges = "; ".join(description_edges)
    
    return "Nodes:" + "\n" + description_nodes + "\n" + "Edges:" + "\n" + description_edges

def load_grounding_frames(grounding_item):
    # see more in AQGA README 'Scene graph grounding'
    # CORNER CASE 1: NO GROUNDING
    if not grounding_item:
        return []
    
    # CORNER CASE 2: only equal cases
    if all([key_pair.split("-")[0] == key_pair.split("-")[1] for key_pair in grounding_item.keys()]):
        return []

    frames_idx = set()
    for ground_elem in list(chain.from_iterable(grounding_item.values())):
        if ground_elem.split("/")[-1].startswith("0"):
            frames_idx.add(ground_elem.split("/")[-1])
    return list(frames_idx)

def parse_sg_keys(sg_item):
    seq = []
    frames = sorted([i for i in sg_item.keys() if i.startswith('0')], key=lambda x: int(x[-6:]))
    for key_frame in frames:
        relevante_keys = [key for key in sg_item.keys() if key.endswith(key_frame)]

        G = nx.DiGraph()
        object_set = set() # there are no multi object cases
        object_set.add("o1") # action genome graph is always related to person

        for entity_key in [key for key in relevante_keys if key.startswith('o')]:
            object_set.add(sg_item[entity_key]["class"])
        # take objects also from relations and verbs
        for entity_key in [key for key in relevante_keys
                if (key.startswith('r') or key.startswith('v'))]:
            object_set.update([item["class"] for item in sg_item[entity_key]["objects"]])
        # Stable node ordering when data is rebuilt on another job/node.
        objects = sorted(object_set)
        sg_object_mapping = {o_class: idx for idx, o_class in enumerate(objects)}

        # add nodes
        for obj in objects:
            G.add_node(sg_object_mapping[obj], label=MAPPING[obj])
        # add edges
        for entity_key in [key for key in relevante_keys
                if (key.startswith('r') or key.startswith('v'))]:
            edge_obj = sg_item[entity_key]["objects"]
            for e_obj in edge_obj:
                G.add_edge(sg_object_mapping["o1"], sg_object_mapping[e_obj["class"]],
                           label=MAPPING[sg_item[entity_key]["class"]])
        seq.append(G)
    return seq, frames

def preprocess_graphs(splits=("train", "test")):
    initialize_embedding_model()
    logger.info("working with graphs")

    os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/", exist_ok=True)
    
    for split in splits:
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}", exist_ok=True)
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/graphs/", exist_ok=True)
        if SAVE_NETWORKX:
            os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/graphs_networkx/", exist_ok=True)
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/descs", exist_ok=True)

        # iterate over splits
        sg_file = f"AGQA_{split}_stsgs.pkl"
        with open(f"{root_path}/data/AGQA_scene_graphs/{sg_file}", "rb") as file:
            sg_data = pickle.load(file)

        for sg_name, sg_item in tqdm(sg_data.items()):
            sg_seq_nx, seq_frame_names = parse_sg_keys(sg_item)
            assert len(sg_seq_nx) > 0
            if SAVE_NETWORKX:
                with open(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/graphs_networkx/{sg_name}.pkl", "wb") as f:
                    pickle.dump(sg_seq_nx, f)
            
            # leave only unique graphs and save indices
            uniq_start_index = None
            for i in range(len(sg_seq_nx)):
                if len(list(nx.get_node_attributes(sg_seq_nx[i], "label").values())) > 0:
                    uniq_start_index = i
                    break
            if uniq_start_index is not None: 
                unique_idx = [uniq_start_index]
                sg1 = sg_seq_nx[uniq_start_index]
                for i in range(uniq_start_index, len(sg_seq_nx)):
                    sg2 = sg_seq_nx[i]
                    if nx.utils.misc.graphs_equal(sg1, sg2) == False:
                        sg1 = sg2
                        # if graph has nodes
                        if len(list(nx.get_node_attributes(sg2, "label").values())) > 0:
                            unique_idx.append(i)
            assert len(unique_idx) > 0
            if (len(sg_seq_nx) - 1) not in unique_idx:
                unique_idx.append(len(sg_seq_nx) - 1) # to close range to the end

            # convert to pyg.Data and embed
            all_pyg_graphs = {}
            all_labels = []
            all_edge_labels = []
            
            for idx in unique_idx:
                labels = list(nx.get_node_attributes(sg_seq_nx[idx], "label").values())
                all_labels.extend(labels)
                edges = list(nx.get_edge_attributes(sg_seq_nx[idx], "label").values())
                all_edge_labels.extend(edges)

            node_embed = text2embedding(model, tokenizer, device, all_labels)
            edge_embed = text2embedding(model, tokenizer, device, all_edge_labels)
            
            pairwise_array = deepcopy(unique_idx)
            pairwise_array.append(len(sg_seq_nx) - 1) # duplicate last elem because right bound is not included
            node_idx_start, edge_idx_start = 0, 0
            description = {}
            for idx_l, idx_r in pairwise(pairwise_array):
                pyg_graph = from_networkx(sg_seq_nx[idx_l])
                description_local = textualize_graph(sg_seq_nx[idx_l])
                description[(seq_frame_names[idx_l], seq_frame_names[idx_r])] = description_local

                node_idx_end = len(pyg_graph.label)
                pyg_graph.x = node_embed[node_idx_start:node_idx_start + node_idx_end]
                node_idx_start += node_idx_end
                
                pe_transform = AddLaplacianEigenvectorPE(k=min(pyg_graph.x.shape[0] - 1, LPE_NUM), attr_name="laplacian_eigenvector_pe")
                pyg_graph = pe_transform(pyg_graph)
                pe = pyg_graph.laplacian_eigenvector_pe
                if pe.size(1) < LPE_NUM:
                    num_missing = LPE_NUM - pe.size(1)
                    pad = pe.new_zeros(pe.size(0), num_missing)
                    pe = torch.cat([pe, pad], dim=1)
                pyg_graph.x = torch.cat([pyg_graph.x, pe], dim=-1) # n, 1024 + LPE_NUM

                if hasattr(pyg_graph, "edge_label"):
                    edge_idx_end = len(pyg_graph.edge_label)
                    pyg_graph.edge_attr = edge_embed[edge_idx_start:edge_idx_start + edge_idx_end]
                    edge_idx_start += edge_idx_end
                else:
                    pyg_graph.edge_attr = torch.zeros((0, 1024))
                    pyg_graph.edge_label = []
                
                # we have to pad edge features too to match extended by LPE_NUM g.x
                pad = pyg_graph.edge_attr.new_zeros(pyg_graph.edge_attr.size(0), LPE_NUM)
                pyg_graph.edge_attr = torch.cat([pyg_graph.edge_attr, pad], dim=-1)

                # left range include, right exclude (except last)
                all_pyg_graphs[(seq_frame_names[idx_l], seq_frame_names[idx_r])] = pyg_graph
                assert len(all_pyg_graphs) > 0
            
            assert node_idx_start == len(node_embed)
            assert edge_idx_start == len(edge_embed)

            # Do not retain a second full copy of every graph tensor in process
            # RAM when the serialized graphs already occupy tmpfs RAM.
            SG_GLOBAL[(split, sg_name)] = tuple(all_pyg_graphs)
            torch.save(all_pyg_graphs, f"{root_path}/preprocessed_{MODEL_NAME}/{split}/graphs/{sg_name}.pt")
            with open(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/descs/{sg_name}.pkl", "wb") as f:
                pickle.dump(description, f)
        # Assignment to sg_data on the next iteration would otherwise unpickle
        # the next whole split while the previous split is still referenced.
        del sg_data
        gc.collect()


def ground_qa_item(qa_item, all_ranges):
    """Preserve upstream interval membership, duplicate ranges, and order."""
    grounding_idx = sorted(load_grounding_frames(qa_item["sg_grounding"]),
                           key=lambda x: int(x[-6:]))
    unique_g_idx = list()
    if grounding_idx:
        for key_range in all_ranges:
            for d_idx in grounding_idx:
                if int(d_idx.lstrip()) in range(int(key_range[0].lstrip()), int(key_range[1].lstrip())):
                    unique_g_idx.append(key_range)
        # range excludes the right endpoint; the terminal interval is special.
        if grounding_idx[-1] == all_ranges[-1][0]:
            unique_g_idx.append(all_ranges[-1])
    else:
        unique_g_idx = list(all_ranges)
    result = sorted(unique_g_idx, key=lambda x: int(x[0]))
    if not result:
        raise ValueError(f"No scene graphs grounded for QA from video {qa_item['video_id']!r}")
    return result


def preprocess_qa(splits=("train", "test")):
    for split in splits:
        os.makedirs(f"{root_path}/preprocessed_{MODEL_NAME}/{split}", exist_ok=True)
        qa_data_path = f"{root_path}/data/AGQA_balanced/{split}_balanced.txt"

        def grounding_for_item(qa_item):
            return ground_qa_item(qa_item, SG_GLOBAL[(split, qa_item["video_id"])])

        if os.environ.get("DYGENC_INDEXED_QA", "0") == "1":
            logger.info(f"Streaming {split} QA into bounded-heap SQLite index")
            index_path = f"{root_path}/preprocessed_{MODEL_NAME}/{split}/{INDEX_NAME}"
            count = build_qa_index(index_path, tqdm(iter_qa_json(qa_data_path)), grounding_for_item)
            logger.info(f"Indexed {count} QA records in source order: {index_path}")
        else:
            logger.info("Loading QA (legacy eager mode)")
            with open(qa_data_path, mode='r', encoding='utf8') as file:
                qa_json = json.load(file)
            QA2SQ = {qa_key: grounding_for_item(qa_item) for qa_key, qa_item in tqdm(qa_json.items())}
            with open(f"{root_path}/preprocessed_{MODEL_NAME}/{split}/qa2sg.pkl", 'wb') as f:
                pickle.dump(QA2SQ, f)
            del qa_json, QA2SQ
        gc.collect()

if __name__ == "__main__":
    set_seed(int(os.environ.get("DYGENC_PREPROCESS_SEED", "18")))
    # Finish and release each split before loading the next monolithic SG
    # pickle. Indexed QA streams one record at a time instead of json.load.
    for split in ("train", "test"):
        preprocess_graphs((split,))
        preprocess_qa((split,))
        SG_GLOBAL.clear()
        gc.collect()
