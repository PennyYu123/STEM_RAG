from t_gnn import *
import torch
from torch_geometric.data import Data


def entities_to_mask(entities, num_nodes):
    mask = torch.zeros(num_nodes)
    mask[entities] = 1
    return mask

entity_model = QueryNBFNet(
    input_dim = 512,
    hidden_dims = [512, 512, 512, 512, 512, 512],
    message_func = 'distmult',
    aggregate_func = 'sum',
    short_cut = True,
    layer_norm = True
)
graph_retriever = GNNRetriever(entity_model=entity_model, rel_emb_dim=512).to('cuda')

torch.manual_seed(42)

num_nodes = 100
num_edges = 150
num_relations = 30
embedding_dim = 1024

entity_ids = [0, 1, 36]
question_entities_masks = (
        entities_to_mask(entity_ids, num_nodes).unsqueeze(0).to('cuda')
    )

graph_retriever_input = {
        "question_embeddings": torch.randn([1,1024], device='cuda'), 
        "question_entities_masks": question_entities_masks,
    }


edge_index = torch.randint(0, num_nodes, (2, num_edges))
valid_mask = edge_index[0] != edge_index[1]
edge_index = edge_index[:, valid_mask]
edge_type = torch.randint(0, num_relations, (num_edges,))
rel_emb = torch.randn(num_relations, embedding_dim)

graph = Data(
    edge_index=edge_index,
    edge_type=edge_type,
    num_nodes=num_nodes,
    num_relations=num_relations,
    rel_emb=rel_emb 
).to('cuda')

ent_pred = graph_retriever(
            graph, graph_retriever_input
        )

print(ent_pred)
