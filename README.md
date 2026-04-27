# Structure-Tracing Evidence Mining (STEM)

#### Original Code for STEM:"STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation"

We propose Structure-Tracing Evidence Mining (STEM), a novel framework that reframes multi-hop reasoning as a schema-guided graph search task. First, we design a Semantic-to-Structural Projection pipeline that leverages KG structural priors to decompose queries into atomic relational assertions and construct an adaptive query schema graph. Subsequently, we execute globally-aware node anchoring and subgraph retrieval to obtain the final evidence reasoning graph from KG. To more effectively integrate global structural information during the graph construction process, we design a Triple-Dependent GNN (Triple-GNN) to generate a Global Guidance Subgraph (Guidance Graph) that guides the construction. STEM significantly improves both the accuracy and evidence completeness of multi-hop reasoning graph retrieval, and achieves State-of-the-Art performance on multiple multi-hop benchmarks.


## Requirements
```
pip install -r requirements.txt
redis==7.2.4
```

> [!IMPORTANT]
To facilitate immediate research exchange and community engagement, we have released the core logic and key components of our framework. We are currently refactoring and streamlining our original research codebase—which was initially developed for internal testing—to ensure it is user-friendly and easy to reproduce. The complete implementation, including end-to-end inference scripts, evaluation pipelines, and module training code, will be released incrementally in the coming weeks. We appreciate your patience as we work to provide a high-quality, clean implementation for the community. Stay tuned!

Roadmap:
> - [x] **Core Structure:** SGDA & SAGB logic and T-GNN implementation.
> - [x] **RGD Scripts:** Structure-to-Query Reverse Generation data workflow execution script.
> - [ ] **Model Weights:** SGDA & SAGB logic and T-GNN. 
> - [ ] **End-to-End Inference**
> - [ ] **Evaluation Scripts**
> - [ ] **Training Pipeline**

## Retrieval & Inference Excecutation

Run main.py to initiate subgraph retrieval + LLM prediction, and save the predicted answers in the path specified by the "output" parameter: 
```
python ./main.py \
        --config ../config.yaml \
        --mode build \
        --split test \
        --n_proc 1 \
        --output ./predictions.jsonl
```

## Evaluation Metrics (Hit@1 and F1)
Run eval_results.py to calculate metric scores based on the generated predictions file.
```
python ./eval_results.py \
        --d ./predictions.jsonl \
        --cal_f1 True \
        --top_k 10
```

## LLM-Driven Reverse Generation: Random Walk with Masking for Subgraph Sampling
```
python ./scripts/run_graph_sampling.py \
        --d webqsp \
        --output_path ../masked_sample_subgraphs.jsonl \
        --n 50
```

## Training

### SGDA & SAGB Training Execution
The training of SGDA​ and SAGB​ relies on the **[LlamaFactory](https://github.com/hiyouga/LlamaFactory)** framework (version 0.9.3). You need to place their respective training data .jsonlfiles into the llamafactory dataset directory and add corresponding entries in the dataset_info.jsonlfile. Then execute the following commands:

For SGDA, execute:
```
llamafactory-cli train ./train/train_config_sgda.yaml
```
For SAGB, execute:
```
llamafactory-cli train ./train/train_config_sagb.yaml
```
### T-GNN Training Execution
Run the following command to launch T-GNN training：
```
python ./train/tgnn_train.py \
        --config ../config.yaml \
        --train_data_path ../rgd_train_20k.jsonl
```
