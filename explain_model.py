from torch_geometric.explain import Explainer, GNNExplainer


def explain_model(model, test_dataset, device, feat_path=None, top_k=10, subgraph_path=None):
    sample_data = test_dataset[0].to(device)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    node_index = min(0, sample_data.x.shape[0] - 1)
    explanation = explainer(
        x=sample_data.x,
        edge_index=sample_data.edge_index,
        batch=getattr(sample_data, "batch", None),
        index=node_index
    )

    print(f'Generated explanations in {explanation.available_explanations}')

    if feat_path is not None:
        explanation.visualize_feature_importance(feat_path, top_k=top_k)
        print(f"Feature importance plot has been saved to '{feat_path}'")

    if subgraph_path is not None:
        explanation.visualize_graph(subgraph_path)
        print(f"Subgraph visualization plot has been saved to '{subgraph_path}'")
