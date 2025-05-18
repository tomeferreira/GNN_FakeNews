from torch_geometric.explain import Explainer, GNNExplainer
import torch
import matplotlib.pyplot as plt

def explain_model(model, architecture, test_dataset, device, feat_path=None, top_k=10, subgraph_path=None):
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

    if architecture == 'GCN':
        explanation = explainer(
            x=sample_data.x,
            edge_index=sample_data.edge_index,
            batch=sample_data.batch,
            index=node_index
        )
    elif architecture == 'GraphSAGE':
        explanation = explainer(
            x=sample_data.x,
            edge_index=sample_data.edge_index,
            batch=sample_data.batch,
            index=node_index
        )

    print(f'Generated explanations in {explanation.available_explanations}')

    if feat_path is not None:
        explanation.visualize_feature_importance(feat_path, top_k=top_k)
        print(f"Feature importance plot has been saved to '{feat_path}'")

    if subgraph_path is not None:
        explanation.visualize_graph(subgraph_path)
        print(f"Subgraph visualization plot has been saved to '{subgraph_path}'")
    
    feature_names=['Verified','Enable geo-spatial positioning','Followers count','Friends count','Status count','Favorite count','Number of lists','Created time (No. of months since Twitter established)','Number of words in the description','Number of words in the screen name']

    feature_mask = explanation.node_mask  # shape: [num_features]
    
    avg_feature_mask = feature_mask.mean(dim=0)

    # Get top-K most important features (optional)
    topk = 10
    topk_indices = avg_feature_mask.topk(topk).indices.tolist()
    print(topk_indices)
    topk_names = [feature_names[i] for i in topk_indices]

    print(f"Top {topk} important features:")
    topk_scores = avg_feature_mask[torch.tensor(topk_indices)]

    for i, (name, score) in enumerate(zip(topk_names, topk_scores)):
        print(f"{i+1}. {name}: {score.item():.4f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.barh(range(topk), topk_scores.cpu().detach().numpy()[::-1])
    plt.yticks(range(topk), topk_names[::-1])
    plt.xlabel("Average Importance Score")
    plt.title(f"Top-{topk} Node Feature Importances (Averaged)")
    plt.tight_layout()

    plt.show()