# Graph Neural Networks: Social Spread of Misinformation

Project for the course "Complex Systems" at the University of Coimbra, 2025.

## Authors

- Tomé Filipe Gilberto Ferreira, 2021232615
- Miguel Gameiro Leopoldo, 2021225940

## Brief Description

Online social networks are prime examples of complex systems, where information (credible or not) can spread rapidly through user interactions. In this project, you will explore the emergence of misinformation cascades by leveraging Graph Neural Networks (GNNs) on real-world data. Various publicly available Twitter or Facebook datasets (e.g., FakeNewsNet, available onGitHub) include user–user interactions and post-level content that can be transformed into graph structures. The goal is to analyze how misleading posts propagate, identify influential nodes, and examine conditions underwhich misinformation dominates or fizzles out.

## Goals

From a publicly available social media dataset, represent user interactions (e.g., follows, retweets, replies) as graph edges and embed relevant user or content features as node attributes. The goal is to build a GNN to classify each news item as fake or real, and compare its performance against a baseline. Explore which features (e.g., user credibility, network centrality) the GNN relies on most for detection, and examine whether certain network structures or activity patterns increase the likelihood of fake news proliferation. Finally, discuss any threshold or emergent effects that manifest as misinformation spreads at scale.
