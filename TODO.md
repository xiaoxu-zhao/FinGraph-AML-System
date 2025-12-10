# Project To-Do List

## 🚀 Immediate Priorities (v1.0 Release)
- [ ] **Deploy to AWS App Runner**
    - Follow steps in `docs/DEPLOYMENT_GUIDE.md`.
    - Push Docker image to AWS ECR.
    - Create App Runner service.

## 🔮 Future Roadmap (v2.0)
- [ ] **Multi-Class Typology Prediction**
    - Upgrade model from Binary Classification (0/1) to Multi-Class.
    - Target classes: Smurfing, Cyclic Laundering, Scatter-Gather.
    - Update Loss function to `CrossEntropyLoss`.
- [ ] **Temporal Graph Learning**
    - Implement Temporal GNNs (TGN) to capture time-evolving patterns.
- [ ] **Explainable AI (XAI)**
    - Integrate GNNExplainer to visualize *why* a node was flagged.
