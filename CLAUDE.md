# Meta-IFD Project

  Master thesis: Ethereum fraud detection (Ponzi + Phishing) using unified multi-task GNN.

  ## Architecture
  - UnifiedHMSL in unified_model.py: multi-task heterogeneous GNN
  - Two node types: CA (Contract Accounts) → Ponzi, EOA (Externally Owned Accounts) → Phishing
  - Augmentation: CVAE pretrained models in ./pretrain_model/
  - Key components: CrossPathAttention + TaskGate + ExpertRules + TripletLoss

  ## Train command
  python run_unified.py --expert_mode feature --aug_method cvae --hidden 128 --epochs 1000 --lr 0.001 --batch_size 512 --gpu 0