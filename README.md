# NSDTI
Nested Feature Decoupling and Semantic Affinity for Drug–Target Interaction Prediction
NSDTI is a novel framework for drug–target interaction (DTI) prediction that leverages:

🧱 A tree-based nested feature decoupling algorithm to hierarchically partition high-dimensional node embeddings,

🔗 A semantic-affinity-driven subgraph interaction mechanism for fine-grained modeling between drugs and targets,

🧠 A self-attention-enhanced fusion module to produce accurate DTI scores.
NSDTI/
├── code/               # Model, training, and evaluation code
│   ├── models/         # GNNs and fusion modules
│   ├── utils/          # Helper functions
│   ├── train.py        # Main training script
├── dataset/            # Preprocessed data and loading scripts
│   └── human/          # Human dataset
├── README.md
└── requirements.txt

---

## ⚙️ Installation

We recommend using `conda`:

```bash
conda create -n nsdti python=3.8
conda activate nsdti
pip install -r requirements.txt

2. Train the Model
cd code
python train.py --config config_biosnap.yaml
