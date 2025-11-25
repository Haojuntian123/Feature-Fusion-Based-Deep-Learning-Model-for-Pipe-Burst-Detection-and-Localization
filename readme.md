
```markdown
# Feature Fusion Based Deep Learning Model for Pipe Burst Detection and Localization for Water Distribution Networks

---
## Environment Setup
```bash
# Install dependencies (Python 3.8+)
pip install -r requirements.txt

# Core dependencies:
# torch==1.12.1 | wntr==0.4.1 | pandas==1.5.0
# scikit-learn==1.1.2 | numpy==1.23.5
```

```python
# === MANDATORY CONFIGURATION ===
# Create config.py in project root with these fields:

# System Parameters
SEED = 66                     # Fix random seeds
MODEL_PATH = 'output/models/' # Model checkpoint directory
DATA_PATH = 'datas/processed/'# Preprocessed data location

# Hydraulic Settings (customize per network)
EPANET_FILE = 'resources/inp/network.inp'  # .inp model path
MONITOR_SCHEME = 'resources/cluster/monitor-scheme.xlsx' # Partitioning matrix
SIMULATION_DURATION = 48      # Simulation hours
BURST_LEVELS = [0.2, 0.5, 0.8]# Leak coefficients
```

---
## Workflow Architecture
```plaintext
project_root/
├── requirements.txt
├── datas/                      Data simulation 
│   ├── create_data.py         # Generate burst scenarios (EPANET)
│   ├── process_data.py        # Create spatio-temporal tensors
│   ├── dataset.py             # PyTorch Dataset pipelines
│   ├── operate_inp.py         # Configure PDD dynamics
│   └── analyze_data.py        # Hydraulic metric analysis
├── evaluation_indicators/     # Training scripts 
│   ├── run_detection.py       # Train baseline models
│   ├── run_improve_detection.py # Train enhanced detectors
│   ├── run_improve_location.py # Train enhanced locators
│   └── run_location.py        # Baseline localization
├── model/                     #Baseline models 
│   ├── CNN.py                 # Vanilla convolutional model
│   ├── FADenseNet.py          # Feature-Attention DenseNet
│   └── RNN.py                 # Standard recurrent network
├── Improve_model/             #Enhanced models 
│   ├──ImproveFeatures.py     # Feature augmentation modules
│   ├── ICNN.py                # Improved CNN with cross-sensor attention
│   ├── IFADenseNet.py         # FA-DenseNet + pressure-flow fusion
│   └── IRNN.py                # RNN with LAQ attention [Key innovation]
└── resources/                 # Hydraulic assets
    ├── inp/                   # EPANET .inp files (required)
    └── cluster/               # Topology files
        ├── monitor-scheme.xlsx # Node partitioning matrix
        └── pipe_zones.csv     # Diameter-based clusters
```

---
##  Technical Modules
### 1. Hydraulic Simulation Engine (`datas/create_data.py`)
```python
def simulate_pipe_burst(epanet_file: str, burst_nodes: list):
    """
    Generates burst scenarios using pressure-driven EPANET modeling
  
    Args:
        epanet_file: Path to .inp network file
        burst_nodes: Target pipe burst locations
  
    Returns:
        pressure_data: (time_steps × nodes)
        flow_data: (time_steps × pipes)
        labels: Binary burst indicators
    """
    wn = wntr.network.WaterNetworkModel(epanet_file)
    wn.options.hydraulic.demand_model = 'PDD'  # Pressure-driven mode
  
    # Add burst events at designated nodes
    for node in burst_nodes:
        burst = wntr.network.elements.PipeBurst(node, diameter=0.1)
        wn.add_burst(burst)
  
    # Execute transient simulation
    sim = wntr.sim.EpanetSimulator(wn)
    return sim.run_sim()
```

### 2. LAQ Attention Module (`Improve_model/ImproveFeatures.py`)
```python
class LAQ(nn.Module):
    """
    Location-Aware flow Attention (LAQ): Fuses pressure observations 
    with hydraulic flow context using graph adjacency
  
    Architecture:
        - Flow-pressure gating mechanism
        - Neighborhood feature aggregation
        - Residual attention scaling
    """
    def __init__(self, input_dim: int, hidden_dim: int, adj_matrix: torch.Tensor):
        super().__init__()
        self.adj = adj_matrix  # Node connectivity (n×n)
        self.flow_gate = nn.Linear(input_dim, hidden_dim)
        self.pressure_transform = nn.Linear(input_dim, hidden_dim)

    def forward(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """ 
        P: Pressure tensor (batch × nodes × time)
        Q: Flow tensor (batch × pipes × time)
        """
        # Gated flow features
        flow_weights = torch.sigmoid(self.flow_gate(Q)) 
        # Neighborhood aggregation
        aggregated = torch.matmul(self.adj, P)
        return aggregated * flow_weights  # Context-weighted outputs
```

### 3. Enhanced RNN Locator (`Improve_model/IRNN.py`)
```python
class IRNN(nn.Module):
    """
    Burst localization model integrating:
    - Sequence modeling (GRU)
    - Hydraulic context attention (LAQ)
    - Multi-scale feature extraction
    """
    def __init__(self, adjacency_matrix: torch.Tensor, feature_dim: int):
        self.laq = LAQ(feature_dim, 32, adjacency_matrix)
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=64, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, adjacency_matrix.size(0))  # Nodes
        )
  
    def forward(self, pressure: torch.Tensor, flow: torch.Tensor):
        enhanced = self.laq(pressure, flow)
        temporal, _ = self.gru(enhanced)
        return self.classifier(temporal[:, -1, :])  # Final state -> node probs
```


---
## License
MIT License - See [LICENSE](LICENSE) for full terms.
```

