import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0
    IMAGE_SIZE = (300, 300)
    INPUT_CHANNELS = 3
    NUM_CLASSES = 2
    CLASS_NAMES = ['Normal', 'Glaucoma']
    
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)
    GREEN_CHANNEL_WEIGHT = 0.6
    
    BACKBONE_NAME = 'efficientnet_b3'
    PRETRAINED = True
    FINE_TUNE = True
    BACKBONE_OUT_CHANNELS = 1536
    
    GRID_SIZE = (8, 8)
    NUM_NODES = 64
    NODE_EMB_DIM = 256
    INCLUDE_SPATIAL_COORDS = True
    
    NUM_SPATIAL_HYPEREDGES = 16
    NUM_SEMANTIC_HYPEREDGES = 16
    NUM_VASCULAR_HYPEREDGES = 8
    TOTAL_HYPEREDGES = 40
    KNN_K = 6
    
    HGNN_HIDDEN_DIM = 128
    HGNN_OUT_DIM = 64
    HGNN_NUM_LAYERS = 2
    HGNN_DROPOUT = 0.2
    
    GUMBEL_TEMPERATURE = 1.0
    GUMBEL_MIN_TEMP = 0.2
    GUMBEL_ANNEAL_RATE = 0.95
    GUMBEL_HARD = False
    SUBGRAPH_TOP_K = 12
    
    CLINICAL_CONCEPTS = [
        'optic_disc_morphology',
        'optic_cup_characteristics',
        'cup_to_disc_ratio_proxy',
        'neuroretinal_rim_integrity',
        'vessel_attenuation_changes'
    ]
    NUM_CONCEPTS = len(CLINICAL_CONCEPTS)
    
    VLT_HIDDEN_DIM = 256
    VLT_NUM_HEADS = 4
    VLT_NUM_LAYERS = 2
    VLT_MAX_SEQ_LEN = 128
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    SPARSITY_LOSS_WEIGHT = 0.05
    ENTROPY_LOSS_WEIGHT = 0.01
    CONCEPT_LOSS_WEIGHT = 0.1
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    ODIR_DIR = os.getenv('ODIR_DIR', os.path.join(DATA_DIR, 'ODIR'))
    G1020_DIR = os.getenv('G1020_DIR', os.path.join(DATA_DIR, 'G1020'))
    PAPILA_DIR = os.getenv('PAPILA_DIR', os.path.join(DATA_DIR, 'PAPILA'))
    AIROGS_DIR = os.getenv('AIROGS_DIR', os.path.join(DATA_DIR, 'AIROGS'))
    RIMONE_DIR = os.getenv('RIMONE_DIR', os.path.join(DATA_DIR, 'RIM-ONE'))
    SYNTHETIC_DIR = os.path.join(DATA_DIR, 'synthetic_fundus')
    
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    BASELINE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'baseline_cnn.pth')
    
    UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    GENERATED_DIR = os.path.join(BASE_DIR, 'static', 'generated')
    
    DISCLAIMER_TEXT = (
        'AI-assisted screening result. This system is a research prototype and is '
        'not a substitute for professional ophthalmological examination or clinical diagnosis.'
    )
