MINDGAN: EEG-Based Motor Imagery Neural Decoding with GAN-Augmented Hybrid Deep Learning
MINDGAN is a unified deep learning framework designed to address two major challenges in motor imagery (MI) EEG classification:
1. Data scarcity in EEG recordings
2. Inter‑subject variability that limits generalisation
The framework integrates:
• A conditional DCGAN (cDCGAN) for class‑conditioned EEG data augmentation
• A hybrid CNN–Transformer classifier for local–global spatiotemporal feature extraction
• A three‑phase curriculum training schedule for stable optimisation
Key Features
• EEGNet‑inspired depthwise separable convolutions
• Six‑block pre‑LayerNorm Transformer encoder
• Wasserstein loss with gradient penalty
• Spectral normalisation + class‑conditional batch norm
• EMA‑based synthetic sample quality filtering
• Class‑balanced replay buffer
• Ablation study showing LSTM layers degrade performance
• Real‑time prototype using Emotiv EPOC X
Performance
Evaluated on BCI Competition IV datasets:
• 2A (4‑class MI): 81.17% mean accuracy
• 2B (binary MI): 86.87% mean accuracy
• GAN‑generated EEG achieves r = 0.9923 spectral fidelity
• Augmentation benefit strongly depends on per‑class data availability
Included in this Repository
• Full implementation of MINDGAN
• Training scripts for 2A and 2B
• Preprocessing pipeline
• Synthetic EEG generation code
• Results, logs, and trained models
