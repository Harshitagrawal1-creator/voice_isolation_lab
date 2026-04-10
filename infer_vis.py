import argparse
import torch
import sys
from pathlib import Path

# Local imports from your project
from src.models.eend_ss import EENDSS
from src.training.config import get_config
from src.utils.visualizer import EENDVisualizer
from infer import load_audio # Reusing your existing audio loader

def run_visual_inference(args):
    # Set device to CPU for visualization to ensure stability with hooks
    device = torch.device("cpu")
    cfg = get_config()
    
    # 1. Initialize Model
    model = EENDSS(
        N=cfg.N, L=cfg.L, B=cfg.B, H=cfg.H, P=cfg.P,
        X=cfg.X, R=cfg.R, C=cfg.C,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, d_ff=cfg.d_ff,
        dropout=cfg.dropout, subsample=cfg.subsample,
    )
    
    # 2. Load Checkpoint
    print(f"Loading weights from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k: v for k, v in ckpt["model_state"].items() if 'pos_enc.pe' not in k}
    model.load_state_dict(state, strict=False)
    model.eval()

    # 3. Setup Glowing Visualizer
    visualizer = EENDVisualizer(output_dir=args.output_dir)
    visualizer.attach_hooks(model)

    # 4. Process Audio
    mixture = load_audio(Path(args.input)).unsqueeze(0).to(device)
    
    print("Capturing neural activations...")
    with torch.no_grad():
        # This triggers the 'glowing' hooks
        _ = model.inference(mixture, threshold=args.threshold)

    # 5. Render and Save
    visualizer.render_glowing_map()
    visualizer.create_animation("active_neural_map.gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input .wav")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output_dir", default="outputs/neural_glow", help="Where to save the GIF")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    run_visual_inference(args)