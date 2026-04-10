import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import imageio

class EENDVisualizer:
    def __init__(self, output_dir="outputs/neural_map"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.activations = {}
        self.frames = []

    def _get_hook(self, name):
        def hook(model, input, output):
            # Capture the output tensor from the layer
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
        return hook

    def attach_hooks(self, model):
        """Recursively attaches to Linear and Convolutional layers."""
        for name, module in model.named_modules():
            # We target layers that represent 'learned' features
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                module.register_forward_hook(self._get_hook(name))

    def render_glowing_map(self):
        """Renders neurons as individual glowing dots on a black background."""
        if not self.activations:
            print("No activations captured! Ensure model.inference() was called.")
            return

        # Sort layers to maintain the chronological 'flow' of data
        sorted_layers = sorted(self.activations.keys())
        
        print(f"Rendering {len(sorted_layers)} neural frames...")

        for idx, name in enumerate(sorted_layers):
            tensor = self.activations[name].squeeze(0)
            
            # Convert 3D/2D tensors into a flat 1D array of 'Neuron' strengths
            # We average over the Time dimension to see the total activation for this audio
            if tensor.ndim == 2: # [Channels, Time]
                neuron_values = tensor.mean(dim=-1).numpy()
            else:
                neuron_values = tensor.numpy().flatten()[:400] # Cap at 400 nodes for visibility

            # Normalize values to 0.0 - 1.0 for the color map
            norm_vals = (neuron_values - neuron_values.min()) / (neuron_values.max() - neuron_values.min() + 1e-8)

            # Arrange neurons in a square grid
            grid_size = int(np.ceil(np.sqrt(len(norm_vals))))
            yy, xx = np.mgrid[:grid_size, :grid_size]
            x = xx.flatten()[:len(norm_vals)]
            y = yy.flatten()[:len(norm_vals)]

            # Create the dark 'Neural Map'
            plt.figure(figsize=(10, 10), facecolor='black')
            ax = plt.gca()
            ax.set_facecolor('black')

            # 1. Plot 'inactive' neurons as faint dots
            ax.scatter(x, y, color='#1a1a1a', s=100, edgecolors='#333333', linewidths=0.5)

            # 2. Plot 'active' neurons with a glowing gradient
            # 'inferno' map goes from black -> purple -> orange -> yellow
            sc = ax.scatter(x, y, c=norm_vals, s=100, cmap='inferno', alpha=0.9)

            # 3. Add the 'Glow' (Halo) for neurons with high activation (> 0.7)
            hot_mask = norm_vals > 0.7
            if np.any(hot_mask):
                # Outer glow
                ax.scatter(x[hot_mask], y[hot_mask], s=400, c='orange', alpha=0.2)
                # Inner bright glow
                ax.scatter(x[hot_mask], y[hot_mask], s=800, c='yellow', alpha=0.05)

            plt.title(f"NEURAL FLOW: {name}", color='white', fontsize=16, pad=20)
            plt.axis('off')
            
            # Save the frame
            save_path = self.output_dir / f"glow_{idx:03d}.png"
            plt.savefig(save_path, facecolor='black', bbox_inches='tight')
            self.frames.append(imageio.v2.imread(save_path))
            plt.close()

    def create_animation(self, filename="neural_travel.gif"):
        """Compiles PNG frames into a single GIF animation."""
        if self.frames:
            anim_path = self.output_dir / filename
            imageio.mimsave(str(anim_path), self.frames, fps=3) # 3 layers per second
            print(f"\n*** SUCCESS: Animation saved to {anim_path} ***")