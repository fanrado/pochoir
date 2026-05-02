#!/usr/bin/env python3
"""
Visualize FDM Domain

This script visualizes the finite-difference method domain including:
- Initial value array
- Boundary conditions array
- Domain geometry
- Edge conditions (fixed/periodic)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse


def load_array(filepath):
    """Load a numpy array from file."""
    try:
        file_ =  np.load(filepath)
        keys = list(file_.keys())[0]
        print(f"Loaded .npy file with keys: {list(file_.keys())}")
        return file_[keys]
    except:
        # Try loading as text if .npy fails
        return np.loadtxt(filepath)


def parse_edges(edge_string, ndim):
    """Parse edge condition string into boolean list."""
    edges = edge_string.split(",")
    bool_edges = [e.strip().startswith("per") for e in edges]
    
    if len(bool_edges) != ndim:
        raise ValueError(f"Number of edge conditions ({len(bool_edges)}) "
                        f"doesn't match dimensions ({ndim})")
    return bool_edges


def visualize_2d_domain(initial, boundary, edge_conditions, domain_info=None):
    """Visualize 2D FDM domain."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Initial values
    im1 = axes[0].imshow(initial, cmap='viridis', origin='lower')
    axes[0].set_title('Initial Values')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # Add grid
    axes[0].grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Plot 2: Boundary conditions
    im2 = axes[1].imshow(boundary, cmap='RdBu_r', origin='lower')
    axes[1].set_title('Boundary Conditions')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    axes[1].grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Plot 3: Combined view with boundary mask
    combined = initial.copy()
    boundary_mask = np.abs(boundary) > 0
    combined[boundary_mask] = boundary[boundary_mask]
    
    im3 = axes[2].imshow(combined, cmap='plasma', origin='lower')
    axes[2].set_title('Combined (Boundary overwrites Initial)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    # Highlight boundary cells
    for i in range(boundary.shape[0]):
        for j in range(boundary.shape[1]):
            if boundary_mask[i, j]:
                rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                               fill=False, edgecolor='red', linewidth=2)
                axes[2].add_patch(rect)
    
    # Add edge condition labels
    edge_labels = []
    directions = ['X', 'Y']
    for i, (is_periodic, direction) in enumerate(zip(edge_conditions, directions)):
        condition = 'Periodic' if is_periodic else 'Fixed'
        edge_labels.append(f'{direction}: {condition}')
    
    fig.suptitle(f'FDM Domain Visualization\nEdge Conditions: {", ".join(edge_labels)}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def visualize_1d_domain(initial, boundary, edge_conditions):
    """Visualize 1D FDM domain."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    x = np.arange(len(initial))
    
    # Plot 1: Initial and boundary values
    axes[0].plot(x, initial, 'o-', label='Initial Values', markersize=8)
    axes[0].plot(x, boundary, 's-', label='Boundary Conditions', markersize=8)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Initial and Boundary Values')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Boundary mask
    boundary_mask = np.abs(boundary) > 0
    axes[1].bar(x, boundary_mask, color='red', alpha=0.6, label='Boundary Cells')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Is Boundary')
    axes[1].set_title('Boundary Cell Locations')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add edge condition label
    condition = 'Periodic' if edge_conditions[0] else 'Fixed'
    fig.suptitle(f'1D FDM Domain Visualization\nEdge Condition: {condition}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def visualize_3d_domain(initial, boundary, edge_conditions):
    """Visualize 3D FDM domain using slice views."""
    fig = plt.figure(figsize=(16, 10))
    
    # Get middle slices
    mid_z = initial.shape[0] // 2
    mid_y = initial.shape[1] // 2
    mid_x = initial.shape[2] // 2
    
    # Create 3x2 subplot grid
    # Row 1: Initial values
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(initial[mid_z, :, :], cmap='viridis', origin='lower')
    ax1.set_title(f'Initial: XY slice (Z={mid_z})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(initial[:, mid_y, :], cmap='viridis', origin='lower')
    ax2.set_title(f'Initial: XZ slice (Y={mid_y})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(initial[:, :, mid_x], cmap='viridis', origin='lower')
    ax3.set_title(f'Initial: YZ slice (X={mid_x})')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    plt.colorbar(im3, ax=ax3)
    
    # Row 2: Boundary conditions
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(boundary[mid_z, :, :], cmap='RdBu_r', origin='lower')
    ax4.set_title(f'Boundary: XY slice (Z={mid_z})')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(boundary[:, mid_y, :], cmap='RdBu_r', origin='lower')
    ax5.set_title(f'Boundary: XZ slice (Y={mid_y})')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Z')
    plt.colorbar(im5, ax=ax5)
    
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(boundary[:, :, mid_x], cmap='RdBu_r', origin='lower')
    ax6.set_title(f'Boundary: YZ slice (X={mid_x})')
    ax6.set_xlabel('Y')
    ax6.set_ylabel('Z')
    plt.colorbar(im6, ax=ax6)
    
    # Add edge condition labels
    edge_labels = []
    directions = ['X', 'Y', 'Z']
    for i, (is_periodic, direction) in enumerate(zip(edge_conditions, directions)):
        condition = 'Periodic' if is_periodic else 'Fixed'
        edge_labels.append(f'{direction}: {condition}')
    
    fig.suptitle(f'3D FDM Domain Visualization (Cross-sections)\n'
                f'Edge Conditions: {", ".join(edge_labels)}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize FDM domain from input arrays')
    
    parser.add_argument('-i', '--initial', type=str, required=True,
                       help='Path to initial value array file (.npy or .txt)')
    parser.add_argument('-b', '--boundary', type=str, required=True,
                       help='Path to boundary array file (.npy or .txt)')
    parser.add_argument('-e', '--edges', type=str, required=True,
                       help='Comma separated list of "fixed" or "periodic" edge conditions')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output file path for saving the plot (optional)')
    parser.add_argument('--dpi', type=int, default=100,
                       help='DPI for saved figure (default: 100)')
    
    args = parser.parse_args()
    
    # Load arrays
    print(f"Loading initial values from: {args.initial}")
    initial = load_array(args.initial)
    
    print(f"Loading boundary conditions from: {args.boundary}")
    boundary = load_array(args.boundary)
    
    # Validate shapes match
    if initial.shape != boundary.shape:
        raise ValueError(f"Shape mismatch: initial {initial.shape} "
                        f"vs boundary {boundary.shape}")
    
    print(f"Domain shape: {initial.shape}")
    print(f"Domain dimensions: {initial.ndim}D")
    
    # Parse edge conditions
    edge_conditions = parse_edges(args.edges, initial.ndim)
    print(f"Edge conditions: {args.edges}")
    
    # Create visualization based on dimensionality
    if initial.ndim == 1:
        fig = visualize_1d_domain(initial, boundary, edge_conditions)
    elif initial.ndim == 2:
        fig = visualize_2d_domain(initial, boundary, edge_conditions)
    elif initial.ndim == 3:
        fig = visualize_3d_domain(initial, boundary, edge_conditions)
    else:
        raise ValueError(f"Unsupported dimensionality: {initial.ndim}D")
    
    # Save or show
    if args.output:
        print(f"Saving visualization to: {args.output}")
        fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        print("Done!")
    else:
        print("Displaying visualization...")
        plt.show()


if __name__ == '__main__':
    main()