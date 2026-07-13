import matplotlib.pyplot as plt

# --- Color Palette ---
COLOR_CPU_JIT = "#1b7837"    # Forest green 
COLOR_CPU_NO_JIT = "#e66101" # Vibrant orange 
COLOR_GPU = "#00a2f3"        # Bright blue
GRID_COLOR = "#cccccc"       # Light gray

# --- Data ---
data_jit = [
    ("Naive", 794.34), ("Tile 8", 214.82), ("Transpose Naive", 117.26),
    ("Locality Exploitation", 87.05), ("Tile 256", 73.93), ("Tile 16", 73.65),
    ("Tile 128", 70.06), ("CuPy (GPU)", 69.95), ("Tile 512", 64.53),
    ("Tile 32", 63.54), ("Tile 1024", 63.41), ("Tile 64", 62.58)
]

data_no_jit = [
    ("Tile 8", 42781.91), ("Naive", 41834.25), ("Tile 16", 40548.84),
    ("Locality Exploitation", 40259.05), ("Tile 1024", 40025.15), ("Transpose Naive", 39726.54),
    ("Tile 32", 39608.49), ("Tile 256", 39515.65), ("Tile 128", 39512.94),
    ("Tile 512", 39332.61), ("Tile 64", 38973.51), ("CuPy (GPU)", 69.95)
]

def create_horizontal_bar_chart(data, title, filename, max_x_lim, cpu_color, is_jit=True):
    # Reverse to plot from top to bottom
    labels, values = zip(*reversed(data))
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # Strictly enforce color mapping per bar
    colors = []
    for label in labels:
        if label == "CuPy (GPU)":
            colors.append(COLOR_GPU)
        else:
            colors.append(cpu_color) # Forces green for JIT, orange for non-JIT
            
    # Plot bars
    bars = ax.barh(labels, values, color=colors, height=0.75)
    
    # Styling and Grid (Regular title weight matches original Non-JIT image style)
    title_weight = 'bold' if is_jit else 'normal'
    ax.set_title(title, fontsize=14, pad=20, fontweight=title_weight)
    ax.set_xlabel("Execution Time (ms) - Lower is Better", fontsize=12, labelpad=10)
    ax.grid(axis='x', linestyle='--', color=GRID_COLOR, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    
    # Set boundaries
    ax.set_xlim(0, max_x_lim)
    
    # Add text labels next to the bars
    for bar in bars:
        width = bar.get_width()
        label_text = f"{width:,.2f} ms" if width > 1000 else f"{width:.2f} ms"
        
        ax.annotate(
            label_text,
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),  
            textcoords="offset points",
            ha='left', va='center',
            fontsize=10
        )
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# --- Execution ---

# 1. Generate With JIT (Forces Green)
create_horizontal_bar_chart(
    data_jit, 
    "Detailed Comparison: Highly Optimized CPU (With JIT) vs. CuPy (GPU)", 
    "ConfrontJIT_unified.png", 
    max_x_lim=910,
    cpu_color=COLOR_CPU_JIT,
    is_jit=True
)

# 2. Generate Without JIT (Forces Orange)
create_horizontal_bar_chart(
    data_no_jit, 
    "Detailed Comparison: CPU (Without JIT) vs. CuPy (GPU)", 
    "ConfrontoNon-JIT_unified.png", 
    max_x_lim=49000,
    cpu_color=COLOR_CPU_NO_JIT,
    is_jit=False
)