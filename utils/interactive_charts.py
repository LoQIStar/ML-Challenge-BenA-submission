import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def create_interactive_comparison():
    """Creates an interactive comparison dashboard"""
    # Create sample data
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    throughput = [
        100, 180, 320, 550, 900, 1400, 2000, 2400  # imgs/sec
    ]
    memory_usage = [
        1.2, 1.4, 1.8, 2.3, 3.1, 4.5, 7.2, 12.8  # GB
    ]
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=batch_sizes, y=throughput, name="Throughput", mode='lines+markers')
    )
    fig.add_trace(
        go.Scatter(x=batch_sizes, y=memory_usage, name="Memory", 
                  yaxis="y2", mode='lines+markers')
    )
    
    # Add figure layout
    fig.update_layout(
        title="Performance vs Resource Usage",
        xaxis_title="Batch Size",
        yaxis_title="Throughput (imgs/sec)",
        yaxis2=dict(
            title="Memory Usage (GB)", 
            overlaying="y", 
            side="right"
        ),
        hovermode='x'
    )
    
    # Ensure the docs/assets directory exists
    import os
    os.makedirs('docs/assets', exist_ok=True)
    
    # Save as HTML
    fig.write_html("docs/assets/interactive_comparison.html")
    print("Interactive chart saved to docs/assets/interactive_comparison.html")

if __name__ == "__main__":
    create_interactive_comparison()