import plotly.express as px
import plotly.graph_objects as go

def create_interactive_comparison():
    """Creates an interactive comparison dashboard"""
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=batch_sizes, y=throughput, name="Throughput")
    )
    fig.add_trace(
        go.Scatter(x=batch_sizes, y=memory_usage, name="Memory", yaxis="y2")
    )
    
    # Add figure layout
    fig.update_layout(
        title="Performance vs Resource Usage",
        xaxis_title="Batch Size",
        yaxis_title="Throughput (imgs/sec)",
        yaxis2=dict(title="Memory Usage (MB)", overlaying="y", side="right")
    )
    
    # Save as HTML
    fig.write_html("docs/assets/interactive_comparison.html")