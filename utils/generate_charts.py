import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def create_performance_chart():
    # Try to use seaborn style, fall back to default if not available
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')  # Fallback to built-in style
        print("Note: Using default style. For better visualizations, install seaborn: pip install seaborn")
    
    # Data
    metrics = ['Inference Time (ms)', 'Memory Usage (MB)', 'Model Size (MB)']
    original = [45.3, 342, 86.4]
    optimized = [28.7, 126, 22.1]
    
    # Calculate positions for bars
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create figure and axis with specified size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars
    rects1 = ax.bar(x - width/2, original, width, label='Original', color='#2C3E50')
    rects2 = ax.bar(x + width/2, optimized, width, label='Optimized', color='#27AE60')
    
    # Customize chart
    ax.set_title('Performance Comparison: Original vs Optimized Model', pad=20, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Add improvement percentages
    improvements = [
        '36.6% ⬇️',
        '63.2% ⬇️',
        '74.4% ⬇️'
    ]
    
    for i, improvement in enumerate(improvements):
        ax.text(i, max(original[i], optimized[i]) + 20, 
                improvement, ha='center', va='bottom',
                color='#E74C3C', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('docs/assets/performance_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_progression_chart():
    """Creates a chart showing training progression across all three parts"""
    plt.figure(figsize=(12, 6))
    
    # Data for each part
    epochs = range(1, 31)
    part1_accuracy = [/* accuracy data */]
    part2_accuracy = [/* accuracy data */]
    part3_accuracy = [/* accuracy data */]
    
    plt.plot(epochs, part1_accuracy, label='Part 1: Quantized Model', marker='o')
    plt.plot(epochs, part2_accuracy, label='Part 2: Optimized Architecture', marker='s')
    plt.plot(epochs, part3_accuracy, label='Part 3: TensorRT Model', marker='^')
    
    plt.title('Training Progression Across All Parts')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/assets/training_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_resource_usage_chart():
    """Creates a chart comparing resource usage across implementations"""
    metrics = ['GPU Memory (GB)', 'CPU Usage (%)', 'Disk I/O (MB/s)']
    part1_resources = [/* resource data */]
    part2_resources = [/* resource data */]
    part3_resources = [/* resource data */]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.bar(x - width, part1_resources, width, label='Part 1')
    plt.bar(x, part2_resources, width, label='Part 2')
    plt.bar(x + width, part3_resources, width, label='Part 3')
    
    plt.title('Resource Usage Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.savefig('docs/assets/resource_usage.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_latency_distribution_chart():
    """Creates a violin plot showing inference latency distribution"""
    plt.figure(figsize=(10, 6))
    
    # Sample data for each implementation
    part1_latencies = [/* latency data */]
    part2_latencies = [/* latency data */]
    part3_latencies = [/* latency data */]
    
    plt.violinplot([part1_latencies, part2_latencies, part3_latencies])
    plt.xticks([1, 2, 3], ['Part 1', 'Part 2', 'Part 3'])
    plt.ylabel('Latency (ms)')
    plt.title('Inference Latency Distribution')
    plt.savefig('docs/assets/latency_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_radar():
    """Create radar chart comparing all implementations"""
    metrics = ['Inference Speed', 'Memory Usage', 'Accuracy', 'Model Size', 'Throughput']
    
    implementations = {
        'Original': [0.6, 0.4, 1.0, 0.3, 0.5],
        'Quantized': [0.8, 0.7, 0.98, 0.7, 0.7],
        'Optimized': [0.9, 0.8, 0.97, 0.8, 0.9]
    }
    
    # Create radar plot using plotly
    fig = go.Figure()
    for name, values in implementations.items():
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            name=name,
            fill='toself'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Implementation Comparison"
    )
    fig.write_html("docs/assets/radar_comparison.html")

def create_training_convergence():
    """Create interactive training convergence visualization"""
    epochs = range(1, 31)
    
    fig = go.Figure()
    
    # Add traces for each implementation
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[/* accuracy data */],
        name="Part 1: Quantized",
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[/* accuracy data */],
        name="Part 2: Hyperparameter Optimized",
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=[/* accuracy data */],
        name="Part 3: TensorRT",
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title="Training Convergence Comparison",
        xaxis_title="Epochs",
        yaxis_title="Accuracy",
        hovermode='x unified'
    )
    
    fig.write_html("docs/assets/training_convergence.html")

if __name__ == "__main__":
    create_performance_chart()