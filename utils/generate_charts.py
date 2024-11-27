import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":
    create_performance_chart()