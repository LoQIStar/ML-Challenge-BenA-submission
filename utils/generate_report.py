def generate_metric_report():
    """Generates a comprehensive metric report with all visualizations"""
    # Add model information section
    model_info = """
    <div class="model-info">
        <h2>Model Information</h2>
        <ul>
            <li>Base Architecture: Vision Transformer (ViT-Base/16)</li>
            <li>Available Variants: Original, Quantized, TensorRT</li>
            <li>Model Location: part{1,3}/models/</li>
            <li>Testing Support: Automatic dummy model generation</li>
        </ul>
    </div>
    """
    
    report_template = """
    <html>
        <head>
            <title>Performance Analysis Report</title>
            <style>
                .model-info { padding: 20px; background: #f5f5f5; border-radius: 5px; }
                .chart-container { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Challenge Performance Analysis</h1>
            """ + model_info + """
            <div class="chart-container">
                <img src="assets/performance_chart.png">
                <img src="assets/training_progression.png">
            </div>
            <div class="interactive-charts">
                <iframe src="assets/interactive_comparison.html"></iframe>
            </div>
        </body>
    </html>
    """
    
    with open('docs/performance_report.html', 'w') as f:
        f.write(report_template) 