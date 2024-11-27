def generate_metric_report():
    """Generates a comprehensive metric report with all visualizations"""
    # Create all charts
    create_performance_chart()
    create_training_progression_chart()
    create_resource_usage_chart()
    create_latency_distribution_chart()
    create_interactive_comparison()
    
    # Generate HTML report
    report_template = """
    <html>
        <head><title>Performance Analysis Report</title></head>
        <body>
            <h1>Challenge Performance Analysis</h1>
            <div class="chart-container">
                <img src="assets/performance_chart.png">
                <img src="assets/training_progression.png">
                <!-- Add more charts -->
            </div>
            <div class="interactive-charts">
                <iframe src="assets/interactive_comparison.html"></iframe>
            </div>
        </body>
    </html>
    """
    
    with open('docs/performance_report.html', 'w') as f:
        f.write(report_template) 