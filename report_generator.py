import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from datetime import datetime
import pandas as pd

def generate_pdf_report(store_id, forecast_df, metrics, logo_path, output_path):
    """
    Generates a 3-page branded executive PDF report.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=20,
        textColor=colors.HexColor("#004aad")
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        fontSize=18,
        spaceBefore=12,
        spaceAfter=12,
        textColor=colors.HexColor("#1f2937")
    )

    story = []

    # --- PAGE 1: EXECUTIVE SUMMARY ---
    if os.path.exists(logo_path):
        img = Image(logo_path, width=150, height=50)
        img.hAlign = 'LEFT'
        story.append(img)
    
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Sales Forecast Executive Report: Store {store_id}", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 30))
    
    story.append(Paragraph("Executive Summary", header_style))
    summary_text = f"""
    This report provides a detailed sales projection for Rossmann Store {store_id}. 
    Our AI model, powered by Meta Prophet, has analyzed historical transactional data to project 
    sales performance over the next {len(forecast_df)} days. 
    <br/><br/>
    The analysis accounts for weekly seasonality, German national holidays, and ongoing retail promotions.
    Current projections indicate a <b>{metrics['growth']:.1f}%</b> change in average daily sales 
    compared to historical benchmarks.
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    
    story.append(Spacer(1, 40))
    story.append(Paragraph("Key Performance Metrics", header_style))
    
    metrics_data = [
        ["Metric", "Value"],
        ["Historical Avg Daily Sales", f"{metrics['hist_avg']:,.2f}€"],
        ["Forecasted Avg Daily Sales", f"{metrics['forecast_avg']:,.2f}€"],
        ["Projected Total Volume", f"{metrics['total_volume']:,.2f}€"],
        ["Projected Growth", f"{metrics['growth']:+.1f}%"]
    ]
    
    t = Table(metrics_data, colWidths=[200, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#004aad")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    
    story.append(PageBreak())

    # --- PAGE 2: MODEL PERFORMANCE ---
    story.append(Paragraph("Model Performance & Comparison", header_style))
    perf_text = """
    Our forecasting pipeline evaluates multiple statistical and machine learning architectures. 
    The current production model (Meta Prophet) has been benchmarked against naive baselines 
    and seasonal ARIMA models to ensure maximum accuracy.
    """
    story.append(Paragraph(perf_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    comparison_data = [
        ["Model Architecture", "MAE", "RMSE", "MAPE (%)"],
        ["Naive Baseline (Lag-7)", "981.45", "1,162.32", "26.29%"],
        ["SARIMAX (2,0,1)(2,0,2)[7]", "491.46", "594.05", "11.71%"],
        ["Meta Prophet (Production)", "305.61", "381.16", "7.03%"]
    ]
    
    t2 = Table(comparison_data, colWidths=[180, 80, 80, 80])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])
    ]))
    story.append(t2)
    
    story.append(Spacer(1, 30))
    story.append(Paragraph("Insights:", styles['Heading4']))
    insights = [
        "• The Prophet model achieves a 73% error reduction over the baseline.",
        "• High accuracy (7.03% MAPE) allows for precise inventory and staffing optimization.",
        "• The model successfully captures holiday-driven surges and weekend dips."
    ]
    for insight in insights:
        story.append(Paragraph(insight, styles['Normal']))

    story.append(PageBreak())

    # --- PAGE 3: DETAILED FORECAST ---
    story.append(Paragraph("Detailed Forecast Projection", header_style))
    
    # In a real app, we'd save a chart image here and include it.
    # For now, we'll list the top 10 rows of the forecast.
    story.append(Paragraph(f"Next {min(15, len(forecast_df))} Days Point Forecast:", styles['Heading4']))
    
    forecast_table_data = [["Date", "Forecasted Sales", "Confidence Low", "Confidence High"]]
    for _, row in forecast_df.tail(15).iterrows():
        forecast_table_data.append([
            row['ds'].strftime('%Y-%m-%d'),
            f"{row['yhat']:,.2f}€",
            f"{row['yhat_lower']:,.2f}€",
            f"{row['yhat_upper']:,.2f}€"
        ])
    
    t3 = Table(forecast_table_data, colWidths=[100, 100, 100, 100])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#004aad")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(t3)
    
    story.append(Spacer(1, 40))
    disclaimer = "<i>Disclaimer: This forecast is based on historical patterns and statistical probability. Actual results may vary due to unforeseen market conditions or logistics disruptions.</i>"
    story.append(Paragraph(disclaimer, styles['Normal']))

    # Build PDF
    doc.build(story)
    return output_path
