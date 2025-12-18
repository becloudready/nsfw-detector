"""
Fear Sensor - Streamlit Application
====================================
This app analyzes video transcripts (YouTube or manual input) to detect fear-mongering
content using machine learning, and optionally correlates findings with Fitbit heart rate data.

Key Features:
- YouTube transcript fetching and analysis
- Fear-mongering detection using a pre-trained classifier
- Text segmentation by characters/sentences
- Interactive visualizations (line, bar, area charts)
- Fitbit heart rate data integration
- Time-based alignment of fear scores with physiological data
"""

# ======================================================
# IMPORTS
# ======================================================
from datetime import datetime, timedelta        # Handling dates & times
import pandas as pd                            # Data manipulation
from pathlib import Path                        # File system paths
import matplotlib.pyplot as plt                # Plotting (not heavily used)
import streamlit as st                          # Web app interface
import pytz                                    # Time zone handling
import nltk                                    # NLP for text segmentation
import plotly.express as px                     # Interactive charts

# === MODEL LOADING ===
from backend.fear_monger_processor.model import load_classifier       # Load fear model
from backend.fear_monger_processor.inference import run_inference    # Run inference on text
from backend.fear_monger_processor.transcript import get_video_id, fetch_transcript  # TED/YouTube transcripts
from backend.fear_monger_processor.utils import segment_text, assign_timestamps, create_analysis_df, create_plotly_chart, display_results_table  # Utils for text, chart, dataframe

# === CONFIG & UTILITIES ===
from frontend.correlation_engine.config import MAX_CHARS, DEFAULT_FEAR_THRESHOLD, DEFAULT_SMOOTHING_WINDOW, DEFAULT_CHART_TYPE
from backend.fitbit_app.fitbit_utils import get_fitbit_heart_data, plot_fitbit_heart
from backend.fitbit_app.fitbit_client import fetch_fitbit_data
from backend.fitbit_app.aligner import align_fear_and_heart # Align fear vs heart rate
from backend.fitbit_app.playback_window import estimate_playback_window
from backend.fitbit_app.config import TOKEN_FILE
from backend.ted_talks_app.data_loader import load_transcripts  # Load TED transcripts

# Get base directory for relative path resolution
base_dir = Path(__file__).resolve().parents[2]

# ======================================================
# CSS LOADING FUNCTION
# ======================================================
def load_css(file_path):
    """
    Load custom CSS styling for the Streamlit app.
    
    Args:
        file_path: Path to CSS file
    """
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
# ======================================================
# MAIN STREAMLIT APP FUNCTION
# ======================================================
def main():
    """
    Main application entry point.
    Orchestrates the entire UI flow and data processing pipeline.
    """

    # === Load Styles ===
    load_css("styles.css")

     # === Ensure NLP dependencies ===
    # Download NLTK punkt tokenizer if not present (needed for sentence segmentation)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # === Load Classifier Model ===
    # Use session_state to cache the model - only load once per session
    if "classifier" not in st.session_state:

        # Only load once — stored in session_state for efficiency
        with st.spinner("Loading fear detection model..."):
            st.session_state.classifier = load_classifier()

    classifier = st.session_state.classifier

  
    # ======================================================
    # UI SETUP
    # ======================================================
    st.title("NSF Detector")

    # ======================================================
    # SIDEBAR: Segmentation Settings
    # ======================================================
    # Allow users to control how text is chunked for analysis
    with st.sidebar.expander("Select Fear Threshold ", expanded=False):

        # Segment text by Characters, Sentences, or Both
        segment_mode = st.radio(
            "Segment text by:",
            ["Characters", "Sentences", "Both"],
            index=2,  # Default to "Both"
            help="Choose how to split the text into paragraphs for analysis."
        )

        # Default values
        max_chars = MAX_CHARS
        max_sentences = 5

        # Conditional UI: show character slider if relevant
        if segment_mode in ("Characters", "Both"):
            max_chars = st.slider(
                "Maximum Characters per Segment",
                min_value=200,
                max_value=600,
                value=MAX_CHARS,
                step=5,
                help="Split the text once this many characters are reached."
            )

        # Conditional UI: show sentence slider if relevant
        if segment_mode in ("Sentences", "Both"):
            max_sentences = st.slider(
                "Maximum Sentences per Segment",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Split the text once this many sentences are reached."
            )

        # ======================================================
        # Fear Threshold Settings
        # ======================================================
        # Set the threshold above which content is flagged as fear-mongering
        threshold = st.slider(
            "Fear Mongering Threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_FEAR_THRESHOLD,
            step=0.05,
            help="Adjust the threshold for analysis."
        )

        # Smoothing reduces noise in the score timeline
        smoothing_window = st.slider(
            "Smoothing Window Size",
            min_value=1,
            max_value=10,
            value=DEFAULT_SMOOTHING_WINDOW,
            step=1,
            help="Larger window smooths the fear mongering score more."
        )


    # ======================================================
    # SIDEBAR: Chart Options
    # ======================================================
    with st.sidebar.expander("Chart Options", expanded=False):
        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Bar Chart", "Area Chart"],
            index=["Line Chart", "Bar Chart", "Area Chart"].index(DEFAULT_CHART_TYPE),
            help="Choose how to display fear mongering trends."
        )

        # Limit hover text length to prevent UI clutter
        max_hover_length = st.slider(
            "Max Hover Text Length",
            min_value=20,
            max_value=500,
            value=30,
            step=10,
            help="Limit number of characters shown in chart hover text."
        )

    
    # ===============================
    # MAIN CONTENT: YouTube URL Input
    # ===============================
    url_input = st.text_input(
        "Enter YouTube URL:",
        help="Paste a YouTube video URL to fetch its transcript."
    )

    # Create the expander immediately after URL input for transcript preview
    with st.expander("Transcript Preview"):
        expander_content = st.empty()

    transcript_text = None
    video_id = None

    # Process YouTube URL if provided
    if url_input.strip():
        video_id = get_video_id(url_input.strip())

        if video_id:
            # Fetch transcript using YouTube API/library
            transcript_text = fetch_transcript(video_id)  # progress bar handled inside function

            if transcript_text:
                # Show preview (first 4000 characters)
                expander_content.write(transcript_text[:4000])  # first 4000 chars preview
            else:
                expander_content.write("No transcript available for this video.")

        else:
            st.error("Invalid YouTube URL.")

    # ===============================
    # Manual Transcript Input
    # ===============================
    # Fallback option if user doesn't have a YouTube URL
    quick_text = st.text_area(
        "Or paste transcript text here:",
        height=100,
        help="Paste transcript text directly if you don’t want to use a URL."
    )

    # Use manual text if no YouTube transcript was fetched
    if not transcript_text and quick_text.strip():
        transcript_text = quick_text
        expander_content.write(transcript_text[:4000])  # preview manual transcript

    # ===============================
    # YouTube Video Embed
    # ===============================
    # Display the video inline if we have a valid video ID
    if video_id:
        embed_code = f"""
        <div class="video-container">
            <div class="video-wrapper">
                <iframe
                    src="https://www.youtube.com/embed/{video_id}"
                    frameborder="0"
                    allowfullscreen>
                </iframe>
            </div>
        </div>
        """
        st.markdown(embed_code, unsafe_allow_html=True)

    # Early exit if no transcript available
    if not transcript_text:
        st.info("Paste a YouTube URL above or enter transcript text to begin analysis.")


    # ========================
    # TEXT SEGMENTATION
    # ========================
    # Split transcript into analyzable chunks based on user settings
    text_to_analyze = transcript_text or quick_text

    paragraphs = segment_text(
        text_to_analyze,
        max_chars=max_chars if segment_mode in ("Characters", "Both") else float('inf'),
        max_sentences=max_sentences if segment_mode in ("Sentences", "Both") else float('inf')
    )

    # Create fake timestamps based on text length (for visualization)
    if not paragraphs:
        st.warning("No paragraphs detected.")
        return

    fake_duration = max(len(text_to_analyze) // 10, 10)
    timestamps = assign_timestamps(paragraphs)


    # ========================
    # RUN ML INFERENCE
    # ========================
    # Core model execution: classify each paragraph for fear-mongering
    predictions = run_inference(classifier, paragraphs)

    st.markdown("---")


    # ======================================================
    # Display Segmented Transcript
    # ======================================================
    st.subheader("View Transcript Segments")

    st.write(f"Text split into {len(paragraphs)} segments (max {max_chars} chars each)")

    # Show all segments in expandable section
    with st.expander("View All Segments"):
        df_paragraphs = pd.DataFrame({
            "Segment #": list(range(1, len(paragraphs) + 1)),
            "Text": paragraphs
        })
        st.dataframe(df_paragraphs, use_container_width=True)

    st.markdown("---")


    # ======================================================
    # Create analysis DataFrame
    # ======================================================
    # Combine paragraphs, timestamps, predictions, and apply smoothing
    analysis_df = create_analysis_df(
        paragraphs=paragraphs,
        timestamps=timestamps,
        predictions=predictions,
        smoothing_window=smoothing_window,
        video_duration_seconds=fake_duration,
    )

    # Store results in session state for downstream correlation with Fitbit
    st.session_state["fear_results_df"] = analysis_df
    st.session_state["video_duration_seconds"] = fake_duration


    # ======================================================
    # Extract Time Series Data
    # ======================================================
    seconds = timestamps["seconds"]
    scores = analysis_df["Fear Mongering Score"]

    # ========================
    # SUMMARY STATISTICS & VISUALIZATION
    # ========================
    st.subheader("Quick Analysis Summary")

    # Two-column layout: stats + pie chart
    col1, col2 = st.columns([2, 1], gap="small")  # Main column + pie chart column

    with col1:
        # Calculate key metrics
        avg_score = scores.mean()
        max_score = scores.max()
        min_score = scores.min()
        high_risk_count = len(analysis_df[analysis_df["Fear Mongering Score"] >= threshold])
        percentage = (high_risk_count / len(paragraphs)) * 100

        # Display metrics in 4-column grid
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

        with stats_col1:
            st.metric("Paragraphs", len(paragraphs))

        with stats_col2:
            st.metric(
                "Average Score",
                f"{avg_score:.3f}",
                delta=f"{((avg_score - threshold) * 100):.0f}% vs threshold",
                delta_color="inverse" # Red for higher scores
            )

        with stats_col3:
            st.metric(
                "Peak Score",
                f"{max_score:.3f}",
                delta=f"Low: {min_score:.3f}"
            )

        with stats_col4:
            st.metric(
                "High Risk",
                f"{high_risk_count}",
                delta=f"{percentage:.1f}% of talk"
            )

        st.markdown("---")

        # ======================================================
        # Overall Assessment with Color-Coded Status
        # ======================================================
        st.subheader("Overall Assessment")

        # High risk: average score exceeds threshold
        if avg_score >= threshold:
            st.markdown(f'<div class="theme-status-box error">'
                        f"### High Fear Mongering Detected<br>"
                        f"- Average score: **{avg_score:.3f}** (threshold: {threshold:.2f})<br>"
                        f"- **{high_risk_count}** of {len(paragraphs)} paragraphs exceed threshold<br>"
                        f"- Peak fear score: **{max_score:.3f}**"
                        f"</div>", unsafe_allow_html=True)

        # Moderate risk: score between 0.5 and threshold
        elif avg_score >= 0.5:
            st.markdown(f'<div class="theme-status-box warning">'
                        f"### Moderate Concern<br>"
                        f"- Average score: **{avg_score:.3f}** (threshold: {threshold:.2f})<br>"
                        f"- **{high_risk_count}** segments above threshold<br>"
                        f"- Approaching concerning levels"
                        f"</div>", unsafe_allow_html=True)

        # Low risk: score below 0.5
        else:
            st.markdown(f'<div class="theme-status-box success">'
                        f"### Low Risk Content<br>"
                        f"- Average score: **{avg_score:.3f}** (well below {threshold:.2f})<br>"
                        f"- Only **{high_risk_count}** high-risk segments<br>"
                        f"- Generally balanced messaging"
                        f"</div>", unsafe_allow_html=True)

    with col2:
        # ======================================================
        # Pie Chart: Score Distribution
        # ======================================================
        st.markdown('<div style="margin-top:-5px; padding-top:0;"><h5 style="margin-bottom:5px;">Distribution</h5></div>', unsafe_allow_html=True)

        # Categorize paragraphs into Low/Medium/High risk
        low = len(analysis_df[analysis_df["Fear Mongering Score"] < 0.5])
        medium = len(analysis_df[(analysis_df["Fear Mongering Score"] >= 0.5) &
                                (analysis_df["Fear Mongering Score"] < threshold)])
        high = len(analysis_df[analysis_df["Fear Mongering Score"] >= threshold])

        dist_df = pd.DataFrame({
            "Category": ["Low", "Medium", "High"],
            "Count": [low, medium, high]
        })

        # Create pie chart with custom colors
        fig = px.pie(
            dist_df,
            names="Category",
            values="Count",
            color="Category",
            color_discrete_map={
                "Low": "#4FC3F7",     # Light blue
                "Medium": "#FFD54F",  # Yellow
                "High": "#EF5350"     # Red
            }
        )

        fig.update_traces(
            textinfo="percent+label",
            hoverinfo="label+percent+value",
            marker=dict(line=dict(color="#0D1526", width=0))
        )

        fig.update_layout(
            height=320,  # make pie chart height match nicely
            margin=dict(l=10, r=10, t=0, b=0),
            showlegend=False,
            plot_bgcolor="#0D1526",
            paper_bgcolor="#0D1526"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =======================
    # INTERACTIVE TIME SERIES CHART
    # =======================
    st.subheader("Fear Mongering Trend")
    fig = create_plotly_chart(seconds, scores, paragraphs, chart_type=chart_type, max_hover_length=max_hover_length)
    st.plotly_chart(fig, use_container_width=True)


    # =======================
    # DETAILED TABLE ANALYSIS
    # =======================
    st.subheader("Paragraph-Level Analysis")
    display_results_table(analysis_df, threshold)


    # ========================
    # DOWNLOAD RESULTS
    # ========================
    # Allow users to export analysis results
    csv = analysis_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results as CSV",
        data=csv,
        file_name="quick_fear_analysis.csv",
        mime="text/csv",
        key="download_fear_csv"
    )

 


if __name__ == "__main__":
    main()
