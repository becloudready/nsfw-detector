# NSFW Detection: Detect Image and Youtube Transcripts


A research platform that detects **fear-mongering patterns in video transcripts** and correlates them with **biometric stress signals** (heart rate, HRV, EDA) from wearables like Fitbit, Apple Health, or Google Fit.

---

## 🎯Features

* **Multi-source transcript ingestion**: YouTube URLs, manual text input, SRT/VTT subtitles, or MP4 transcription via Whisper.
* **ML-powered fear detection**: NLP pipeline combining emotion classification, zero-shot NLI, and lexical cue analysis.
* **Flexible text segmentation**: Character-based, sentence-based, or hybrid segmentation with adjustable parameters.
* **Interactive visualization**: Dual-axis charts, distribution analytics, and downloadable datasets.


* **Backend**: FastAPI + Celery for async processing
* **Storage**: Postgres/TimescaleDB + MinIO (object storage)
* **ML/NLP**: [Falconsai/fear_mongering_detection](https://huggingface.co/Falconsai/fear_mongering_detection)
* **Frontend**: Streamlit (MVP) or Next.js (scalable)
* **Visualization**: Plotly, Matplotlib


## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Docker & docker-compose (for running services)

---

### 📦Installation
```bash
# Clone repository
git clone https://github.com/becloudready/nsfw-detector.git
cd nsfw-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your Fitbit credentials
```



## Running the Demo
<p style="font-size:16px;">
This guide demonstrates how to analyze a video transcript to detect fear-mongering and correlate the results with biometric stress data, such as heart rate from a Fitbit device.
</p>

---
1️⃣ Launch the Application

1. Navigate to the app:
```bash
cd src/frontend/
streamlit run app.py
```

<p style="font-size:14px;"> The app will open in your browser at http://localhost:8501 </p>

---

2️⃣ Run Fear Analysis

Option A: YouTube Video

1. Paste a YouTube URL into the input field
2. The transcript will be fetched automatically
3. Preview appears in the "Transcript Preview" expander

Option B: Manual Input
1. Paste transcript text into the text area
2. Text is immediately ready for analysis
---
### Configuration Options

#### Segmentation Settings (Sidebar)
* **Segment Mode**: Characters | Sentences | Both
* **Max Characters**: 200-600 (default: 400)
* **Max Sentences**s: 1-10 (default: 5)

### Analysis Parameters
* **Fear Threshold**: 0.0-1.0 (default: 0.7)
  - Scores above threshold = high risk
* **Smoothing Window**: 1-10 segments (default: 3)
  - Reduces noise in timeline

### Chart Options
* **Type**: Line | Bar | Area chart
* **Hover Length**: 20-500 characters (default: 30)

---
### Analysis Output

#### ✅ Quick Summary
- **Total paragraphs analyzed**
- **Average fear score vs. threshold**
- **Peak score and minimum score**
- **High-risk segment count and percentage**

#### ✅ Overall Assessment (Color-coded)
- 🔴 **High Risk**: Average score ≥ threshold  
- 🟡 **Moderate**: 0.5 ≤ Average score < threshold  
- 🟢 **Low Risk**: Average score < 0.5  

#### ✅ Visualizations
- **Distribution pie chart** (Low / Medium / High)  
- **Interactive timeline chart** (Line / Bar / Area)  
- **Paragraph-level analysis table**

#### ✅ Downloads
- **CSV export** with all scores and timestamps
CSV export with all scores and timestamps


**📊 Output**
- **Dual-axis chart**: Fear score (line) + heart rate (bars)
- **Summary metrics**:
  - Average fear score
  - Average heart rate (bpm)
  - Number of aligned data points
- **Aligned data table**: Preview first 20 rows
- **CSV download**: Full merged dataset



### Ethical Considerations:
- Use responsibly and transparently
- Respect privacy when analyzing content
- Consider potential biases in training data
- Do not use for surveillance or manipulation
---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---


