**au-scraper** is a tool that scrapes, parses, and analyzes **African Union Peace and Security Council (AUPSC) communiqués for 2025**.  
It leverages **Natural Language Processing (NLP)** and **sentiment analysis** to measure **diplomatic intent and tone**, transforming unstructured diplomatic texts into structured data for further research and policy insights.  

---

## ✨ Features  
- 🌐 Scrapes AUPSC communiqués (2025) from official sources  
- 📄 Parses and cleans raw text into machine-readable formats  
- 🧠 Applies **NLP pipelines** for sentiment and tone analysis  
- 📊 Measures **diplomatic intent and tone** with custom scales  
- 📈 Exports structured datasets for visualization and policy modeling  

---

## ⚙️ Tech Stack  
- **Python 3**  
- **BeautifulSoup4** / **Requests** (web scraping)  
- **NLTK / spaCy / Transformers** (NLP & sentiment analysis)  
- **Pandas / NumPy** (data processing)  
- **Matplotlib / Seaborn / Plotly** (visualization)  

---

## 🚀 Usage  
```bash
# Clone the repository  
git clone https://github.com/tkodjo/aupsc-scraper.git  

# Navigate into the project  
cd aupsc-scraper  

# Install dependencies  
pip install -r requirements.txt  

# Run the scraper & analysis: you will get the chart in the output\charts folder
python run.py  

