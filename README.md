# PDF Chat Application with Google Gemini

## Overview
This Streamlit application allows users to upload PDF files and ask questions about their content using Google's Gemini AI.

## Prerequisites
- Python 3.8+
- Google AI API Key

## Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/hawkh/chatwithpdf.git
cd pdf-chat-app
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up Google AI API Key
- Go to https://makersuite.google.com/app/apikey
- Create an API key
- Add the key to the `.env` file

5. Run the application
```bash
streamlit run main.py
```

## Deployment
- For Streamlit Cloud: 
  1. Push to GitHub
  2. Connect Streamlit Cloud to your repository
  3. Set environment variables in Streamlit Cloud settings

## Notes
- Ensure you have a stable internet connection
- Large PDFs may take longer to process

## 🤝 Contributing

- Contributions are welcome! Please check the outstanding issues and feel free to open a pull request.


