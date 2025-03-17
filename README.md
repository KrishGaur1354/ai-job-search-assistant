# AI-Powered Job Search Assistant

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green)](https://www.langchain.com/)
[![BrowserUse](https://img.shields.io/badge/BrowserUse-Latest-orange)](https://docs.browser-use.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🚀 Overview

The AI-Powered Job Search Assistant is an intelligent tool that automates your job search process using advanced AI capabilities. It navigates career websites, analyzes job postings, compares them with your resume, and identifies the best matches for your skills and experience.

![Job Search Demo](./demo.mp4)

## ✨ Features

- **Automated Job Exploration**: Searches across top tech companies for relevant positions
- **AI-Powered Matching**: Analyzes job descriptions and compares with your CV using advanced language models
- **Fit Score Analysis**: Calculates a match percentage between your skills and job requirements
- **Multi-LLM Support**: Works with OpenAI, Azure OpenAI, HuggingFace, and Google Gemini
- **Data Persistence**: Saves job opportunities to CSV for later review
- **Browser Automation**: Handles website navigation and data extraction
- **Resume Upload Support**: Automatically uploads your CV to job applications

## 🛠️ Technologies Used

- **Python**: Core programming language
- **LangChain**: Framework for LLM applications
- **BrowserUse**: Browser automation library
- **LLM Providers**:
  - OpenAI GPT-4o
  - Azure OpenAI
  - HuggingFace (Mixtral-8x7B)
  - Google Gemini
- **PyPDF2**: PDF processing for resume analysis
- **Asyncio**: Asynchronous operations for concurrent job searches
- **Pydantic**: Data validation and settings management
- **CSV/JSON**: Data storage and configuration

## 📋 Requirements

- Python 3.8+
- Microsoft Edge browser
- API key for at least one LLM provider (OpenAI, Azure, HuggingFace, or Google Gemini)
- A PDF resume/CV

## 🚀 Getting Started

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/KrishGaur1354/ai-job-search-assistant
   cd ai-job-search-assistant
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables or config file:
   ```
   # Create a .env file with your API keys
   OPENAI_API_KEY=your_openai_key
   AZURE_OPENAI_KEY=your_azure_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   HUGGINGFACE_API_KEY=your_huggingface_key
   GEMINI_API_KEY=your_gemini_key
   ```

The application will:
1. Ask for your CV/resume location if not found
2. Prompt for any missing API keys
3. Allow you to select target companies
4. Let you specify job types (e.g., ML, Software Engineer)
5. Define position types (internship, full-time, etc.)
6. Automatically search and analyze job postings
7. Save matching jobs to a CSV file with fit scores

## 📊 How It Works

1. **CV Analysis**: The system reads and processes your PDF resume
2. **Company Selection**: Choose from top tech companies or specify your targets
3. **Browser Automation**: Navigates to career pages and job listings
4. **Job Analysis**: Uses AI to extract key requirements from job postings
5. **Skill Matching**: Compares job requirements with your skills and experience
6. **Fit Scoring**: Calculates a 0.0-1.0 score for each position
7. **Data Storage**: Saves promising opportunities to a CSV file

## 🔄 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request