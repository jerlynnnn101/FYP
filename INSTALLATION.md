# Installation Guide: Intelligent Document Retrieval and Question Answering System

## Prerequisites
- Python 3.11 (recommended)
- Git

## 1. Clone the Repository
```
git clone https://github.com/jerlynnnn101/FYP.git
cd FYP
```

## 2. (Optional) Create a Virtual Environment
```
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

## 3. Install Dependencies
```
pip install -r requirements.txt
```

## 4. Configure Environment Variables
- Copy `.env.example` to `.env`:
	```
	cp .env.example .env
	```
- Sign up for IBM Watsonx and IBM Cloud Object Storage at https://cloud.ibm.com/ and create your own API credentials and service endpoints.
- Fill in your `.env` file with your personal IBM Watsonx API key, project ID, endpoint, and IBM Cloud Object Storage credentials.
- Do not share or commit your `.env` file to GitHub.

## 5. Run the Application
```
python app.py
```
- The app will start locally (default: http://localhost:5001)

## 6. Access the Web Interface
- Open your browser and go to: http://localhost:5001

## 7. Usage
- Log in or continue as guest.
- Upload PDF or DOCX documents.
- Submit queries and view AI-generated answers.

## 8. Troubleshooting
- Ensure all dependencies are installed.
- Check your `.env` file for correct API keys and configuration.
- If you encounter errors, consult the README or contact the project maintainer.

---
**Note:**
- For production deployment, use a WSGI server like Gunicorn and configure environment variables securely.
- IBM Watsonx and COS credentials are required for full functionality.
