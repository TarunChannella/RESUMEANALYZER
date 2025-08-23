# JOBSINLINE - AI & ML Based Resume Analyzer

JOBSINLINE is a web-based application designed to automate the resume screening process. It analyzes a candidate's resume against a specific job role's requirements, providing a probability score for suitability and personalized feedback on skill gaps.

## ✨ Features

- **Automated Resume Parsing**: Extracts text from uploaded PDF resumes.
- **Skill-Based Matching**: Compares extracted skills against a curated dataset of job requirements.
- **Probability Score**: Calculates a match percentage indicating job suitability.
- **Personalized Feedback**: Highlights matched skills, identifies missing skills, and provides improvement suggestions.
- **Related Jobs Feature**: Proactively suggests alternative career paths based on the user's entire skillset.
- **User-Friendly Interface**: Simple and intuitive web interface for easy interaction.

## 🛠️ Technologies Used

### Backend
- **Python** with **Flask** web framework
- * **PyPDF2** for PDF text extraction

### Frontend
- **HTML**, **CSS**, **JavaScript**

### Data Storage & Processing
- **Excel (CSV/XLSX)** for storing job roles, required skills, and frameworks.

## 📦 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/jobsinline.git
    cd jobsinline
    ```

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Example `requirements.txt` contents:*
    ```
    Flask==2.3.3
    PyMuPDF==1.23.8
    openpyxl==3.1.2
    ```

4.  **Prepare the Dataset**
    - Ensure your Excel file (e.g., `jobs_dataset.xlsx`) is in the project directory.
    - The file should have columns for `Job Role`, `Required Skills`, `Frameworks`, etc.

5.  **Run the Flask Application**
    ```bash
    python app.py
    ```
    The application will start, typically at `http://127.0.0.1:5000`.

## 🚀 How to Use

1.  **Access the Application**: Open your web browser and navigate to the local host address provided after running the app.
2.  **Upload Resume**: On the homepage, click to upload your resume in PDF format.
3.  **Select Job Role**: Choose your desired job role from the dropdown menu.
4.  **Submit for Analysis**: Click the "Submit" button to analyze your resume.
5.  **Review Results**: The results page will display:
    - Your **Probability Score** for the selected job.
    - A list of **Matched Skills**.
    - A list of **Missing Skills**.
    - **Personalized suggestions** for improving your resume.

## 📊 Project Architecture

The system workflow is as follows:
1.  **User Uploads** a PDF resume and selects a job role via the frontend.
2.  **Flask Backend** receives the file and selection.
3.  **Text Extraction** using PyMuPDF parses the raw text from the PDF.
4.  **Data Preprocessing** cleans and tokenizes the text to identify skills.
5.  **Skill Matching**: Extracted skills are compared against the selected job role's requirements from the Excel dataset.
6.  **Scoring & Analysis**: A probability score is calculated, and missing skills are identified.
7.  **Results Generation**: The analysis is sent back to the frontend to be displayed to the user.

## 🔮 Future Scope

- Integration of **OCR (e.g., Tesseract)** to handle scanned PDFs.
- Implementation of **NLP techniques (e.g., spaCy, BERT)** for better context understanding and synonym matching (e.g., "ML" vs. "Machine Learning").
- Dynamic updating of the job role and skills database.
- Integration with LinkedIn profiles and online job portals for real-time analysis and recommendations.
- Support for additional file formats (e.g., DOCX, TXT).

## 👥 Contributors

- Tarun Channella
- Phaneendra Melapu
- Ranjit Kommasasni
- Ekshvak Sai Malladi

## 📄 License

This project is created for academic purposes as part of a university project.
