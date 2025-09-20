HR Talent Management Dashboard

This is an enterprise-grade, interactive dashboard built with Streamlit for HR talent management. It provides a comprehensive suite of tools for analyzing employee data, including performance metrics, potential ratings, and talent distribution across various departments.

🌟 Key Features
Multi-File Upload: Securely upload and consolidate multiple Excel files (.xlsx, .xls) into a single master dataset.
Interactive Dashboard: A Power BI-style dashboard with dynamic filters for real-time data exploration.
9-Box Grid Analysis: Visualize employee talent distribution based on performance and potential, helping to identify stars, future leaders, and at-risk employees.
Advanced Analytics: Includes charts for departmental distribution, employee tenure, performance vs. potential scatter plots, and talent category breakdowns.
AI-Powered Insights: Optional integration with the Google Gemini API to generate strategic, actionable insights and recommendations based on your data.
Secure Data Export: Download filtered or complete datasets to a formatted Excel file.
Enhanced UX: A user-friendly interface with a guided tour, pagination, view presets, and user-friendly error messages, designed for non-technical users.

🛠️ Tech Stack
Python 3.8+
Streamlit: For the web application interface.
Pandas: For data manipulation and analysis.
Plotly & Plotly Express: For interactive data visualizations.
NumPy: For numerical operations.
Google Gemini API: For generating AI-powered insights.

⚙️ Setup and Installation
Follow these steps to get the dashboard running on your local machine.
1. Prerequisites
Python 3.8 or higher installed on your system.
A Google Gemini API key for the AI Insights feature. You can obtain one from Google AI Studio.
2. Clone the Repository
(If this were a Git repository, you would clone it)
git clone https://your-repository-url/hr-dashboard.git
cd hr-dashboard

3. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate


4. Install Dependencies
Install all the required Python packages using the requirements.txt file.
pip install -r requirements.txt


5. Configure Environment Variables
Create a file named .env in the root directory of the project and add your Google Gemini API key.
# .env
GEMINI_API_KEY="your_gemini_api_key_here"


The application will automatically load this key. Alternatively, you can enter the key directly in the UI.
🚀 Running the Application
Once the setup is complete, you can run the Streamlit application with the following command:
streamlit run hr_dashboard.py


Your web browser should open a new tab with the dashboard running.
📁 Data Format
For the application to work correctly, your uploaded Excel files must contain the following columns:
Employee ID
Employee Name
Department
Designation
Reporting Manager
Date of Joining
HOD (Head of Department)
Performance (on a 1-3 scale)
Potential Rating (on a 1-3 scale)
You can download a sample template from the sidebar in the application to ensure your data is formatted correctly.
💡 App Configuration (Optional)
The application can be configured using environment variables. The following variables are supported (see the AppConfig class in the script for more details):
HR_DASHBOARD_PAGE_TITLE
HR_DASHBOARD_MAX_FILE_SIZE_MB
HR_DASHBOARD_SESSION_TIMEOUT_MINUTES
HR_DASHBOARD_ENABLE_AI_INSIGHTS
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

