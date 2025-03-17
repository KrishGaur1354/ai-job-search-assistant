import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, SecretStr
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI

# Adjust the sys.path if necessary for your setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.context import BrowserContext
from browser_use.browser.browser import Browser, BrowserConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize controller
controller = Controller()

# Top 50 Tech Companies to search for job opportunities
TOP_TECH_COMPANIES = [
    "Apple", "Microsoft", "Amazon", "Alphabet (Google)", "Meta (Facebook)", "Tesla", "NVIDIA",
    "TSMC", "Samsung Electronics", "Broadcom", "Oracle", "Cisco", "Adobe", "Salesforce",
    "IBM", "Intel", "Qualcomm", "AMD", "Sony", "Dell Technologies", "SAP", "Uber", "Airbnb",
    "Netflix", "PayPal", "Shopify", "Square (Block)", "Spotify", "Twitter (X)", "Snap",
    "Zoom Video", "Slack", "Twilio", "Palantir", "Snowflake", "MongoDB", "Databricks",
    "Stripe", "Instacart", "Lyft", "DoorDash", "Pinterest", "LinkedIn", "Dropbox",
    "Unity Software", "VMware", "Workday", "ServiceNow", "Atlassian", "HubSpot"
]

# Define config file path
CONFIG_FILE = Path.cwd() / 'job_finder_config.json'

# Global CV path variable
CV = Path.cwd() / 'Krish_CV.pdf'


def load_api_keys():
    """Load API keys from config file or environment variables."""
    config = {}

    # Try to load from config file first
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logger.info("Config loaded from file")
        except json.JSONDecodeError:
            logger.error("Error parsing config file, falling back to environment variables")

    # Load from environment as fallback or to supplement config
    load_dotenv()

    # Check for OpenAI API keys
    if 'OPENAI_API_KEY' not in config and os.getenv('OPENAI_API_KEY'):
        config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    # Check for Azure OpenAI keys
    if 'AZURE_OPENAI_KEY' not in config and os.getenv('AZURE_OPENAI_KEY'):
        config['AZURE_OPENAI_KEY'] = os.getenv('AZURE_OPENAI_KEY')

    if 'AZURE_OPENAI_ENDPOINT' not in config and os.getenv('AZURE_OPENAI_ENDPOINT'):
        config['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')

    # Check for Hugging Face API keys
    if 'HUGGINGFACE_API_KEY' not in config and os.getenv('HUGGINGFACE_API_KEY'):
        config['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

    # Check for Gemini API keys
    if 'GEMINI_API_KEY' not in config and os.getenv('GEMINI_API_KEY'):
        config['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

    return config


def save_api_keys(config):
    """Save API keys to config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info("Config saved to file")


@controller.action('Configure API keys')
def configure_api_keys(openai_api_key: Optional[str] = None,
                      azure_openai_key: Optional[str] = None,
                      azure_openai_endpoint: Optional[str] = None,
                      huggingface_api_key: Optional[str] = None,
                      gemini_api_key: Optional[str] = None):
    """Configure API keys for the job finder."""
    config = load_api_keys()

    if openai_api_key:
        config['OPENAI_API_KEY'] = openai_api_key

    if azure_openai_key:
        config['AZURE_OPENAI_KEY'] = azure_openai_key

    if azure_openai_endpoint:
        config['AZURE_OPENAI_ENDPOINT'] = azure_openai_endpoint

    if huggingface_api_key:
        config['HUGGINGFACE_API_KEY'] = huggingface_api_key

    if gemini_api_key:
        config['GEMINI_API_KEY'] = gemini_api_key

    save_api_keys(config)

    return ActionResult(extracted_content="API keys configured successfully")


class Job(BaseModel):
    title: str
    link: str
    company: str
    fit_score: float
    location: Optional[str] = None
    salary: Optional[str] = None


@controller.action('Save jobs to file - with a score how well it fits to my profile', param_model=Job)
def save_jobs(job: Job):
    # Create CSV if it doesn't exist
    csv_exists = Path('jobs.csv').exists()

    with open('jobs.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not csv_exists:
            writer.writerow(["Title", "Company", "Link", "Fit Score", "Salary", "Location"])

        writer.writerow([job.title, job.company, job.link, job.fit_score, job.salary, job.location])

    return f'Saved job "{job.title}" at {job.company} to file with fit score of {job.fit_score}'


@controller.action('List all top tech companies')
def list_tech_companies():
    """Return the list of top tech companies to search for jobs."""
    return ActionResult(extracted_content="\n".join([f"{i + 1}. {company}" for i, company in enumerate(TOP_TECH_COMPANIES)]))


@controller.action('Read jobs from file')
def read_jobs():
    if not Path('jobs.csv').exists():
        return ActionResult(extracted_content="No jobs saved yet.")

    with open('jobs.csv', 'r') as f:
        return ActionResult(extracted_content=f.read())


@controller.action('Read my cv for context to fill forms')
def read_cv():
    global CV
    logging.info(f"Attempting to read CV from {CV}")
    if not CV.exists():
        logging.error(f'CV file not found at {CV}. Please set the correct path.')
        return ActionResult(error=f'CV file not found at {CV}. Please set the correct path.')

    try:
        pdf = PdfReader(CV)
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''
        logging.info(f'Successfully read CV with {len(text)} characters')
        return ActionResult(extracted_content=text, include_in_memory=True)
    except Exception as e:
        logging.error(f"Error reading CV: {e}")
        return ActionResult(error=f"Error reading CV: {e}")


@controller.action('Upload cv to element')
async def upload_cv(index: int, browser: BrowserContext):
    if not CV.exists():
        return ActionResult(error=f'CV file not found at {CV}. Please set the correct path.')

    path = str(CV.absolute())
    dom_el = await browser.get_dom_element_by_index(index)

    if dom_el is None:
        return ActionResult(error=f'No element found at index {index}')

    file_upload_dom_el = dom_el.get_file_upload_element()

    if file_upload_dom_el is None:
        logger.info(f'No file upload element found at index {index}')
        return ActionResult(error=f'No file upload element found at index {index}')

    file_upload_el = await browser.get_locate_element(file_upload_dom_el)

    if file_upload_el is None:
        logger.info(f'No file upload element found at index {index}')
        return ActionResult(error=f'No file upload element found at index {index}')

    try:
        await file_upload_el.set_input_files(path)
        msg = f'Successfully uploaded file "{path}" to index {index}'
        logger.info(msg)
        return ActionResult(extracted_content=msg)
    except Exception as e:
        logger.debug(f'Error in set_input_files: {str(e)}')
        return ActionResult(error=f'Failed to upload file to index {index}')


@controller.action('Generate job search task for company')
def generate_job_search_task(company_name: str, job_type: str = "ML", position_type: str = "internship"):
    """Generate a job search task for a specific company."""
    task = (
        f'You are a professional job finder. '
        f'1. Read my cv with read_cv\n'
        f'2. Navigate to {company_name}\'s careers page\n'
        f'3. Search for {job_type} {position_type} positions\n'
        f'4. For each relevant job posting:\n'
        f'   a. Analyze the job requirements\n'
        f'   b. Compare with my CV\n'
        f'   c. Score fit from 0.0 to 1.0\n'
        f'   d. Save jobs with fit score >= 0.6\n'
        f'5. Return to search results and continue until you\'ve checked at least 5 positions'
    )
    return ActionResult(extracted_content=task)


def initialize_browser():
    """Initialize the browser with Edge configuration."""
    # Try to detect Edge path based on OS
    edge_path = None
    if sys.platform == "win32":  # Windows
        paths = [
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        ]
        for path in paths:
            if os.path.exists(path):
                edge_path = path
                break

    if not edge_path:
        logger.warning("Microsoft Edge not found in standard locations. Please specify path manually.")
        # edge_path = input("Enter path to Microsoft Edge executable: ")
        # Removed the input because this would not work in a ui
        messagebox.showerror("Error", "Microsoft Edge not found in standard locations. Please specify the path manually in the code.")
        exit(1)

    logger.info(f"Using browser at path: {edge_path}")

    # Try with minimal parameters
    return Browser(
        config=BrowserConfig(
            chrome_instance_path=edge_path,
            disable_security=True,
            headless=False  # Set to True if you don't want to see the browser UI
        )
    )


async def setup_llm_model(model_provider="openai"):
    """Setup language model based on available API keys and provider preference."""
    config = load_api_keys()

    if model_provider == "azure" and 'AZURE_OPENAI_KEY' in config and 'AZURE_OPENAI_ENDPOINT' in config:
        logger.info("Using Azure OpenAI API")
        return AzureChatOpenAI(
            model='gpt-4o',
            api_version='2024-10-21',
            azure_endpoint=config['AZURE_OPENAI_ENDPOINT'],
            api_key=SecretStr(config['AZURE_OPENAI_KEY']),
        )
    elif model_provider == "openai" and 'OPENAI_API_KEY' in config:
        logger.info("Using OpenAI API")
        return ChatOpenAI(
            model='gpt-4o',
            api_key=SecretStr(config['OPENAI_API_KEY']),
        )
    elif model_provider == "huggingface" and 'HUGGINGFACE_API_KEY' in config:
        logger.info("Using Hugging Face API")
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # You can change this to another model
            huggingfacehub_api_token=config['HUGGINGFACE_API_KEY'],
        )
    elif model_provider == "gemini" and 'GEMINI_API_KEY' in config:
        logger.info("Using Gemini API")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # You can change this to another Gemini model
            google_api_key=config['GEMINI_API_KEY'],
        )
    else:
        raise ValueError("No API keys found. Please configure using configure_api_keys action.")


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logging.info("BrowserUse logging setup complete with level info")


class JobSearchApp:
    def __init__(self, root):
        self.root = root
        root.title("Job Search App")

        self.selected_companies = []
        self.job_type = tk.StringVar(value="ML")
        self.position_type = tk.StringVar(value="internship")
        self.model_provider = tk.StringVar(value="openai")

        self.create_widgets()
        self.load_cv_path()

    def load_cv_path(self):
        global CV
        if CV.exists():
            self.cv_path_label.config(text=str(CV))
        else:
            self.cv_path_label.config(text="No CV selected")

    def create_widgets(self):
        # CV Selection Frame
        cv_frame = ttk.LabelFrame(self.root, text="CV Selection")
        cv_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.cv_path_label = ttk.Label(cv_frame, text="No CV selected")
        self.cv_path_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        cv_button = ttk.Button(cv_frame, text="Select CV", command=self.select_cv)
        cv_button.grid(row=0, column=1, padx=5, pady=5)

        # Company Selection Frame
        company_frame = ttk.LabelFrame(self.root, text="Company Selection")
        company_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.company_listbox = tk.Listbox(company_frame, selectmode="multiple", height=10)
        for company in TOP_TECH_COMPANIES:
            self.company_listbox.insert(tk.END, company)
        self.company_listbox.grid(row=0, column=0, padx=5, pady=5)

        select_all_button = ttk.Button(company_frame, text="Select All", command=self.select_all_companies)
        select_all_button.grid(row=1, column=0, padx=5, pady=5)

        # Job Type and Position Frame
        job_frame = ttk.LabelFrame(self.root, text="Job Details")
        job_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(job_frame, text="Job Type:").grid(row=0, column=0, padx=5, pady=5)
        job_type_entry = ttk.Entry(job_frame, textvariable=self.job_type)
        job_type_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(job_frame, text="Position Type:").grid(row=1, column=0, padx=5, pady=5)
        position_type_entry = ttk.Entry(job_frame, textvariable=self.position_type)
        position_type_entry.grid(row=1, column=1, padx=5, pady=5)

        # LLM Provider Frame
        llm_frame = ttk.LabelFrame(self.root, text="LLM Provider")
        llm_frame.grid(row=3, column=0, padx=10, pady=10,                        sticky="ew")

        llm_provider_options = ["openai", "azure", "huggingface", "gemini"]
        llm_provider_menu = ttk.OptionMenu(llm_frame, self.model_provider, llm_provider_options[0], *llm_provider_options)
        llm_provider_menu.grid(row=0, column=0, padx=5, pady=5)

        # Buttons Frame
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=4, column=0, padx=10, pady=10)

        start_button = ttk.Button(button_frame, text="Start Search", command=self.start_search)
        start_button.grid(row=0, column=0, padx=5, pady=5)

    def select_cv(self):
        global CV
        filepath = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if filepath:
            CV = Path(filepath)
            self.load_cv_path()

    def select_all_companies(self):
        for i in range(len(TOP_TECH_COMPANIES)):
            self.company_listbox.select_set(i)
        self.update_selected_companies()

    def update_selected_companies(self):
        self.selected_companies = [self.company_listbox.get(i) for i in self.company_listbox.curselection()]

    def start_search(self):
        asyncio.run(self._start_search())

    async def _start_search(self):
        self.update_selected_companies()
        if not self.selected_companies:
            messagebox.showwarning("Warning", "Please select at least one company.")
            return

        if not CV.exists():
            messagebox.showwarning("Warning", "Please select a CV file.")
            return

        try:
            model = await setup_llm_model(self.model_provider.get())
            browser = initialize_browser()

            tasks = []
            for company in self.selected_companies:
                task_result = generate_job_search_task(company, self.job_type.get(), self.position_type.get())
                task = task_result.extracted_content + f"\nCompany: {company}"
                tasks.append(task)

            agents = []
            for task in tasks:
                agent = Agent(task=task, llm=model, controller=controller, browser=browser)
                agents.append(agent)

            await asyncio.gather(*[agent.run() for agent in agents])

            messagebox.showinfo("Success", "Job search completed!")

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")


def main():
    setup_logging()
    root = tk.Tk()
    app = JobSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

