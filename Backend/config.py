import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
