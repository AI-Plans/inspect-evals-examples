import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')
PROVIDER = os.getenv('PROVIDER')
DATASET = os.getenv('DATASET')
