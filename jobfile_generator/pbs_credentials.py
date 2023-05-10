from pydantic import BaseSettings, Field
from pathlib import Path

env_path = Path(__file__).parent / '.env'
# NOTE! Make sure the .env file is present, and has the required variables


class PbsCreds(BaseSettings):
    alloc_code: str = Field(..., env='ALLOC_CODE')
    
    class Config:
        env_prefix = ''
        case_sensitive = False
        env_file = env_path
        env_file_encoding = 'utf-8'