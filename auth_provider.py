import os
from typing import Dict, List

class AuthProvider:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def execute_query(self, sql: str) -> List[Dict]:
        if sql == "select * from dual":
            pdf_files = self._get_pdf_files()
            return [
                {"user_id": "admin", "permissions": ",".join(pdf_files)}
            ]
        return []

    def _get_pdf_files(self) -> List[str]:
        pdf_dir = "./docs"
        return [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]