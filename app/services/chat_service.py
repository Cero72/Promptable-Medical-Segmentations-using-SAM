from typing import List
# import openai

class MedicalChatService:
    def __init__(self):
        self.context = """You are a medical imaging assistant. Help analyze 
        medical images and provide clinical insights."""
        
    async def get_response(self, 
                          message: str, 
                          image_findings: dict,
                          chat_history: List[dict] = None):
        prompt = self._build_medical_prompt(message, image_findings)
        
        response = await openai.ChatCompletion.create( # type: ignore
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.context},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def _build_medical_prompt(self, message, findings):
        return f"""
        Image Analysis Findings:
        - Segmented regions: {findings.get('regions', [])}
        - Measurements: {findings.get('measurements', {})}
        
        User Question: {message}
        
        Please provide medical context and analysis.
        """
