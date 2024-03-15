import openai

class Selected_Model:
    def __init__(self, api_key, model_engine):
        self.api_key = api_key
        self.model_engine = model_engine
        openai.api_key = self.api_key

    def get_response(self, prompt, contexts, max_tokens=1024, temp=0.5):
        try:
            # Attempt to use the latest Chat API
            response = openai.ChatCompletion.create(
                model=self.model_engine,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}\nContexts:\n{contexts}"}
                ],
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=temp,
            )
            return response.choices[0].message['content'].strip()
            
        except openai.APIError:
            # Fallback to Completion API if Chat API is not available
            response = openai.Completion.create(
                model=self.model_engine,
                prompt=f"{prompt}\nContexts:\n{contexts}",
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=temp,
            )
            return response.choices[0]['text'].strip()