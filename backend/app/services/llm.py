from openai import OpenAI
import google.generativeai as genai
import re
from app.config import GEMINI_API_KEY, LLM_MODEL_NAME

class AnswerGenerator:
    def __init__(self):
        # OpenRouter uses the exact same client syntax as OpenAI!
        # self.client = OpenAI(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=OPENROUTER_API_KEY,
        # )

        # --- GEMINI INITIALIZATION (ACTIVE) ---
        genai.configure(api_key=GEMINI_API_KEY)
        system_instruction = (
            "You are an expert regulatory compliance assistant for the Abu Dhabi Global Market (ADGM). "
            "Your task is to synthesize a comprehensive, highly accurate answer to the user's question. "
            "CRITICAL RULES: "
            "1. You MUST base your answer EXCLUSIVELY on the provided Context Passages. "
            "2. DO NOT introduce any outside information or hallucinate rules. "
            "3. If the provided context does not contain the answer, explicitly state: 'The provided regulatory context does not contain enough information to answer this query.' "
            "4. Ensure all relevant regulatory obligations from the context are integrated without contradiction.",
            "5. Write in flowing prose paragraphs only. Do NOT use markdown formatting, bullet points, numbered lists, asterisks, or bold text. Plain text only."
        )
        self.model = genai.GenerativeModel(
            model_name=LLM_MODEL_NAME,
            system_instruction=system_instruction
        )

    def generate_answer(self, query: str, retrieved_passages: list) -> str:
        # 1. Combine the retrieved passages into a single context string
        context_blocks = []
        for i, passage in enumerate(retrieved_passages):
            context_blocks.append(f"[Document {passage['document_id']}] {passage['text']}")
        
        context_text = "\n\n".join(context_blocks)
        user_prompt = f"Context Passages:\n{context_text}\n\nQuestion:\n{query}"

        try:
            # --- OPENROUTER GENERATION (DISABLED) ---
            # response = self.client.chat.completions.create(
            #     model=LLM_MODEL_NAME,
            #     messages=[
            #         {"role": "system", "content": system_prompt}, # (System prompt was defined locally here previously)
            #         {"role": "user", "content": user_prompt}
            #     ],
            #     temperature=0.1, 
            #     max_tokens=500
            # )
            # content = response.choices[0].message.content
            # if content is None:
            #     print("⚠️ DEBUG: OpenRouter returned None. Raw response:", response)
            #     return "The AI provider failed to generate a response (Rate limit or moderation filter). Please try again."
            # return content

            # --- GEMINI GENERATION (ACTIVE) ---
            response = self.model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,     # Low temperature for factual, grounded answers
                    max_output_tokens=2048,
                )
            )
            text = response.text
            
            text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)   # **bold** and *italic*
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # ## headings
            text = re.sub(r'^\s*[\*\-]\s+', '', text, flags=re.MULTILINE)  # bullet points
            text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)   # numbered lists
            return text.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"