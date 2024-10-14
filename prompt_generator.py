import os
import logging
import anthropic
from dotenv import load_dotenv
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

system_prompt = """
You are a creative prompt generator for AI image generation. 
Your task is to create a detailed description for a waist-level portrait photo of a {gender} in a setting suitable for an Instagram post, incorporating the concept: {concept}.

In 2-3 short sentences, The description should include:
1. Clothing description (including colors and styles)
2. Background setting (natural or urban environments)
3. Pose or action
4. Additional details (accessories, lighting, etc.)

Ensure the subject is:
- Directly facing the camera
- Looking straight into the lens
- Facing forward

Base on the gender, the desctiption then will be merged with the following prompt base;
if female;
    "female": 
        "white": "caucasian woman, beautiful, smiling, iphone photo, waist level portrait, + generated_description",
        "black": "african-american woman, beautiful, smiling, iphone photo, waist level portrait, + generated_description"

if male; 
    "male": 
        "white": "caucasian man, handsome, smiling, iphone photo, waist level portrait, + generated_description",
        "black": "african-american man, black man, handsome, smiling, iphone photo, waist level portrait, + generated_description"

so that the generated description should be suited for the gender and race and must be generalizable that suits both white and black races.
        
Avoid;
1. hat, sunglasses, mask, or any other accessory in the description
2. neutral or serious expression
3. Exaggerated accessories
4. Holding objects such as glasses, phones, or bags

Output only the description, without introductions or formatting.
Ensure the description is vivid and Instagram-worthy.
"""

def generate_prompt(gender: str, 
                    concept: str, 
                    previous_descriptions: list[str]) -> str:
    previous_descriptions_text = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(previous_descriptions))
    formatted_system_prompt = system_prompt.format(gender=gender, concept=concept)
    
    if previous_descriptions:
        formatted_system_prompt += f"\n\nHere are the previously generated descriptions:\n{previous_descriptions_text}\n\nPlease generate a new description that incorporates the given concept while being distinctly different from those listed above. Especially the clothing description should be different. Ensure creativity and variety in your new description."

    message = client.messages.create(
        model = "claude-3-5-sonnet-20240620",
        max_tokens = 300,
        temperature = 0.9,
        system = formatted_system_prompt,
        messages = [
            {
                "role": "user",
                "content": "Generate a description."
            }
        ]
    )
    description = message.content[0].text.strip()
    
    return description

def format_prompt(gender: str, race: str, description: str) -> str:
    base_prompt = {
        "female": {
            "white": "caucasian woman, beautiful, smiling, iphone photo, waist level portrait,",
            "black": "african-american woman, beautiful, smiling, iphone photo, waist level portrait,"
        },
        "male": {
            "white": "caucasian man, handsome, smiling, iphone photo, waist level portrait,",
            "black": "african-american man, black man, handsome, smiling, iphone photo, waist level portrait,"
        }
    }
    
    return f"{base_prompt[gender][race]} {description} posted on Instagram"

def prompt_generator(gender: str, concept: str, num_prompts: int) -> dict:
    if gender not in ["male", "female"]:
        raise ValueError("Gender must be either 'male' or 'female'")
    
    result = {
        "metadata": {
            "gender": gender,
            "concept": concept,
            "total_prompts": num_prompts
        },
        "prompts": []
    }
    
    previous_descriptions = []
    
    for i in range(num_prompts):
        try:
            description = generate_prompt(gender, concept, previous_descriptions)
            white_prompt = format_prompt(gender, "white", description)
            black_prompt = format_prompt(gender, "black", description)
            
            prompt_pair = {
                "prompt_id": str(uuid.uuid4().hex[:6]),
                "white_prompt": white_prompt,
                "black_prompt": black_prompt
            }
            result["prompts"].append(prompt_pair)
            
            previous_descriptions.append(description)
            
            logger.info(f"Generated prompt pair {i+1}")
        except Exception as e:
            logger.error(f"Error generating prompt pair {i+1}: {str(e)}")
    
    logger.info(f"Generated {len(result['prompts'])} prompt pairs")
    
    return result
