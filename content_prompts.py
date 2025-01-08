def main_instruction(keywords:str) -> str:
    return f"""
- Include keywords naturally from: {keywords}
- Avoid repeating phrases or sentence structures
- Add small punctuation errors in specific sentences noted below
- Do not place any phrase or sentence in parenthesis or quotes or brackets
- Do not use any short form of words or phrases (e.g. don't, doesn't, hasn't, haven't, etc.)
- Do not use any emojis or special characters
- Use natural, conversational language that can pass AI detection
- Avoid overused AI-generated words and phrases like:
  * Elevate, Transform, Revolutionize, Empower
  * Seamless, Cutting-edge, Innovative, State-of-the-art
  * Captivate, Engage, Enhance, Optimize
  * Streamline, Leverage, Unlock, Maximize
  * Game-changing, Next-level, Best-in-class
  * Robust, Comprehensive, Holistic
  * Dynamic, Intuitive, User-friendly
- Instead use natural, specific language that describes actual benefits and features
"""



def get_keywords_prompt(main_topic: str, sub_topic: str, sub_sub_topic: str) -> str:
    return f"""Generate 5 relevant SEO keywords for the following topic hierarchy:
Main Topic: {main_topic}
Sub Topic: {sub_topic}
Sub Sub Topic: {sub_sub_topic}

Requirements:
- Keywords should be relevant to PowerPoint templates and presentations
- Include long-tail keywords
- Include transactional intent keywords
- Keywords should be 2-4 words long
- No single word keywords

Format: Return only the keywords separated by ||| (triple pipe).
Example: professional slide design ||| business presentation template ||| corporate strategy slides"""

def get_image_title(sub_sub_topic: str, keywords: list) -> str:
    return f"""Create an SEO-friendly image title for a PowerPoint template about "{sub_sub_topic}".

Requirements:
- Maximum 60 characters
- Include the topic: {sub_sub_topic}
{main_instruction(keywords)}

Format: Return only the title in a plain text with no additional formatting."""

def get_image_alt(sub_sub_topic: str, keywords: list) -> str:
    return f"""Create an SEO-friendly image alt text for a PowerPoint template about "{sub_sub_topic}".

Requirements:
- Maximum 125 characters
- Include the topic: {sub_sub_topic}
{main_instruction(keywords)}

Format: Return only the alt text in a plain text with no additional formatting."""

def get_image_caption(sub_sub_topic: str, keywords: list) -> str:
    return f"""Create an image caption for a PowerPoint template about "{sub_sub_topic}".

Requirements:
- Maximum 125 characters
- Include the topic: {sub_sub_topic}
- Highlight user benefits
{main_instruction(keywords)}

Format: Return only the caption in a plain text with no additional formatting."""

def get_image_description(sub_sub_topic: str, keywords: list) -> str:
    return f"""Create an image description for a PowerPoint template about "{sub_sub_topic}".

Requirements:
- Should be less than 250 characters
- Include the topic: {sub_sub_topic}
{main_instruction(keywords)}

Format: Return only the description in a plain text with no additional formatting."""

def get_page_title(sub_sub_topic: str, keywords: list) -> str:
    return f"""Create an SEO-friendly page title for a PowerPoint template about "{sub_sub_topic}".

Requirements:
- Maximum 65 characters
- Include "{sub_sub_topic}" in the early part
- Include 1-2 power words in the early part
{main_instruction(keywords)}

Format: Return only the title in a plain text with no additional formatting."""

def get_meta_description(sub_sub_topic: str, keywords: list) -> str:
    return f"""Create an SEO-friendly meta description for a PowerPoint template about "{sub_sub_topic}".

Requirements:
- Maximum 165 characters
- Include "{sub_sub_topic}" in the early part
- Include 1-2 power words
{main_instruction(keywords)}

Format: Return only the meta description in a plain text with no additional formatting."""

def get_short_description(sub_sub_topic: str, keywords: list) -> str:
    return f"""Create a short description for a PowerPoint template about "{sub_sub_topic}".

Requirements:
- Exactly 40 words in 2 sentences
- First sentence: Start with capital letter, focus on {sub_sub_topic}
- Second sentence: Start with capital letter, include "{sub_sub_topic}", explain template benefits
{main_instruction(keywords)}
Format: Return only the description in a plain text with no additional formatting."""



def get_long_description(topic: str, keywords: list) -> str:
    return f"""Create a detailed long description for a PowerPoint template about "{topic}".

Requirements:
- Write 60 words in 2 paragraphs
{main_instruction(keywords)}
- Keep all sentences under 10 words

Paragraph 1 (30-40 words):
- Start with a capital letter, describe why {topic} is important for PowerPoint users (include punctuation error)
- Highlight practical challenges related to {topic}
- Discuss real-life presentation challenges that show need for a professional template
- Explain how this template helps solve those challenges

Paragraph 2 (20-30 words):
- Begin with a unique transition phrase about additional {topic} challenges
- Add a CTA

Format: Return only the description in a plain text with no additional formatting."""
