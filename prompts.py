def get_subtopic_prompt(main_topic: str, n: int) -> str:
    return f"""Generate {n} unique and specific sub-topics for the main topic "{main_topic}".

Each sub-topic must:

Be directly related to "{main_topic}"
Be specific, actionable, and professional
Be unique and distinct from others
Contain exactly 4 words
Output format:
Sub-Topic 1 ||| Sub-Topic 2 ||| Sub-Topic 3

Important: Return only the sub-topics in plain text, separated by triple pipes (|||). Do not include any additional text, explanations, or formatting.
"""

def get_subsubtopic_prompt(main_topic: str, sub_topic: str, n: int) -> str:
    return f"""Generate {n} unique and specific sub-sub-topics for the topic hierarchy:
Main Topic: {main_topic}
Sub Topic: {sub_topic}

Each sub-sub-topic must meet the following criteria:

Relates specifically to both the main topic and sub-topic
SEO-friendly, professional, and unique
Less than 40 characters
Contains transactional or long-tail LSI keywords
Output format:
Sub-Sub-Topic 1 ||| Sub-Sub-Topic 2 ||| Sub-Sub-Topic 3

Important: Return only the sub-sub-topics in plain text separated by triple pipes (|||). Do not include any additional text, explanations, or formatting.
""" 