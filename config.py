OPENAI_API_KEY = "YOUR KEY "

# OpenAI Model Settings
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.9

# Processing Settings
BATCH_SIZE = 5  # Number of items to generate in a single API call

# Static Keywords (These will be included in all generations)
STATIC_KEYWORDS = [
    "PowerPoint template",
    "professional presentation",
    "ppt templates",
    "template for ppt",
    "ppt presentation",
    "ppt slides",
    "download ppt",
]

# Input/Output Files
INPUT_CSV = "Complete_Topics_and_Subtopics.csv"
OUTPUT_CSV = "Generated_Topics.csv"

# Content Generation Controls
PROCESS_NEW_TOPICS = True  # Set to False to only process existing topics

# Content Generation Toggles - Set which content to generate
GENERATE_CONTENT = {
    'keywords': True,
    'image_title': True,
    'image_alt': True,
    'image_caption': True,
    'image_description': True,
    'page_title': True,
    'meta_description': True,
    'short_description': True,
    'long_description': True
}


