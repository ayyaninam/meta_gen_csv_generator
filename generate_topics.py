import pandas as pd
from openai import OpenAI
from typing import List, Tuple, Dict
import config
import prompts
import content_prompts
import logging
from multiprocessing import Pool, cpu_count
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
import os
import shutil
import glob
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_openai():
    return OpenAI(api_key=config.OPENAI_API_KEY)

def call_gpt(client: OpenAI, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.TEMPERATURE
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling GPT API: {str(e)}")
        raise

def generate_subtopics(client: OpenAI, main_topic: str, num_subtopics: int) -> List[str]:
    logging.info(f"Generating {num_subtopics} subtopics for '{main_topic}'")
    prompt = prompts.get_subtopic_prompt(main_topic, num_subtopics)
    response = call_gpt(client, prompt)
    print(response)
    topics = [topic.strip() for topic in response.split("|||")]
    logging.info(f"Generated subtopics: {topics}")
    return topics

def generate_subsubtopics(client: OpenAI, main_topic: str, sub_topic: str, num_subsubtopics: int) -> List[str]:
    logging.info(f"Generating {num_subsubtopics} sub-subtopics for '{sub_topic}'")
    prompt = prompts.get_subsubtopic_prompt(main_topic, sub_topic, num_subsubtopics)
    response = call_gpt(client, prompt)
    print(response)
    topics = [topic.strip() for topic in response.split("|||")]
    logging.info(f"Generated sub-subtopics: {topics}")
    return topics

def generate_keywords(client: OpenAI, main_topic: str, sub_topic: str, sub_sub_topic: str) -> List[str]:
    logging.info(f"Generating keywords for '{sub_sub_topic}'")
    prompt = content_prompts.get_keywords_prompt(main_topic, sub_topic, sub_sub_topic)
    response = call_gpt(client, prompt)
    keywords = [kw.strip() for kw in response.split("|||")]
    all_keywords = config.STATIC_KEYWORDS + keywords
    logging.info(f"Generated keywords: {all_keywords}")
    return all_keywords

def generate_content_sequential(client: OpenAI, main_topic: str, sub_topic: str, sub_sub_topic: str, content_type: str) -> str:
    """Generate a single type of content"""
    try:
        if content_type == 'keywords':
            prompt = content_prompts.get_keywords_prompt(main_topic, sub_topic, sub_sub_topic)
            return call_gpt(client, prompt)
            
        # Get keywords for other content types
        keywords = call_gpt(client, content_prompts.get_keywords_prompt(main_topic, sub_topic, sub_sub_topic))
        keywords_list = [k.strip() for k in keywords.split('|||')]
        
        # Map content types to their prompt functions
        prompt_map = {
            'image_title': content_prompts.get_image_title,
            'image_alt': content_prompts.get_image_alt,
            'image_caption': content_prompts.get_image_caption,
            'image_description': content_prompts.get_image_description,
            'page_title': content_prompts.get_page_title,
            'meta_description': content_prompts.get_meta_description,
            'short_description': content_prompts.get_short_description,
            'long_description': content_prompts.get_long_description
        }
        
        if content_type in prompt_map:
            prompt = prompt_map[content_type](sub_sub_topic, keywords_list)
            return call_gpt(client, prompt)
            
        return ''
        
    except Exception as e:
        logging.error(f"Error generating {content_type}: {str(e)}")
        return f"Error generating {content_type}"

def process_single_column(df: pd.DataFrame, content_type: str, csv_column: str) -> pd.DataFrame:
    """Process a single content column for all rows"""
    client = setup_openai()
    total_rows = len(df)
    
    logging.info(f"\nStarting processing of {csv_column}")
    logging.info(f"Total rows to process: {total_rows}")
    
    # Create a copy of the DataFrame to work with
    working_df = df.copy()
    
    for index, row in working_df.iterrows():
        try:
            # Skip if content already exists
            if pd.notna(row.get(csv_column)) and row.get(csv_column).strip():
                logging.info(f"Row {index + 1}/{total_rows}: Content exists, skipping")
                continue
                
            # Skip if missing required fields
            if not all(pd.notna(row.get(field)) for field in ['Main Topic', 'Sub-Topic', 'Sub-Sub-Topic']):
                logging.info(f"Row {index + 1}/{total_rows}: Missing required fields, skipping")
                continue
                
            logging.info(f"Row {index + 1}/{total_rows}: Generating {csv_column}")
            content = generate_content_sequential(
                client,
                row['Main Topic'],
                row['Sub-Topic'],
                row['Sub-Sub-Topic'],
                content_type
            )
            
            working_df.at[index, csv_column] = content
            
            # Save intermediate results every 10 rows
            if (index + 1) % 10 == 0:
                working_df.to_csv(config.OUTPUT_CSV, index=False)
                logging.info(f"Saved intermediate results after {index + 1} rows")
                
        except Exception as e:
            logging.error(f"Error processing row {index + 1}: {str(e)}")
            continue
            
    return working_df

def cleanup_backup_files():
    """Clean up backup and checkpoint files"""
    try:
        # Remove main backup file
        backup_path = f"{config.OUTPUT_CSV}.backup"
        if os.path.exists(backup_path):
            os.remove(backup_path)
            logging.info(f"Removed backup file: {backup_path}")
            
        # Remove all checkpoint files
        checkpoint_pattern = f"{config.OUTPUT_CSV}.checkpoint_*"
        for checkpoint_file in glob.glob(checkpoint_pattern):
            os.remove(checkpoint_file)
            logging.info(f"Removed checkpoint file: {checkpoint_file}")
            
    except Exception as e:
        logging.warning(f"Error cleaning up backup files: {str(e)}")

def process_subtopics(df: pd.DataFrame, client: OpenAI, num_subtopics: int) -> pd.DataFrame:
    """Process all subtopics first"""
    logging.info("\nProcessing all subtopics...")
    working_df = df.copy()
    
    for index, row in working_df.iterrows():
        main_topic = row['Main Topic']
        if not isinstance(main_topic, str) or not main_topic.strip():
            continue
            
        if pd.isna(row['Sub-Topic']):
            logging.info(f"Generating subtopics for: {main_topic}")
            subtopics = generate_subtopics(client, main_topic, num_subtopics)
            # Store all subtopics in the first row, we'll expand them later
            working_df.at[index, 'Sub-Topic'] = ' ||| '.join(subtopics)
            
            # Save progress
            working_df.to_csv(f"{config.OUTPUT_CSV}.subtopics", index=False)
    
    return working_df

def process_subsubtopics(df: pd.DataFrame, client: OpenAI, num_subsubtopics: int) -> pd.DataFrame:
    """Process all sub-subtopics after subtopics are done"""
    logging.info("\nProcessing all sub-subtopics...")
    results = []
    
    for index, row in df.iterrows():
        main_topic = row['Main Topic']
        if not isinstance(main_topic, str) or not main_topic.strip():
            continue
            
        subtopics = row['Sub-Topic'].split(' ||| ')
        for subtopic_num, subtopic in enumerate(subtopics, 1):
            logging.info(f"Generating sub-subtopics for: {main_topic} -> {subtopic}")
            subsubtopics = generate_subsubtopics(client, main_topic, subtopic, num_subsubtopics)
            
            for subsubtopic_num, subsubtopic in enumerate(subsubtopics, 1):
                new_row = row.copy()
                new_row['Serial Number'] = f"{row['Serial Number']}.{subtopic_num}.{subsubtopic_num}"
                new_row['Sub-Topic'] = subtopic
                new_row['Sub-Sub-Topic'] = subsubtopic
                results.append(new_row)
                
            # Save progress after each subtopic
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"{config.OUTPUT_CSV}.subsubtopics", index=False)
    
    return pd.DataFrame(results)

def process_column(df: pd.DataFrame, client: OpenAI, column_name: str, content_type: str) -> pd.DataFrame:
    """Process a single column for all rows"""
    logging.info(f"\nProcessing column: {column_name}")
    total_rows = len(df)
    
    for index, row in df.iterrows():
        try:
            # Skip if content already exists and is not empty
            if pd.notna(row[column_name]) and str(row[column_name]).strip():
                logging.info(f"Row {index + 1}/{total_rows}: Content exists for {column_name}, skipping")
                continue
                
            # Skip if missing required fields
            if pd.isna(row['Main Topic']) or pd.isna(row['Sub-Topic']) or pd.isna(row['Sub-Sub-Topic']):
                logging.info(f"Row {index + 1}/{total_rows}: Missing required fields, skipping")
                continue
            
            logging.info(f"Processing {column_name} for row {index + 1}/{total_rows}")
            
            try:
                # Generate keywords first if needed
                if content_type != 'keywords':
                    # Try to get existing keywords or generate new ones
                    if pd.isna(row['Keywords']) or not str(row['Keywords']).strip():
                        keywords = generate_keywords(client, row['Main Topic'], row['Sub-Topic'], row['Sub-Sub-Topic'])
                        df.at[index, 'Keywords'] = ' ||| '.join(keywords)
                        keywords_list = keywords
                    else:
                        keywords_list = str(row['Keywords']).split(' ||| ') if pd.notna(row['Keywords']) else config.STATIC_KEYWORDS
                
                # Generate content based on type
                if content_type == 'keywords':
                    content = ' ||| '.join(generate_keywords(client, row['Main Topic'], row['Sub-Topic'], row['Sub-Sub-Topic']))
                else:
                    prompt_map = {
                        'image_title': content_prompts.get_image_title,
                        'image_alt': content_prompts.get_image_alt,
                        'image_caption': content_prompts.get_image_caption,
                        'image_description': content_prompts.get_image_description,
                        'page_title': content_prompts.get_page_title,
                        'meta_description': content_prompts.get_meta_description,
                        'short_description': content_prompts.get_short_description,
                        'long_description': content_prompts.get_long_description
                    }
                    
                    prompt_func = prompt_map[content_type]
                    
                    # Special handling for long description to ensure uniqueness
                    if content_type == 'long_description':
                        # Include hierarchy information in the prompt
                        full_topic = f"{row['Sub-Sub-Topic']} (in the context of {row['Sub-Topic']} under {row['Main Topic']})"
                        content = call_gpt(client, prompt_func(full_topic, keywords_list))
                        
                        # Verify the content is unique
                        existing_descriptions = df[df['Long Description'].notna()]['Long Description'].tolist()
                        similarity_threshold = 0.7  # Adjust this value if needed
                        
                        # If content is too similar to existing descriptions, regenerate with higher temperature
                        retry_count = 0
                        while any(similar_text(content, existing) > similarity_threshold for existing in existing_descriptions) and retry_count < 3:
                            logging.info(f"Generated content too similar to existing, retrying with higher temperature ({retry_count + 1}/3)")
                            response = client.chat.completions.create(
                                model=config.MODEL_NAME,
                                messages=[{"role": "user", "content": prompt_func(full_topic, keywords_list)}],
                                temperature=min(1.0, config.TEMPERATURE + 0.1 * (retry_count + 1))
                            )
                            content = response.choices[0].message.content.strip()
                            retry_count += 1
                    else:
                        content = call_gpt(client, prompt_func(row['Sub-Sub-Topic'], keywords_list))
                
                df.at[index, column_name] = content
                
                # Save progress every 5 rows
                if (index + 1) % 5 == 0:
                    df.to_csv(config.OUTPUT_CSV, index=False)
                    logging.info(f"Saved progress after {index + 1} rows")
                
            except Exception as e:
                logging.error(f"Error generating content for row {index + 1} in {column_name}: {str(e)}")
                continue
            
        except Exception as e:
            logging.error(f"Error processing row {index + 1} for {column_name}: {str(e)}")
            continue
    
    return df

def similar_text(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using simple word overlap"""
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

def process_csv(num_subtopics: int, num_subsubtopics: int):
    try:
        # Read or create DataFrame
        try:
            df = pd.read_csv(config.OUTPUT_CSV)
            logging.info(f"Found existing output file with {len(df)} rows")
            # Create backup
            shutil.copy2(config.OUTPUT_CSV, f"{config.OUTPUT_CSV}.backup")
            logging.info(f"Created backup at: {config.OUTPUT_CSV}.backup")
        except FileNotFoundError:
            df = pd.read_csv(config.INPUT_CSV)
            logging.info("Starting with fresh input file")
            
        # Ensure all required columns exist
        required_columns = [
            'Main Topic', 'Sub-Topic', 'Sub-Sub-Topic',
            'Keywords', 'Image Title', 'Image Alt Text',
            'Image Caption', 'Image Description', 'Page Title',
            'Page Meta Description', 'Short Description', 'Long Description'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''

        client = setup_openai()

        # First, check if we need to generate new topics
        if config.PROCESS_NEW_TOPICS and (num_subtopics > 0 or num_subsubtopics > 0):
            new_rows = []
            
            # Get existing combinations to avoid duplicates
            existing_combinations = set()
            for _, row in df.iterrows():
                if pd.notna(row['Sub-Topic']) and pd.notna(row['Sub-Sub-Topic']):
                    existing_combinations.add((row['Main Topic'], row['Sub-Topic'], row['Sub-Sub-Topic']))
            
            # Process each main topic for new combinations
            for index, row in df.iterrows():
                main_topic = row['Main Topic']
                if not isinstance(main_topic, str) or not main_topic.strip():
                    continue
                
                # If row has no sub-topic, generate everything
                if pd.isna(row['Sub-Topic']) or not row['Sub-Topic'].strip():
                    logging.info(f"\nProcessing Main Topic: {main_topic}")
                    subtopics = generate_subtopics(client, main_topic, num_subtopics)
                    
                    for i, subtopic in enumerate(subtopics, 1):
                        subsubtopics = generate_subsubtopics(client, main_topic, subtopic, num_subsubtopics)
                        
                        for j, subsubtopic in enumerate(subsubtopics, 1):
                            if (main_topic, subtopic, subsubtopic) not in existing_combinations:
                                new_row = row.copy()
                                new_row['Serial Number'] = f"{row['Serial Number']}.{i}.{j}"
                                new_row['Sub-Topic'] = subtopic
                                new_row['Sub-Sub-Topic'] = subsubtopic
                                new_rows.append(new_row)
                                existing_combinations.add((main_topic, subtopic, subsubtopic))
                
                # If row has sub-topic but no sub-sub-topic, generate only sub-sub-topics
                elif pd.isna(row['Sub-Sub-Topic']) or not row['Sub-Sub-Topic'].strip():
                    subtopic = row['Sub-Topic']
                    logging.info(f"\nGenerating sub-sub-topics for: {main_topic} -> {subtopic}")
                    subsubtopics = generate_subsubtopics(client, main_topic, subtopic, num_subsubtopics)
                    
                    for j, subsubtopic in enumerate(subsubtopics, 1):
                        if (main_topic, subtopic, subsubtopic) not in existing_combinations:
                            new_row = row.copy()
                            new_row['Serial Number'] = f"{row['Serial Number']}.{j}"
                            new_row['Sub-Sub-Topic'] = subsubtopic
                            new_rows.append(new_row)
                            existing_combinations.add((main_topic, subtopic, subsubtopic))
            
            # Add all new rows and remove original empty rows
            if new_rows:
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                # Remove rows that don't have both Sub-Topic and Sub-Sub-Topic
                df = df.dropna(subset=['Sub-Topic', 'Sub-Sub-Topic'])
                # Save progress
                df.to_csv(config.OUTPUT_CSV, index=False)
                logging.info(f"Added {len(new_rows)} new topic combinations")

        # Generate keywords for all rows that have sub-topics and sub-sub-topics but no keywords
        logging.info("\nGenerating keywords for existing topics...")
        rows_updated = 0
        
        for index, row in df.iterrows():
            try:
                # Check if row has both sub-topic and sub-sub-topic
                if pd.notna(row['Sub-Topic']) and pd.notna(row['Sub-Sub-Topic']):
                    # Check if keywords are missing or empty
                    if pd.isna(row.get('Keywords', '')) or not str(row.get('Keywords', '')).strip():
                        main_topic = str(row['Main Topic'])
                        sub_topic = str(row['Sub-Topic'])
                        sub_sub_topic = str(row['Sub-Sub-Topic'])
                        
                        logging.info(f"Generating keywords for: {main_topic} -> {sub_topic} -> {sub_sub_topic}")
                        
                        # Generate keywords
                        keywords = generate_keywords(client, main_topic, sub_topic, sub_sub_topic)
                        df.at[index, 'Keywords'] = ' ||| '.join(keywords)
                        rows_updated += 1
                        
                        # Save progress every 5 rows
                        if rows_updated % 5 == 0:
                            df.to_csv(config.OUTPUT_CSV, index=False)
                            logging.info(f"Saved progress after updating {rows_updated} rows")
                            
            except Exception as e:
                logging.error(f"Error generating keywords for row {index + 1}: {str(e)}")
                continue

        # Save final progress after keywords generation
        if rows_updated > 0:
            df.to_csv(config.OUTPUT_CSV, index=False)
            logging.info(f"Completed generating keywords for {rows_updated} rows")
            
        # Step 2: Process each content column one by one
        content_columns = [
            ('image_title', 'Image Title'),
            ('image_alt', 'Image Alt Text'),
            ('image_caption', 'Image Caption'),
            ('image_description', 'Image Description'),
            ('page_title', 'Page Title'),
            ('meta_description', 'Page Meta Description'),
            ('short_description', 'Short Description'),
            ('long_description', 'Long Description')
        ]
        
        for content_type, column_name in content_columns:
            if config.GENERATE_CONTENT[content_type]:
                df = process_column(df, client, column_name, content_type)
                # Save after completing each column
                df.to_csv(config.OUTPUT_CSV, index=False)
                logging.info(f"Completed processing column: {column_name}")
        
        logging.info("\nAll content generation completed")
        
        # Clean up backup files
        cleanup_backup_files()
        logging.info("Cleaned up all backup and checkpoint files")
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        # Restore from backup if exists
        backup_path = f"{config.OUTPUT_CSV}.backup"
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, config.OUTPUT_CSV)
            logging.info("Restored from backup due to error")
        raise

def main():
    logging.info("Starting Content Generator")
    
    try:
        # Check if arguments are provided via command line
        if len(sys.argv) > 2:
            num_subtopics = int(sys.argv[1])
            num_subsubtopics = int(sys.argv[2])
        else:
            # If no command line arguments, ask for input
            num_subtopics = int(input("How many Sub-Topics do you want to generate per main topic? "))
            num_subsubtopics = int(input("How many Sub-Sub-Topics do you want to generate per sub-topic? "))
        
        logging.info(f"Generating {num_subtopics} sub-topics and {num_subsubtopics} sub-sub-topics per topic")
        
        process_csv(num_subtopics, num_subsubtopics)
        
    except ValueError:
        logging.error("Invalid input: Please enter valid numbers for the number of topics.")
        print("Please enter valid numbers for the number of topics.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 