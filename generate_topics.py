import pandas as pd
from openai import OpenAI
from typing import List, Tuple, Dict, Any, Optional
import config
import prompts
import content_prompts
import logging
from multiprocessing import Pool, cpu_count, Manager
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import os
import shutil
import glob
import sys
import tqdm
import numpy as np

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
    topics = [topic.strip() for topic in response.split("|||")]
    return topics

def generate_subsubtopics(client: OpenAI, main_topic: str, sub_topic: str, num_subsubtopics: int) -> List[str]:
    logging.info(f"Generating {num_subsubtopics} sub-subtopics for '{sub_topic}'")
    prompt = prompts.get_subsubtopic_prompt(main_topic, sub_topic, num_subsubtopics)
    response = call_gpt(client, prompt)
    topics = [topic.strip() for topic in response.split("|||")]
    return topics

def generate_keywords(client: OpenAI, main_topic: str, sub_topic: str, sub_sub_topic: str) -> List[str]:
    logging.info(f"Generating keywords for '{sub_sub_topic}'")
    prompt = content_prompts.get_keywords_prompt(main_topic, sub_topic, sub_sub_topic)
    response = call_gpt(client, prompt)
    keywords = [kw.strip() for kw in response.split("|||")]
    all_keywords = config.STATIC_KEYWORDS + keywords
    return all_keywords

def generate_content_item(client: OpenAI, main_topic: str, sub_topic: str, sub_sub_topic: str, content_type: str, existing_keywords: Optional[List[str]] = None) -> Tuple[str, Optional[List[str]]]:
    """Generate a single content item and return it along with any generated keywords"""
    try:
        # For keywords content type
        if content_type == 'keywords':
            keywords = generate_keywords(client, main_topic, sub_topic, sub_sub_topic)
            return ' ||| '.join(keywords), keywords
            
        # For other content types, ensure we have keywords
        keywords_list = existing_keywords
        if not keywords_list:
            keywords_list = generate_keywords(client, main_topic, sub_topic, sub_sub_topic)
            
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
            # Special handling for long description to ensure uniqueness
            if content_type == 'long_description':
                full_topic = f"{sub_sub_topic} (in the context of {sub_topic} under {main_topic})"
                prompt = prompt_map[content_type](full_topic, keywords_list)
            else:
                prompt = prompt_map[content_type](sub_sub_topic, keywords_list)
                
            content = call_gpt(client, prompt)
            return content, keywords_list
            
        return '', keywords_list
        
    except Exception as e:
        logging.error(f"Error generating {content_type}: {str(e)}")
        return f"Error generating {content_type}", None

def process_subtopic_task(args) -> List[str]:
    """Worker function for parallel subtopic generation"""
    main_topic, num_subtopics = args
    client = setup_openai()
    return generate_subtopics(client, main_topic, num_subtopics)

def process_subsubtopic_task(args) -> Tuple[str, str, List[str]]:
    """Worker function for parallel sub-subtopic generation"""
    main_topic, sub_topic, num_subsubtopics = args
    client = setup_openai()
    subsubtopics = generate_subsubtopics(client, main_topic, sub_topic, num_subsubtopics)
    return main_topic, sub_topic, subsubtopics

def process_content_generation_task(args) -> Dict[str, Any]:
    """Worker function for parallel content generation for a single row"""
    row_dict, content_types_to_generate, column_name_map = args
    
    client = setup_openai()
    results = row_dict.copy()
    
    # Skip rows with missing required fields
    if not all(pd.notna(row_dict.get(field)) for field in ['Main Topic', 'Sub-Topic', 'Sub-Sub-Topic']):
        return results
    
    main_topic = row_dict['Main Topic']
    sub_topic = row_dict['Sub-Topic']
    sub_sub_topic = row_dict['Sub-Sub-Topic']
    
    # Extract existing keywords if available
    existing_keywords = None
    if pd.notna(row_dict.get('Keywords')):
        existing_keywords = row_dict['Keywords'].split(' ||| ') if isinstance(row_dict['Keywords'], str) else None
    
    # Process each content type
    for content_type in content_types_to_generate:
        column_name = column_name_map[content_type]
        
        # Skip if content already exists
        if pd.notna(results.get(column_name)) and str(results.get(column_name)).strip():
            continue
        
        # Generate the content
        content, keywords = generate_content_item(
            client, 
            main_topic, 
            sub_topic, 
            sub_sub_topic, 
            content_type, 
            existing_keywords
        )
        
        # Update results
        results[column_name] = content
        
        # If keywords were generated, update the Keywords column and our cached keywords
        if content_type == 'keywords' and keywords:
            results['Keywords'] = ' ||| '.join(keywords)
            existing_keywords = keywords
        elif not existing_keywords and keywords:
            existing_keywords = keywords
            results['Keywords'] = ' ||| '.join(keywords)
    
    return results

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

def generate_topics_in_parallel(df: pd.DataFrame, num_subtopics: int, num_subsubtopics: int) -> pd.DataFrame:
    """Generate all topics in parallel"""
    logging.info("\nGenerating topics in parallel...")
    
    # Filter for rows that need subtopics
    main_topics_to_process = []
    for index, row in df.iterrows():
        main_topic = row['Main Topic']
        if isinstance(main_topic, str) and main_topic.strip() and pd.isna(row.get('Sub-Topic')):
            main_topics_to_process.append((main_topic, num_subtopics))
    
    if not main_topics_to_process:
        logging.info("No main topics need subtopic generation. Skipping.")
        return df
        
    logging.info(f"Generating subtopics for {len(main_topics_to_process)} main topics in parallel")
    
    # Process subtopics in parallel
    num_workers = min(max(1, cpu_count()), len(main_topics_to_process))
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_subtopic_task, args) for args in main_topics_to_process]
        
        for i, future in enumerate(as_completed(futures)):
            main_topic = main_topics_to_process[i][0]
            try:
                subtopics = future.result()
                logging.info(f"Generated {len(subtopics)} subtopics for '{main_topic}'")
                
                # Filter the main topic row
                main_topic_row = None
                for _, row in df.iterrows():
                    if row['Main Topic'] == main_topic and pd.isna(row.get('Sub-Topic')):
                        main_topic_row = row
                        break
                
                if main_topic_row is not None:
                    # Now process subsubtopics for this main topic
                    subsubtopic_tasks = [(main_topic, subtopic, num_subsubtopics) for subtopic in subtopics]
                    
                    # Make sure we have at least one worker
                    sub_workers = min(max(1, cpu_count()), len(subsubtopic_tasks))
                    
                    sub_futures = []
                    with ProcessPoolExecutor(max_workers=sub_workers) as sub_executor:
                        sub_futures = [sub_executor.submit(process_subsubtopic_task, args) for args in subsubtopic_tasks]
                    
                        for j, sub_future in enumerate(as_completed(sub_futures)):
                            try:
                                main_t, subtopic, subsubtopics = sub_future.result()
                                
                                for k, subsubtopic in enumerate(subsubtopics):
                                    new_row = main_topic_row.copy()
                                    new_row['Serial Number'] = f"{new_row['Serial Number']}.{j+1}.{k+1}"
                                    new_row['Sub-Topic'] = subtopic
                                    new_row['Sub-Sub-Topic'] = subsubtopic
                                    results.append(new_row)
                                    
                            except Exception as e:
                                logging.error(f"Error processing subsubtopics: {str(e)}")
            
            except Exception as e:
                logging.error(f"Error processing subtopics for '{main_topic}': {str(e)}")
    
    # Add new rows to dataframe
    if results:
        new_rows_df = pd.DataFrame(results)
        df = pd.concat([df, new_rows_df], ignore_index=True)
        
        # Save progress
        df.to_csv(config.OUTPUT_CSV, index=False)
        logging.info(f"Added {len(results)} new rows with generated topics")
    else:
        logging.info("No new topics were generated.")
    
    return df

def process_content_in_parallel(df: pd.DataFrame) -> pd.DataFrame:
    """Process all content generation in parallel"""
    logging.info("\nProcessing content generation in parallel...")
    
    # Define column mapping
    column_mapping = {
        'keywords': 'Keywords',
        'image_title': 'Image Title',
        'image_alt': 'Image Alt Text',
        'image_caption': 'Image Caption',
        'image_description': 'Image Description',
        'page_title': 'Page Title',
        'meta_description': 'Page Meta Description',
        'short_description': 'Short Description',
        'long_description': 'Long Description'
    }
    
    # Determine which content types to generate
    content_types_to_generate = [k for k, v in config.GENERATE_CONTENT.items() if v]
    
    # Skip processing if nothing to generate
    if not content_types_to_generate:
        logging.info("No content types selected for generation. Skipping.")
        return df
    
    # Convert DataFrame to list of dictionaries for parallel processing
    rows_to_process = df.to_dict('records')
    
    # Check if we have rows to process
    if not rows_to_process:
        logging.info("No rows to process. Skipping content generation.")
        return df
    
    # Prepare tasks
    tasks = [(row, content_types_to_generate, column_mapping) for row in rows_to_process]
    
    # Determine number of workers (don't exceed available CPUs or number of tasks)
    num_workers = min(max(config.MAX_PROCESSES, cpu_count()), len(tasks))
    
    logging.info(f"Processing {len(tasks)} rows with {num_workers} workers")
    
    # Create progress bar
    progress_bar = tqdm.tqdm(total=len(tasks), desc="Processing rows")
    
    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_content_generation_task, task): i for i, task in enumerate(tasks)}
        
        # Process results as they complete
        for future in as_completed(future_to_task):
            task_index = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                # Update progress bar
                progress_bar.update(1)
                
                # Save intermediate results periodically
                if len(results) % 10 == 0:
                    # Create a temporary dataframe from results processed so far
                    temp_df = pd.DataFrame(results)
                    # Save to a temporary file to avoid conflicts
                    temp_filename = f"{config.OUTPUT_CSV}.temp"
                    temp_df.to_csv(temp_filename, index=False)
                    # Rename to actual output file
                    shutil.move(temp_filename, config.OUTPUT_CSV)
                    logging.info(f"Saved intermediate results after processing {len(results)} rows")
                    
            except Exception as e:
                logging.error(f"Error processing row {task_index}: {str(e)}")
                # Add the original row to maintain the count
                results.append(tasks[task_index][0])
    
    progress_bar.close()
    
    # Create final dataframe from results
    final_df = pd.DataFrame(results)
    
    return final_df

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
            # Ensure the input file exists
            if not os.path.exists(config.INPUT_CSV):
                logging.error(f"Input file {config.INPUT_CSV} not found!")
                # Create an empty DataFrame with required columns
                df = pd.DataFrame(columns=['Serial Number', 'Main Topic', 'Sub-Topic', 'Sub-Sub-Topic'])
                logging.info("Created empty DataFrame with required columns")
            else:
                df = pd.read_csv(config.INPUT_CSV)
                logging.info(f"Starting with fresh input file containing {len(df)} rows")
            
        # Ensure all required columns exist
        required_columns = [
            'Serial Number', 'Main Topic', 'Sub-Topic', 'Sub-Sub-Topic',
            'Keywords', 'Image Title', 'Image Alt Text',
            'Image Caption', 'Image Description', 'Page Title',
            'Page Meta Description', 'Short Description', 'Long Description'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
                
        # Ensure the output directory exists
        output_dir = os.path.dirname(config.OUTPUT_CSV)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
                
        # Check if we have main topics to process
        if 'Main Topic' not in df.columns or df['Main Topic'].isna().all():
            logging.warning("No main topics found in the input file!")
            # If we still want to proceed, create a sample main topic
            if config.PROCESS_NEW_TOPICS and num_subtopics > 0:
                sample_topic = "PowerPoint Templates"
                df = pd.DataFrame([{'Serial Number': '1', 'Main Topic': sample_topic}])
                logging.info(f"Created sample main topic: {sample_topic}")
            else:
                logging.error("Cannot proceed without main topics and PROCESS_NEW_TOPICS is disabled")
                return

        # Generate new topics if needed
        if config.PROCESS_NEW_TOPICS and (num_subtopics > 0 or num_subsubtopics > 0):
            df = generate_topics_in_parallel(df, num_subtopics, num_subsubtopics)
            
            # Remove rows that don't have both Sub-Topic and Sub-Sub-Topic
            df = df.dropna(subset=['Sub-Topic', 'Sub-Sub-Topic'])
            
            # Save progress
            df.to_csv(config.OUTPUT_CSV, index=False)
            logging.info("Completed topic generation phase")

        # Process all content in parallel
        df = process_content_in_parallel(df)
        
        # Save final results
        df.to_csv(config.OUTPUT_CSV, index=False)
        logging.info("All content generation completed")
        
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