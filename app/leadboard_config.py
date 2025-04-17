import json
from pathlib import Path
import argparse

def set_initial_counts():
    """command-line tool to set initial language data point counts"""
    parser = argparse.ArgumentParser(description='setup initial language data point counts for FeeL leaderboard')
    parser.add_argument('--language', type=str, help='Language to set count for')
    parser.add_argument('--count', type=int, help='Count to set for the language')
    parser.add_argument('--list', action='store_true', help='List current counts')
    
    args = parser.parse_args()
    data_file = Path("language_data_points.json")
    
    if data_file.exists():
        with open(data_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("Error reading data file. Creating new data.")
                data = {}
    else:
        data = {}
    
    languages_file = Path("languages.json")
    if languages_file.exists():
        with open(languages_file, "r", encoding="utf-8") as f:
            languages = json.load(f)
    else:
        print("Warning: languages.json not found. Cannot verify language names.")
        languages = {}
    
    # current counts
    if args.list:
        print("Current language data point counts:")
        for lang in sorted(set(list(languages.keys()) + list(data.keys()))):
            count = data.get(lang, 0)
            print(f"{lang}: {count}")
        return
    
    if args.language and args.count is not None:
        if args.language not in languages and languages:
            print(f"Warning: '{args.language}' is not in languages.json")
            confirm = input("Continue anyway? (y/n): ")
            if confirm.lower() != 'y':
                return
        
        data[args.language] = args.count
        
        # saving
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Set count for {args.language} to {args.count}")
    elif not args.list:
        print("Please provide both --language and --count arguments")
        parser.print_help()

if __name__ == "__main__":
    set_initial_counts()