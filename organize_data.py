import os
import shutil


source_dir = "C:/Users\HP\Downloads/archive (7)/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data" 



target_dir = "data/train"

# the 4 emotions i am using for this rubric
emotions_map = {
    'angry': 'Angry',
    'happy': 'Happy',
    'sad': 'Sad',
    'neutral': 'Neutral'
}

print(" Starting automated data organization...")

# creating the target folders in VS Code
for target_folder in emotions_map.values():
    os.makedirs(os.path.join(target_dir, target_folder), exist_ok=True)

counters = {folder: 0 for folder in emotions_map.values()}
max_files = 400 # grabbing 400 of each

if not os.path.exists(source_dir):
    print(f"Error: Could not find the source folder at {source_dir}. Check your path!")
else:
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        
        if os.path.isdir(folder_path):
            for key, target_emotion in emotions_map.items():
                if key in folder_name.lower():
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith('.wav') and counters[target_emotion] < max_files:
                            source_file = os.path.join(folder_path, file_name)
                            target_file = os.path.join(target_dir, target_emotion, file_name)
                            
                            shutil.copy(source_file, target_file)
                            counters[target_emotion] += 1
                    break

    print("Success! Your dataset is perfectly organized in VS Code.")
    for emotion, count in counters.items():
        print(f"   - {emotion}: {count} files copied.")