import os

def find_subdirectory(target_subdir):

    parent_dir = os.pardir
        
    for root, dirs, files in os.walk(parent_dir):
            
        if target_subdir in dirs:
                
            dataset_dir = os.path.join(root, target_subdir)
            all_files = files
        
    return dataset_dir, all_files