#!/usr/bin/env python3
"""
File backup utility for Clinical Risk Modeling Engine
Creates backups of files before modifications to track changes
"""
import os
import shutil
import datetime
from pathlib import Path


def create_backup(file_path):
    """
    Creates a backup of a file before modification
    Backup is stored in a .backup directory with timestamp
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Warning: File {file_path} does not exist, skipping backup")
        return
    
    # Create backup directory
    backup_dir = Path(".backup")
    backup_dir.mkdir(exist_ok=True)
    
    # Create timestamped backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.name}.{timestamp}"
    backup_path = backup_dir / backup_name
    
    # Copy the file to backup location
    shutil.copy2(file_path, backup_path)
    print(f"Backup created: {backup_path}")


def restore_backup(backup_path, target_path):
    """
    Restores a file from backup
    """
    backup_path = Path(backup_path)
    target_path = Path(target_path)
    
    if not backup_path.exists():
        print(f"Error: Backup {backup_path} does not exist")
        return False
    
    shutil.copy2(backup_path, target_path)
    print(f"Restored: {backup_path} -> {target_path}")
    return True


def list_backups(file_name=None):
    """
    Lists available backups
    """
    backup_dir = Path(".backup")
    if not backup_dir.exists():
        print("No backups directory found")
        return []
    
    backups = []
    for backup_file in backup_dir.iterdir():
        if file_name is None or backup_file.name.startswith(file_name):
            backups.append(backup_file)
    
    backups.sort(key=lambda x: x.name, reverse=True)
    
    for backup in backups:
        print(backup.name)
    
    return backups


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python backup_utility.py create <file_path>     # Create backup")
        print("  python backup_utility.py restore <backup> <target>  # Restore backup") 
        print("  python backup_utility.py list [file_name]       # List backups")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create" and len(sys.argv) == 3:
        create_backup(sys.argv[2])
    elif command == "restore" and len(sys.argv) == 4:
        restore_backup(sys.argv[2], sys.argv[3])
    elif command == "list":
        file_name = sys.argv[2] if len(sys.argv) == 3 else None
        list_backups(file_name)
    else:
        print("Invalid command or arguments")
        sys.exit(1)