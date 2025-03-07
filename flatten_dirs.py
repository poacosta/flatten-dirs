"""
Utility for flattening nested directory structures.

Promotes all directories to the root level, preserving files and handling naming conflicts.
Optimized for large directory structures (20k+ dirs, 400k+ files).
"""
import concurrent.futures
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='flatten_dirs.log'
)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger('').addHandler(console)


def get_all_dirs_and_files(source_dir: str) -> Tuple[Set[str], Dict[str, List[Tuple[str, str]]]]:
    """
    Get all unique directory names and map them to their files.

    Args:
        source_dir: The root directory to start from

    Returns:
        Tuple containing:
        - Set of unique directory names
        - Dictionary mapping directory names to list of (source_path, filename) tuples
    """
    root = Path(source_dir).resolve()
    unique_dirs = set()
    dir_to_files: Dict[str, List[Tuple[str, str]]] = {}

    print("Scanning directory structure...")
    for dirpath, _, filenames in os.walk(source_dir):
        current_dir = Path(dirpath).resolve()

        # Skip the root directory itself
        if current_dir == root:
            continue

        dir_name = current_dir.name
        unique_dirs.add(dir_name)

        if dir_name not in dir_to_files:
            dir_to_files[dir_name] = []

        # Map files to their target directory name (current directory name)
        for filename in filenames:
            source_path = current_dir / filename
            dir_to_files[dir_name].append((str(source_path), filename))

    return unique_dirs, dir_to_files


def move_file(args: Tuple[str, str, str]) -> Optional[str]:
    """
    Move a single file, handling conflicts.

    Args:
        args: Tuple containing (source, target_dir, filename)

    Returns:
        Error message if an error occurred, None otherwise
    """
    source, destination_dir, filename = args
    source_path = Path(source)

    if not source_path.exists():
        return f"Source file no longer exists: {source}"

    destination = Path(destination_dir) / filename

    # Handle filename conflicts
    counter = 1
    while destination.exists() and source_path != destination:
        name, ext = os.path.splitext(filename)
        destination = Path(destination_dir) / f"{name}_{counter}{ext}"
        counter += 1

    # Skip if source and destination are the same
    if source_path == destination:
        return None

    try:
        shutil.move(str(source_path), str(destination))
        return None
    except (OSError, IOError, shutil.Error) as e:
        return f"Error moving {source_path} to {destination}: {e}"


def process_file_movements(tasks: List[Tuple[str, str, str]], parallel: bool = True,
                           max_workers: int = 4) -> List[str]:
    """
    Process file movements with optional parallelization.

    Args:
        tasks: List of (source, target_dir, filename) tuples
        parallel: Whether to use parallel processing
        max_workers: Number of worker threads for parallel processing

    Returns:
        List of error messages
    """
    total_files = len(tasks)
    print(f"Moving {total_files} files to their new locations...")

    errors = []

    if parallel and total_files > 1000:
        with tqdm(total=total_files) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(move_file, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        errors.append(result)
                    pbar.update(1)
    else:
        for task in tqdm(tasks):
            result = move_file(task)
            if result:
                errors.append(result)

    return errors


def create_root_directories(root_path: Path, dir_names: Set[str]) -> List[str]:
    """
    Create all necessary directories at the root level.

    Args:
        root_path: Path object for the root directory
        dir_names: Set of directory names to create

    Returns:
        List of error messages
    """
    errors = []
    print(f"Creating {len(dir_names)} directories at root level...")

    for dir_name in dir_names:
        target_dir = root_path / dir_name
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            error_msg = f"Failed to create directory {target_dir}: {e}"
            logging.error("%s", error_msg)
            errors.append(error_msg)

    return errors


def cleanup_empty_dirs(source_dir: str) -> int:
    """
    Remove empty directories after file migration.

    Args:
        source_dir: Root directory path

    Returns:
        Number of directories removed
    """
    root = Path(source_dir).resolve()
    empty_dirs_removed = 0

    print("Cleaning up empty directories...")
    for dirpath, _, _ in sorted(os.walk(source_dir, topdown=False),
                                key=lambda x: len(x[0]), reverse=True):
        current_dir = Path(dirpath)

        # Skip root and direct children of root
        if current_dir not in (root, current_dir.parent):
            try:
                if not any(current_dir.iterdir()):
                    os.rmdir(str(current_dir))
                    empty_dirs_removed += 1
            except OSError as e:
                logging.warning("Could not remove directory: %s - %s", current_dir, e)

    return empty_dirs_removed


# pylint: disable=C0301
def prepare_file_tasks(root_path: Path, dir_to_files: Dict[str, List[Tuple[str, str]]]) -> List[Tuple[str, str, str]]:
    """
    Prepare file movement tasks.

    Args:
        root_path: The root directory path
        dir_to_files: Mapping of directory names to files

    Returns:
        List of (source, destination_dir, filename) tuples
    """
    tasks = []
    for dir_name, files in dir_to_files.items():
        destination_dir = root_path / dir_name
        for source, filename in files:
            tasks.append((source, str(destination_dir), filename))
    return tasks


def log_and_print_summary(start_time: float, total_files: int, unique_dirs: int,
                          errors: List[str], empty_dirs: int) -> None:
    """
    Log errors and print operation summary.

    Args:
        start_time: Start time of operation
        total_files: Number of files processed
        unique_dirs: Number of unique directories
        errors: List of error messages
        empty_dirs: Number of empty directories removed
    """
    # Log errors
    if errors:
        logging.warning("Encountered %d errors while moving files", len(errors))
        for error in errors:
            logging.warning("%s", error)

    # Print summary
    elapsed_time = time.time() - start_time
    print(f"Removed {empty_dirs} empty directories")
    print(f"Completed in {elapsed_time:.2f} seconds")
    print(f"Processed {total_files} files across {unique_dirs} directories")
    if errors:
        print(f"Encountered {len(errors)} errors. See log file for details.")
    else:
        print("All operations completed successfully.")


def flatten_directories(source_dir: str, parallel: bool = True, max_workers: int = 4) -> None:
    """
    Flatten all nested directories to the root level with optimizations for large datasets.

    Args:
        source_dir: The root directory to start from
        parallel: Whether to use multithreading for file operations
        max_workers: Maximum number of worker threads if parallel is True
    """
    start_time = time.time()
    root_path = Path(source_dir).resolve()

    # Initial scan to get directory structure
    unique_dirs, dir_to_files = get_all_dirs_and_files(source_dir)

    # Create directories at root level
    create_root_directories(root_path, unique_dirs)

    # Prepare and process file movements
    tasks = prepare_file_tasks(root_path, dir_to_files)
    errors = process_file_movements(tasks, parallel, max_workers)

    # Clean up empty directories
    empty_dirs_removed = cleanup_empty_dirs(source_dir)

    # Show summary
    log_and_print_summary(start_time, len(tasks), len(unique_dirs), errors, empty_dirs_removed)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flatten_dirs.py <directory_path> [--sequential]")
        sys.exit(1)

    use_sequential = "--sequential" in sys.argv
    source_directory = sys.argv[1]

    flatten_directories(source_directory, parallel=not use_sequential)
