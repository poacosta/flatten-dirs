# flatten-dirs 📁➡️📁📁📁

My personal battle against deeply nested directory structures turned into a Python solution that might save your sanity
too.

## The Directory Nightmare I Solved 😱

We've all been there - you've inherited a file system that looks like a Russian nesting doll:

```
root/
  ├── dir1/
  │   ├── dir2/
  │   │   ├── dir3/
  │   │   │   └── file.txt
  │   │   └── another_file.txt
  │   └── something.txt
  └── other/
      └── ...
```

After fighting with 20,000+ directories and 400,000+ files (yes, really), I created this utility to transform chaotic
hierarchies into something civilized humans can actually work with:

```
root/
  ├── dir1/
  │   └── something.txt
  ├── dir2/
  │   └── another_file.txt
  ├── dir3/
  │   └── file.txt
  └── other/
      └── ...
```

## Core Capabilities Framework 🧰

What started as a desperate late-night script evolved into something quite robust:

- **Speed Demon Mode**: Optimized for ridiculous directory counts without melting your CPU
- **Parallel Universe**: Multithreaded operations because life's too short to wait for sequential I/O
- **Conflict Management**: Handles naming collisions better than I handle my own schedule conflicts
- **Progress Visibility**: Real-time feedback so you don't think your computer died
- **Paper Trail**: Comprehensive logging for when things inevitably go sideways
- **Data Protection**: Designed to prevent accidental digital apocalypses

## Getting Started (It's Actually Easy) 🚀

```bash
# Grab the code
git clone https://github.com/poacosta/flatten-dirs.git
cd flatten-dirs

# Install the one dependency (I kept it minimal)
pip install tqdm
```

## Usage Patterns

### The "Just Make It Work" Approach

```bash
python flatten_dirs.py /path/to/your/directory/nightmare
```

### The "I Don't Trust Computers" Option

```bash
python flatten_dirs.py /path/to/directory --sequential
```

### Real-world Example (From My Digital Cleanup Journey)

```bash
$ python flatten_dirs.py C:\Users\pedro\Documents\Datasets\ --sequential
Scanning directory structure...
Creating 4203 directories at root level...
Moving 155878 files to their new locations...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 155878/155878 [01:30<00:00, 1729.72it/s]
Cleaning up empty directories...
Removed 4119 empty directories
Completed in 101.66 seconds
Processed 155878 files across 4203 directories
All operations completed successfully.
```

## The Magic Behind the Curtain 🧙‍♂️

My approach breaks down what seemed like an impossible task:

1. **Discovery**: First, we map the digital territory (without getting lost)
2. **Preparation**: Create a new, flattened world at the root level
3. **Migration**: Carefully relocate all your digital citizens
4. **Cleanup**: Remove the abandoned neighborhoods

## Performance Realities

Let's talk resource consumption (because we're all adults here):

- **Memory Hunger**: Expect ~100-200MB RAM consumption for 400k files (my coffee habit uses more resources)
- **Thread Management**: Default is 4 threads, which worked for my setup, but adjust to taste
- **Disk I/O**: This will be your bottleneck - SSDs strongly preferred (HDDs will work, but maybe go make a sandwich)
- **Network Drives**: Technically possible but about as pleasant as a root canal

## Known Quirks and Limitations

- Directory name collisions get merged (because flatten means flatten)
- Super-deep directories might need system stack adjustments
- Running it multiple times is like washing already clean dishes - possible but pointless

## The Fine Print

MIT License (Go wild, just don't blame me if things explode)
