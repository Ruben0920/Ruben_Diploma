# Input Directory

This directory contains the input files for the Ruben_Diploma pipeline.

## Structure

- `videos/` - Place your political debate video files here
  - Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
  - The pipeline will process all videos in this directory

## Usage

1. Place your debate video files in the `videos/` subdirectory
2. Run the pipeline: `python run_pipeline.py`
3. The pipeline will automatically detect and process all videos

## Example

```
input/
├── videos/
│   ├── debate_2020.mp4
│   ├── presidential_debate.mp4
│   └── town_hall.mp4
└── README.md
```

## Notes

- Video files should be in a common format (MP4 recommended)
- Larger files will take longer to process
- The pipeline supports parallel processing of multiple videos
