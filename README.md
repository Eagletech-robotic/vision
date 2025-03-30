## PIP Environment

Useful commands for managing a pip environment:

- Source the pip environment before running a Python script: `source .venv/bin/activate`
- Freeze dependencies: `pip freeze > requirements.txt`
- Install dependencies: `pip install -r requirements.txt`

## Video commands

- List available video devices:

```bash
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

- Record and save a video:

```bash
ffmpeg -f v4l2 -i /dev/video2 -r 10 -s 640x480 -t 5 output.mp4
```

- Display info about a video file:

```bash
ffprobe output.mp4
```
