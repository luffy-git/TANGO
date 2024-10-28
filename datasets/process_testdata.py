import os
import numpy as np
import wave
from moviepy.editor import VideoFileClip


def split_npz(npz_path, output_prefix):
    try:
        # Load the npz file
        data = np.load(npz_path)

        # Get the arrays and split them along the time dimension (T)
        poses = data["poses"]
        betas = data["betas"]
        expressions = data["expressions"]
        trans = data["trans"]

        # Determine the halfway point (T/2)
        half = poses.shape[0] // 2

        # Save the first half (0-5 seconds)
        np.savez(
            output_prefix + "_0_5.npz",
            betas=betas[:half],
            poses=poses[:half],
            expressions=expressions[:half],
            trans=trans[:half],
            model=data["model"],
            gender=data["gender"],
            mocap_frame_rate=data["mocap_frame_rate"],
        )

        # Save the second half (5-10 seconds)
        np.savez(
            output_prefix + "_5_10.npz",
            betas=betas[half:],
            poses=poses[half:],
            expressions=expressions[half:],
            trans=trans[half:],
            model=data["model"],
            gender=data["gender"],
            mocap_frame_rate=data["mocap_frame_rate"],
        )

        print(f"NPZ split saved for {output_prefix}")
    except Exception as e:
        print(f"Error processing NPZ file {npz_path}: {e}")


def split_wav(wav_path, output_prefix):
    try:
        with wave.open(wav_path, "rb") as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(wav_file.getnframes())
            half_frame = len(frames) // 2

            # Create two half files
            for i, start_frame in enumerate([0, half_frame]):
                with wave.open(f"{output_prefix}_{i*5}_{(i+1)*5}.wav", "wb") as out_wav:
                    out_wav.setparams(params)
                    if i == 0:
                        out_wav.writeframes(frames[:half_frame])
                    else:
                        out_wav.writeframes(frames[half_frame:])
        print(f"WAV split saved for {output_prefix}")
    except Exception as e:
        print(f"Error processing WAV file {wav_path}: {e}")


def split_mp4(mp4_path, output_prefix):
    try:
        clip = VideoFileClip(mp4_path)
        for i in range(2):
            subclip = clip.subclip(i * 5, (i + 1) * 5)
            subclip.write_videofile(f"{output_prefix}_{i*5}_{(i+1)*5}.mp4", codec="libx264", audio_codec="aac")
        print(f"MP4 split saved for {output_prefix}")
    except Exception as e:
        print(f"Error processing MP4 file {mp4_path}: {e}")


def process_files(root_dir, output_dir):
    import json

    clips = []
    dirs = os.listdir(root_dir)
    for dir in dirs:
        video_id = dir
        root = os.path.join(root_dir, dir)

        clip = {
            "video_id": video_id,
            "video_path": root,
            "audio_path": root,
            "motion_path": root,
            "mode": "test",
            "start_idx": 0,
            "end_idx": 150,
        }
        clips.append(clip)

    output_json = output_dir + "/test.json"
    with open(output_json, "w") as f:
        json.dump(clips, f, indent=4)


# Set the root directory path of your dataset and output directory
root_dir = "/content/oliver/oliver/Abortion_Laws_-_Last_Week_Tonight_with_John_Oliver_HBO-DRauXXz6t0Y.webm/test/"
output_dir = "/content/test"

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all the files
process_files(root_dir, output_dir)
