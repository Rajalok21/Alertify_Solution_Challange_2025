import subprocess

def extract_audio(video_path, audio_path):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path]
    subprocess.run(command, check=True)

video_path = 'https://scontent.cdninstagram.com/o1/v/t16/f2/m69/An_PKbEZ9DkXGfprZdFaXjTMSmprKKR5bYBytYlsvIJIIkAJqK4mz-UdEHt8f4Vqwzb6MgqCFETp9QA3z9Iy6pjX.mp4?stp=dst-mp4&efg=eyJxZV9ncm91cHMiOiJbXCJpZ193ZWJfZGVsaXZlcnlfdnRzX290ZlwiXSIsInZlbmNvZGVfdGFnIjoidnRzX3ZvZF91cmxnZW4uY2xpcHMuYzIuMTA4MC5iYXNlbGluZSJ9&_nc_cat=106&vs=2300915763585193_4068694443&_nc_vs=HBksFQIYOnBhc3N0aHJvdWdoX2V2ZXJzdG9yZS9HSmlxRHhPNF96NTZKcFVFQUNESXBCWDVOdjFQYnBSMUFBQUYVAALIAQAVAhg6cGFzc3Rocm91Z2hfZXZlcnN0b3JlL0dFblhSQnRoRk1CbkhfVUNBSHgwWkpEMUJUZ1BicV9FQUFBRhUCAsgBACgAGAAbABUAACau%2FI31qa6OQBUCKAJDMywXQEDmZmZmZmYYFmRhc2hfYmFzZWxpbmVfMTA4MHBfdjERAHX%2BBwA%3D&ccb=9-4&oh=00_AYCkbnzrBD-ftaMwwq4_D0ew5gXJOllgSFMzCPnxUMr9vA&oe=66D67ADB&_nc_sid=8f1549'
audio_path = 'output_audio.mp3'