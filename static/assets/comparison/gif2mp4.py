from moviepy.editor import VideoFileClip

# 转换单个文件
clip = VideoFileClip("4-animator.gif")
clip.write_videofile("4-animator.mp4", fps=25)

# 批量转换
import os
for file in os.listdir("."):
    if file.endswith(".gif"):
        clip = VideoFileClip(file)
        output_name = file.replace(".gif", ".mp4")
        clip.write_videofile(output_name, fps=25)