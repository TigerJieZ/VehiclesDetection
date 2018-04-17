from tools import *


if __name__ == '__main__':
    # Get image file names

    cars,notcars=get_data()

    car_image = mpimg.imread(cars[5])
    notcar_image = mpimg.imread(notcars[0])

    compare_images(car_image, notcar_image, "Car", "Not Car")

    # 测试hog模块
    hog_test(car_image, notcar_image, pix_per_cell, cell_per_block, spatial_size)

    svc_test(svc, X_scaler, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    import imageio

    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip

    output = 'test_result.mp4'
    clip = VideoFileClip("test_video.mp4")
    video_clip = clip.fl_image(detect_cars)
    video_clip.write_videofile(output, audio=False)

    output = 'result.mp4'
    clip = VideoFileClip("project_video.mp4")
    video_clip = clip.fl_image(detect_cars)
    video_clip.write_videofile(output, audio=False)

