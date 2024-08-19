import cv2
import numpy as np
import os
import zipfile
import uuid
import gradio as gr
import uuid


def remove_watermark_area(original_image, text_mask_path):
    # Ensure the mask is binary
    text_mask = cv2.imread(text_mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(text_mask, 1, 255, cv2.THRESH_BINARY)

    # Resize the mask to match the size of the original image area
    mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

    # Expand the mask to cover more area if needed
    kernel = np.ones((5, 5), np.uint8)
    expanded_mask = cv2.dilate(mask_resized, kernel, iterations=1)

    # Inpainting using the mask
    inpainted_image = cv2.inpaint(original_image, expanded_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Optionally apply post-processing to improve results
    cleaned_image = cv2.GaussianBlur(inpainted_image, (3, 3), 0)

    return cleaned_image
from PIL import Image

def remove_watermark(image_path,file_type="",saved_path=""):
    # file_type="pil"
    # file_type="opencv"
    # file_type="filepath"
    if file_type=="filepath":
        # Load the image using OpenCV
        image = cv2.imread(image_path) 
    if file_type=="pil":
        image = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
    if file_type=="opencv":
      image=image_path
    # cv2.imwrite("test.jpg",image)
    image=cv2.resize(image,(1280,1280))
    # Define the area of the watermark (adjust this based on the watermark size)
    height, width, _ = image.shape
    watermark_width = 185  # Adjust based on your watermark size
    watermark_height = 185  # Adjust based on your watermark size
    x_start = 50
    y_start = height - watermark_height+17
    x_end = watermark_width-17
    y_end = height-50

    # Extract the watermark area
    watermark_area = image[y_start:y_end, x_start:x_end]
    # cv2.imwrite('watermark_area.jpg', watermark_area)

    # Create the mask for the watermark area
    # text_mask_path = 'watermark_mask.png'
    text_mask_path ='./mask/mask_1.png'
    # text_mask_path ='./mask/mask_2.png'
    cleaned_image = remove_watermark_area(watermark_area, text_mask_path)
    # cv2.imwrite('cleaned_watermark.jpg', cleaned_image)
    # Paste back the cleaned watermark on the original image
    image[y_start:y_end, x_start:x_end] = cleaned_image
    if saved_path=="":
      pass
    else:
      cv2.imwrite(saved_path, image)
    return image

def make_zip(image_list):
    zip_path = f"./temp/{uuid.uuid4().hex[:6]}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for image in image_list:
            zipf.write(image, os.path.basename(image))
    return zip_path

def random_image_name():
    """Generate a random image name."""
    return str(uuid.uuid4())[:8]

    
def process_file(pil_image):
    saved_path = f"./temp/{random_image_name()}.jpg"
    remove_watermark(pil_image,"pil",saved_path)
    return saved_path, saved_path


def process_files(image_files):
    image_list = []
    if len(image_files) == 1:
        # saved_path = os.path.basename(image_files[0])
        # saved_path = f"./temp/{saved_path}"
        saved_path = f"./temp/{random_image_name()}.jpg"
        remove_watermark(image_files[0],"filepath", saved_path)
        return saved_path, saved_path
    else:
        for image_path in image_files:
            # saved_path = os.path.basename(image_path)
            # saved_path = f"./temp/{saved_path}"
            saved_path = f"./temp/{random_image_name()}.jpg"
            remove_watermark(image_path,"filepath",saved_path)
            image_list.append(saved_path)
        zip_path = make_zip(image_list)
        return zip_path,None


import cv2
import numpy as np



def process_video(input_video_path):
    print(input_video_path)
    # output_video_path=""
    # if output_video_path=="":
    output_video_path=f"./temp/{random_image_name()}.mp4"
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You might try 'XVID' or 'H264' if issues persist
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        no_watermark_frame=remove_watermark(frame,"opencv")
        
        # Ensure the frame has the same size and type
        if no_watermark_frame.shape[1] != width or no_watermark_frame.shape[0] != height:
            no_watermark_frame = cv2.resize(no_watermark_frame, (width, height))
        
        video_writer.write(no_watermark_frame)
    
    cap.release()
    video_writer.release()
    return output_video_path,output_video_path
    
if not os.path.exists("./temp"):
    os.mkdir("./temp")


meta_examples = ["./images/7.jpg","./images/6.jpg","./images/1.jpg", "./images/2.jpg", "./images/3.jpg", "./images/4.jpg", "./images/5.jpg"]

gradio_input=[gr.Image(label='Upload an Image')]
gradio_Output=[gr.File(label='Download Image'),gr.Image(label='Display Image')]
gradio_interface = gr.Interface(fn=process_file, inputs=gradio_input,outputs=gradio_Output ,
                              title="Meta Watermark Remover For Image",
                              examples=meta_examples)
# gradio_interface.launch(debug=True)



gradio_multiple_images = gr.Interface(
    process_files,
    [gr.File(type='filepath', file_count='multiple',label='Upload Images')],
    [gr.File(label='Download File'),gr.Image(label='Display Image')],
    title='Meta Watermark Remover For Bulk Images',
    cache_examples=True
)


meta_video_examples = [ "./videos/2.mp4","./videos/1.mp4"]

gradio_video_input=[gr.Video(label='Upload Video')]
gradio_video_Output=[gr.File(label='Download Video'),gr.Video(label='Display Video')]
gradio_video_interface = gr.Interface(fn=process_video, inputs=gradio_video_input,outputs=gradio_video_Output ,
                              title="Meta Watermark Remover For Video",
                              examples=meta_video_examples)


demo = gr.TabbedInterface([gradio_interface, gradio_video_interface,gradio_multiple_images], ["Meta Watermark Remover For Image","Meta Watermark Remover For Video","Meta Watermark Remover For Bulk Images"],title="Meta Watermark Remover")
# demo.launch(debug=True)
demo.queue().launch()