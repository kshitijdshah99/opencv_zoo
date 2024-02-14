import os
import cv2
import argparse
# from ppocr_det import PPOCRDet

# Define the argument parser
parser = argparse.ArgumentParser(description='PP-OCR Text Detection.')
parser.add_argument('--model', '-m', type=str, default='./text_detection_en_ppocrv3_2023may.onnx',
                    help='Path to the model file.')
parser.add_argument('--width', type=int, default=736,
                    help='Width of the input image.')
parser.add_argument('--height', type=int, default=736,
                    help='Height of the input image.')
# Add other arguments as needed

# Parse the arguments
args = parser.parse_args()

# Instantiate the model
model = PPOCRDet(modelPath=args.model,
                 inputSize=[args.width, args.height],
                 binaryThreshold=args.binary_threshold,
                 polygonThreshold=args.polygon_threshold,
                 maxCandidates=args.max_candidates,
                 unclipRatio=args.unclip_ratio,
                 backendId=backend_id,
                 targetId=target_id)

# Get list of .jpg files in folder
folder_path = "./opencv_zoo/tools/eval/MSRA-TD500/test"
jpg_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.JPG')]

# Iterate over each .jpg file in the folder
for jpg_file in jpg_files:
    # Read image
    original_image = cv2.imread(jpg_file)
    original_w = original_image.shape[1]
    original_h = original_image.shape[0]
    scaleHeight = original_h / args.height
    scaleWidth = original_w / args.width
    image = cv2.resize(original_image, [args.width, args.height])

    # Inference
    results = model.infer(image)

    # Scale the results bounding box
    for i in range(len(results[0])):
        for j in range(4):
            box = results[0][i][j]
            results[0][i][j][0] = box[0] * scaleWidth
            results[0][i][j][1] = box[1] * scaleHeight

    # Print results
    print(f'Results for image: {jpg_file}')
    print(f'{len(results[0])} texts detected.')
    for idx, (bbox, score) in enumerate(zip(results[0], results[1])):
        print(f'{idx}: {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}, {score}')
