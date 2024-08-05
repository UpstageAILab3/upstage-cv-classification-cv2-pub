import argparse
import cv2
import os
import pandas as pd
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def process_image(input_path, output_path, args, upsampler, face_enhancer=None):
    if not os.path.exists(input_path):
        print(f"Input file does not exist: {input_path}")
        return
    
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read image: {input_path}")
        return
    
    imgname, extension = os.path.splitext(os.path.basename(input_path))
    
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None
    
    try:
        if face_enhancer:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
    except RuntimeError as error:
        print(f'Error processing {input_path}: {error}')
        return
    
    if args.ext == 'auto':
        extension = extension[1:]
    else:
        extension = args.ext
    if img_mode == 'RGBA':  # RGBA images should be saved in png format
        extension = 'png'
    save_path = os.path.join(output_path, f'{imgname}.{extension}')
    cv2.imwrite(save_path, output)
    print(f"Saved enhanced image: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_csv', type=str, required=True, help='CSV file containing image IDs')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model name')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='Upscaling scale')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension for output images')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance faces')
    parser.add_argument('--gpu-id', type=int, default=None, help='GPU device to use')
    args = parser.parse_args()

    # Read the CSV file
    df = pd.read_csv(args.input_csv)
    image_ids = df['ID'].tolist()

    # Determine model and download if necessary
    if args.model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    # Add other model conditions as needed

    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(url=url, model_dir='weights', progress=True, file_name=None)

    # Set up the Real-ESRGAN model
    upsampler = RealESRGANer(scale=netscale, model_path=model_path, model=model, gpu_id=args.gpu_id)
    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    os.makedirs(args.output, exist_ok=True)

    input_dir = os.path.join(os.path.dirname(args.input_csv), 'test')
    for image_id in image_ids:
        input_path = os.path.join(input_dir, image_id)
        print(f"Processing {input_path}...")
        process_image(input_path, args.output, args, upsampler, face_enhancer)

if __name__ == '__main__':
    main()
