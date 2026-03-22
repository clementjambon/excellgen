import os
from PIL import Image
from pathlib import Path
import argparse


def downscale_images(input_folder, output_folder, scale_factor):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for file in input_folder.iterdir():
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            try:
                with Image.open(file) as img:
                    new_size = (
                        int(img.width * scale_factor),
                        int(img.height * scale_factor),
                    )
                    downscaled = img.resize(new_size, Image.Resampling.LANCZOS)

                    output_path = output_folder / file.name
                    downscaled.save(output_path)
                    print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Skipping {file.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Downscale images in a folder by a given factor."
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing input images."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder to save downscaled images.",
        default=None,
    )
    parser.add_argument(
        "--scale_factors",
        type=float,
        nargs="+",
        help="Factors to downscale images by (e.g., 0.5 for half size).",
        default=[2, 4],
    )

    args = parser.parse_args()

    output_folder = args.output_folder
    if output_folder is None:
        output_folder = os.path.dirname(args.input_folder)
        output_folder = os.path.join(output_folder, "images")

    for scale_factor in args.scale_factors:
        downscale_images(
            args.input_folder, output_folder + f"_{scale_factor}", 1.0 / scale_factor
        )


if __name__ == "__main__":
    main()
