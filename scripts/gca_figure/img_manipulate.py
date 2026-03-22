import cv2
import numpy as np
import typer

app = typer.Typer()

@app.command()
def temp():
    pass

@app.command()
def bw_threshold(img_path):
    img = cv2.imread(img_path)
    bw_img = img.sum(axis=2) == 0
    bw_img = 255 * bw_img.astype(np.uint8)
    out_path = img_path.replace('.png', '_bw.png')
    cv2.imwrite(out_path, bw_img)

@app.command()
def downsample_img(img_path, downsample_rate: int=4):
    img = cv2.imread(img_path)

    downsample_img = 255 * np.ones_like(img)
    for i in range(img.shape[0] // downsample_rate):
        for j in range(img.shape[1] // downsample_rate):
            start_x = downsample_rate * i
            start_y = downsample_rate * j
            downsample_block = img[start_x: start_x + downsample_rate, start_y: start_y + downsample_rate]
            pix_val = (downsample_block.sum(axis=2) ==0).sum() == 0
            downsample_img[downsample_rate * i, downsample_rate * j] =  255 * pix_val
    out_path = img_path.replace('.png', '_downsample.png')
    cv2.imwrite(out_path, downsample_img)

    # upsample_img = 255 * np.ones((upsample * img.shape[0], upsample * img.shape[1], img.shape[2]))
    # cnt = 0
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if np.sum(img[i, j]) == 0:
    #             cnt += 1
    #             x_start = upsample * i
    #             y_start = upsample * j
    #             upsample_img[x_start: x_start + upsample, y_start: y_start + upsample] = downsample_img[i, j]
    # print(f'cnt: {cnt}')
    # out_path = img_path.replace('.png', '_upsample_no_fill.png')
    # cv2.imwrite(out_path, upsample_img)

def get_neighborhood(coord, img_shape, radius=1):
    neigh_range = np.arange(2 * radius + 1) - radius
    neigh_coords = []
    for i in neigh_range:
        for j in neigh_range:
            neigh_coords.append(coord + np.array([[i, j]]))
    neigh_coords = np.concatenate(neigh_coords, axis=0)
    neigh_coords = np.unique(neigh_coords, axis=0)

    neigh_coords = neigh_coords[neigh_coords[:, 0] >= 0]
    neigh_coords = neigh_coords[neigh_coords[:, 0] < img_shape[0]]
    neigh_coords = neigh_coords[neigh_coords[:, 1] >= 0]
    neigh_coords = neigh_coords[neigh_coords[:, 1] < img_shape[1]]

    img = np.zeros(img_shape[:2], dtype=bool)
    img[neigh_coords] = True
    return img

def load_png_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        return img
    trans_mask = img[:, :, 3] == 0
    img[trans_mask, :3] = [255, 255, 255]
    return img[:, :, :3]


@app.command()
def sim_infusion(
    s0_img_path,
    gt_img_path,
    alpha_0: float = 0.3,
    alpha_t: float = 0.1,
    num_steps: int = 5,
):

    s0_img = load_png_img(s0_img_path)
    # map black to 1, white to 0
    # s0 = (np.sum(s0_img, axis=2) == 0).astype(np.float32)
    s = 1 - (s0_img[:, :, :3] / 255)
    # random init color on black voxels
    s0_occ_coord = np.where(s.sum(axis=2) == 3)
    s[s0_occ_coord] = np.random.rand(s0_occ_coord[0].shape[0], 3)
    gt_img = load_png_img(gt_img_path)
    gt = 1 - (gt_img / 255)
    gt_occ = gt.sum(axis=2) != 0

    out_path = s0_img_path.replace('.png', f'_0.png')
    s_img = 255 * (1 - s)
    cv2.imwrite(out_path, s_img.astype(np.uint8))


    for i in range(num_steps):
        # get neighborhood mask
        s_occ = s.sum(axis=2) != 0
        occ_coord = np.stack(np.where(s_occ), axis=1)
        neigh_mask = get_neighborhood(occ_coord, s.shape)

        # update alpha
        alpha = min((i + 1) * alpha_t + alpha_0, 1.)

        # sample voxel occupancy
        next_prob = (1 - alpha) * s_occ + alpha * gt_occ
        next_mask = np.random.binomial(1, next_prob).astype(bool)

        # update on neighborhood
        s_new = np.zeros_like(s)
        blend = (1 - alpha) * s + alpha * gt
        s_new[neigh_mask & next_mask] = blend[neigh_mask & next_mask]

        s = s_new

        out_path = s0_img_path.replace('.png', f'_{i + 1}.png')
        s_img = 255 * (1 - s)
        cv2.imwrite(out_path, s_img.astype(np.uint8))



if __name__ == "__main__":
    app()
