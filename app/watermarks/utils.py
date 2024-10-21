def box_to_xyxy(box, image):
    box = [
        box[0] * image.width,
        box[1] * image.height,
        box[2] * image.width,
        box[3] * image.height,
    ]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    sz = max(box[2] - box[0], box[3] - box[1]) * 1.20
    x0, y0, x1, y1 = (
        int(cx - sz / 2),
        int(cy - sz / 2),
        int(cx + sz / 2),
        int(cy + sz / 2),
    )
    return x0, y0, x1, y1
