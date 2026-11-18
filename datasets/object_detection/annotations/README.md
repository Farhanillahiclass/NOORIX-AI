# Annotation format

Detection annotations use a COCO-style JSON document with three top-level arrays:

- images: each record has a unique integer id, relative file_name, positive integer width and height.
- categories: each record has a unique integer id and non-empty name; supercategory is optional.
- annotations: each record has a unique integer id, valid image_id and category_id, bbox as [x, y, width, height], positive finite area, and optional iscrowd (0 or 1).

Bounding boxes use COCO's top-left-origin pixel coordinates. Width and height must be positive, and the complete box must lie inside its image. The area field can represent segmentation area and therefore is not required to equal bbox width × height.

The validator checks structure and references; it cannot determine whether a human assigned the correct class or box. Keep source annotations and conversion scripts traceable. Never create labels to make a dataset appear complete.
