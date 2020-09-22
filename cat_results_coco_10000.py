import os
detr50_226 = ['/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000599.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000001442.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000001675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002295.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002587.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003711.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004795.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006460.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006589.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000007722.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000008196.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009041.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009466.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009807.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009830.jpg']

detr50_416 = ['/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000042.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000599.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000001675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002295.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002988.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003067.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003711.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004392.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004795.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006397.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006589.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000007722.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009041.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009466.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009572.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009807.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009830.jpg']

detr101_226 = ['/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000599.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000001675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002212.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002295.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002988.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003595.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003711.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004108.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004795.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000005652.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006397.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006589.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006658.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000007288.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000007722.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009466.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009807.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009830.jpg']

detr101_416 = ['/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000599.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000001675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002212.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002295.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002988.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003067.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003711.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004392.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004795.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000005577.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006397.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006589.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000007722.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009170.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009236.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009466.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009807.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009830.jpg']

yolo_226 = ['/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000599.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000001675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002295.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002988.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003067.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003711.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004795.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000005617.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006397.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006589.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000007722.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009041.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009170.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009466.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009807.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009830.jpg']

yolo_416 = ['/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000599.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000000675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000001398.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000001675.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002295.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000002988.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003067.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000003711.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004212.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004392.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000004795.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000005577.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000005617.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000005804.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006397.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006589.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000007722.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009041.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009170.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009466.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009807.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009830.jpg']

yolo_tiny_226 = ['/home/ubuntu/coco/val2014_10000/COCO_val2014_000000006589.jpg', '/home/ubuntu/coco/val2014_10000/COCO_val2014_000000007722.jpg']

yolo_tiny_416 = ['/home/ubuntu/coco/val2014_10000/COCO_val2014_000000009041.jpg']


l = {'detr50_226': detr50_226, 
    'detr50_416': detr50_416 , 
    'detr101_226': detr101_226, 
    'detr101_416': detr101_416, 
    'yolo_226': yolo_226, 
    'yolo_416': yolo_416, 
    'yolo_tiny_226': yolo_tiny_226, 
    'yolo_tiny_416': yolo_tiny_416}


for name, result in l.items():
    os.mkdir(name)
    for img in result:
        cmd = 'cp ' + img + " ./" + name
        os.system(cmd) 

