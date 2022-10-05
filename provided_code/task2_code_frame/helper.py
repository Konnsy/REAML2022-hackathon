import sys
from PIL import ImageDraw, Image
import uuid
import cv2
import io
import os
import os.path as osp
import numpy as np
import math
import torch

def writeBoxesToImage(img, boxes, fill_color=(255,0,0), lineWidth = 2):
	"""
    write coloured boxes to an image of gray values
    img shape: (x, y)
    boxes format: (x1, y1, x2, y2)
	from https://github.com/Konnsy/GraphsAndGeometry
    """
	for box in boxes:
		img = cv2.rectangle(img, (box[0], box[1]), 
					  (box[2],box[3]), fill_color, lineWidth)
	return img


def boxIoU(boxA, boxB):
	"""
	Calculates the Intersection over Union value between to boxes.
	box format: [x1, y1, x2, y2]
	from https://github.com/Konnsy/GraphsAndGeometry
	"""
	# compute the area of the intersecting rectangle
	dx = min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])
	dy = min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])
	if dx < 0 or dy < 0:
		return 0.0
	else:
		interArea = dx * dy

	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def mergeBoxes(boxes):
	"""
	Merges a list of boxes.
	box format: [x1, y1, x2, y2]
	from https://github.com/Konnsy/GraphsAndGeometry
	"""
	if len(boxes)==0:
		return boxes

	x1 = sys.float_info.max
	y1 = sys.float_info.max
	x2 = sys.float_info.min 
	y2 = sys.float_info.min

	for box in boxes:
		x1 = min(x1, box[0])
		y1 = min(y1, box[1])
		x2 = max(x2, box[2])
		y2 = max(y2, box[3])

	return [x1, y1, x2, y2]


def isBoxContained(boxA, boxB):
	"""
	Checks if boxA contains boxB
	box format: [x1, y1, x2, y2]
	from https://github.com/Konnsy/GraphsAndGeometry
	"""
	return isPointContained(boxA, (boxB[0], boxB[1])) and isPointContained(boxA, (boxB[2], boxB[3]))


def isPointContained(box, point):
	"""
	Checks if box contains (x,y)-coordinate tuple point
	box format: [x1, y1, x2, y2], point format: [x,y]
	from https://github.com/Konnsy/GraphsAndGeometry
	"""
	return point[0] >= box[0] and point[0] <= box[2] and point[1] >= box[1] and point[1] <= box[3]


def mergeBoxesByIoU(boxes, iouThreshold):
	"""
	Merges boxes from a list. A merged box will replace a set of boxes that have 
	an IoU value greater than iouThreshold with each other.
	box format: [x1, y1, x2, y2]
	from https://github.com/Konnsy/GraphsAndGeometry
	"""
	# find pairs which have an iou greater than the given threshold
	pairsToMerge = []
	for i in range(len(boxes)):
		for j in range(i+1, len(boxes)):
			if boxIoU(boxes[i], boxes[j]) >= iouThreshold:
				pairsToMerge.append([i,j])
		
	# merge boxes within each clique
	resBoxes = []
	cliques = calcCliques(pairsToMerge, [i for i in range(len(boxes))])
	for clique in cliques:
		if len(clique) == 1:
			resBoxes.append(boxes[clique[0]])
		elif len(clique) > 1:
			cliqueBoxes = [boxes[id] for id in clique]
			resBoxes.append(mergeBoxes(cliqueBoxes))

	return resBoxes


def calcCliques(edges, nodes):
	"""
	Calculates cliques in a graph given by a list of edges and a list of nodes.

	edges: edges of connected elements, format: (idx node 1, idx node 2)
	nodes: all node idxs existent, format: int as id
	result: sorted list of node ids for each clique

	from https://github.com/Konnsy/GraphsAndGeometry

	Example usage:
		> edges = [[1,2], [2,5], [3,0], [5,6]]
		> nodes = [0,1,2,3,4,5,6]

		> cliques = calcCliques(edges, nodes)
		=> cliques returns [[1, 2, 5, 6], [0, 3], [4]]
	"""
	nodes.sort()
	
	# initialize connections directory
	con = {}
	for (a,b) in edges:
		if a in con:
			con[a].add(b)
		else:
			con[a] = set()
			con[a].add(b)

	# merge until nothing to merge is found
	changed = True
	while changed:
		changed = False

		for idx1 in range(len(nodes)-1, 0, -1):			
			for idx2 in range(min(len(nodes), idx1)):								
				if nodes[idx1] in con and nodes[idx2] in con and nodes[idx1] in con[nodes[idx2]]:
					con[nodes[idx2]].update(con[nodes[idx1]])
					del con[nodes[idx1]]
					changed = True
	
	inClique = set()
	for valList in con.values():
		inClique.update(valList)	
	inClique.update(con.keys())

	cliques = []
	for node in nodes:
		if node in con:
			cliques.append([node]+list(con[node]))
		elif not (node in inClique):
			cliques.append([node])

	for idx in range(len(cliques)):
		cliques[idx].sort()

	return cliques


def mergeableBoxCliquesByIoU(boxes, iouThreshold):
	"""
	returns cliques (lists of box indices) which can be merged 
	based in their iou-values
	"""

	# find pairs which have an iou greater than the given threshold
	pairsToMerge = []
	for i in range(len(boxes)):
		for j in range(i+1, len(boxes)):
			biou = boxIoU(boxes[i], boxes[j])
			if biou > 0.0 and biou >= iouThreshold:
				pairsToMerge.append([i,j])
		
	return calcCliques(pairsToMerge, [i for i in range(len(boxes))])

def pilToBytes(img):
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def pilToMutableNumpyArray(img):
    byteArr = pilToBytes(img)    
    decoded = cv2.imdecode(np.frombuffer(byteArr, np.uint8), -1)
    return decoded


def drawEllipseFromBox(im, box):
    x1, y1, x2, y2 = box
    center = (int((x1+x2)*0.5), int((y1+y2)*0.5))
    boxSides = (x2-x1, y2-y1)

    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    draw.ellipse(box, fill=1)
    del draw

    arr = pilToMutableNumpyArray(im)
    del im
    arr = (arr>0).astype(np.uint8) * 255

    poly = []
    steps = 7
    for iAngle in range(steps):
        angle = ((2*math.pi)/float(steps))*iAngle

        px = int((boxSides[1]*0.5)*math.cos(angle) + center[1])
        py = int((boxSides[0]*0.5)*math.sin(angle) + center[0])
        poly.append([px, py])

    px = int((boxSides[1]*0.5)*math.cos(0) + center[1])
    py = int((boxSides[0]*0.5)*math.sin(0) + center[0])
    poly.append((px, py))

    return arr, poly


def maskImgFromPolygons(polygons, im_width, im_height):
    mergedMask = np.zeros(shape=(im_width, im_height), dtype=np.uint8)

    # sequential conversion to masks
    masks = []
    for iPoly, polygon in enumerate(polygons):
        cover = np.equal(mergedMask, 0).astype(np.uint16)
        mask = maskFromPolygon((polygon, im_width, im_height), pixelValue=iPoly+1).astype(np.uint16)
        mask = np.multiply(mask, cover) # blank out mask values at places already covered in the mergedMask
        mergedMask = np.add(mergedMask, mask)

    return mergedMask


def maskFromPolygons(polygons, height, width, heightFirst=True, boolVersion=False):    
    if heightFirst:
        totalMask = np.zeros(shape=(height, width), dtype=np.uint16)
    else:
        totalMask = np.zeros(shape=(width, height), dtype=np.uint16)

    for polygon in polygons:
        singleMask = maskFromPolygon((polygon, height, width), pixelValue=1, heightFirst=heightFirst).astype(np.uint16)
        totalMask = np.add(totalMask, singleMask)
        
    if boolVersion:
        totalMask = totalMask.astype(np.bool)
      
    return totalMask


def maskFromPolygon(data, pixelValue=1, heightFirst=True):
    polygon, height, width = data
    polygon = [list(e) for e in polygon]
    polygon_x = [e[0] for e in polygon]
    polygon_y = [e[1] for e in polygon]
    polygon = list(zip(polygon_x, polygon_y))

    img = Image.new('L', (height, width), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)

    if heightFirst:
        mask = mask.transpose(1, 0)

    return mask


def containsRawFolder(folder):
    if not osp.isdir(folder):
        return False
    return len(list(filter(lambda f : f == "raw", os.listdir(folder)))) > 0


def getAllWithRawFolders(superfolder, recursive=True):
    """
    determine the paths of datasets by searching for 'raw' subfolders
    """
    withRawfolders = list(filter(lambda fd : osp.isdir(osp.join(superfolder, fd)) and
                                 'raw' in list(os.listdir(osp.join(superfolder, fd))), os.listdir(superfolder)))
    withRawfolders = list(map(lambda fd : osp.join(superfolder, fd), withRawfolders))

    if recursive:
        for fp in os.listdir(superfolder):
            fn = osp.join(superfolder, fp)
            if osp.isdir(fn):
                withRawfolders.extend(getAllWithRawFolders(fn, recursive=True))

    return withRawfolders


def scale(x, out_range=(0, 1)):
	"""
	Scale tensor values to a sepcific range
	"""
	domain = torch.min(x), torch.max(x)	
	if (domain[1] - domain[0]) < sys.float_info.epsilon:
		return x

	y = torch.true_divide((x - torch.true_divide((domain[1] + domain[0]), 2)), (domain[1] - domain[0]))
	return y * (out_range[1] - out_range[0]) + torch.true_divide((out_range[1] + out_range[0]), 2)
