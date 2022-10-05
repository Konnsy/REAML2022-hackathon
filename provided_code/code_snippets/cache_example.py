# save preprocessed files generated by a list of training datasets 'ds_train'
count_ex = 0
if not osp.exists('cache_train.dir'):
	with shelve.open('cache_train') as db_train:
		for ds in ds_train:
			for img, mask in ds:
				db_train[str(count_ex)] = [img, mask]
				count_ex += 1
			

# use a random entry (image and mask) from the cache
idx_ex = random.randint(0, len(db_train)-1)
img, mask = db_train[str(idx_ex)]