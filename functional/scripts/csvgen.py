import os, csv

# path = '/datasets/zlai/train_all_frames/JPEGImages'
path = '/home/kuang/DataSets/OxUvA/dev/'
ld = os.listdir(path)

# with open('ytvos_train.csv', 'w') as f:
with open('OxUvA_train.csv', 'w') as f:
    filewriter = csv.writer(f)
    for l in ld:
        n = len(os.listdir(os.path.join(path,l)))
        start_frame = 0
        n_frame = 0
        jpg_files = os.listdir(os.path.join(path,l))
        for file in jpg_files:
            now_frame = int(file.split('.')[0])
            if now_frame >= n_frame-start_frame:
                n_frame = now_frame - start_frame
            if now_frame <= start_frame:
                start_frame = now_frame
        if n_frame+1 != n:
            print(l, start_frame,n,n_frame)
            continue
        filewriter.writerow([l, start_frame, n])
