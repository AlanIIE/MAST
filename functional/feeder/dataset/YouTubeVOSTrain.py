import os, sys
import os.path
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def dataloader(csv_path="ytvos.csv", num_long=2, ref_num=3, dil_int=15, dirates=1, E=0):
    filenames = open(csv_path).readlines()

    frame_all = [filename.split(',')[0].strip() for filename in filenames]
    startframe = [int(filename.split(',')[1].strip()) for filename in filenames]
    nframes = [int(filename.split(',')[2].strip()) for filename in filenames]

    all_index = np.arange(len(nframes))
    np.random.shuffle(all_index)

    refs_train = []
    frame_indices = []

    for index in all_index:
        frame_interval = np.random.choice([2,5,8],p=[0.4,0.4,0.2])

        # compute frame index (ensures length(image set) >= random_interval)
        refs_images = []

        n_frames = nframes[index]
        start_frame = startframe[index]
        if True:
            if dirates == 4:
                frame_set = range((dirates-1)*dil_int+start_frame+frame_interval*(num_long-ref_num)-1,
                                start_frame+n_frames-(ref_num-1)*frame_interval-1)
            else:
                frame_set = range(max(start_frame,(dirates-1)*dil_int+start_frame+frame_interval*(num_long-ref_num)-1),
                                dirates*dil_int+start_frame+frame_interval*(num_long-ref_num)-2)
            # frame_set = range(start_frame+frame_interval+dil_int, start_frame+n_frames-(ref_num-1)*frame_interval-1)
            if frame_set:
                frame_set = np.random.choice(frame_set)
            else:
                continue
            frame_indices_batches = [[x for x in range(frame_set,frame_set+frame_interval*(ref_num-1)+1,frame_interval)]]
        else:
            frame_indices = np.arange(start_frame, start_frame+n_frames, frame_interval)  # start from startframe
            total_batch, batch_mod = divmod(len(frame_indices), ref_num)
            if total_batch == 0: continue # fix the bug for ref_num>2
            if batch_mod > 0:
                frame_indices = frame_indices[:-batch_mod]
            frame_indices_batches = np.split(frame_indices, total_batch)
        for batches in frame_indices_batches:
            batches = np.sort(np.concatenate((batches,
                                list(range(start_frame, start_frame+frame_interval*(num_long-1)+1,frame_interval))+
                                [frame_set+frame_interval*(ref_num-1)+1])))
            # ref_images = [os.path.join(frame_all[index], '{:05d}.jpg'.format(frame))
            #               for frame in [max(start_frame,batches[0]-30)]+ list(batches)]
            ref_images = [os.path.join(frame_all[index], '{:05d}.jpg'.format(frame))
                          for frame in list(batches)]
            refs_images.append(ref_images)
        
        frame_indices.extend([batches])
        refs_train.extend(refs_images)

    return refs_train, frame_indices

if __name__ == '__main__':
    x = dataloader()
