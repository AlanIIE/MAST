import numpy as np
import cv2

def draw_track_trace(source, target, frame=None):
    assert target.shape == source.shape
    assert target.shape[3] == 2
    B, h, w, C = source.shape
    # if frame:
    #     frame = np.zeros((B, h, w, 1))
    source = source.reshape(B, h*w, C)
    target = target.reshape(B, h*w, C)

    N = int(h*w/10)
    color = np.random.randint(0,255,(N,frame.shape[-1]))
    flow = source-target
    rad = np.sqrt(np.square(flow[...,0]) + np.square(flow[...,1]))
    ind = np.argsort(rad,axis=1)
    source_calc = np.zeros((B,N,2), dtype=np.int)
    target_calc = np.zeros((B,N,2), dtype=np.int)
    for i in range(B):
        source_calc[i,...] = source[i,ind[i,-N:],]
        target_calc[i,...] = target[i,ind[i,-N:],]
    

    # draw the tracks
    res = []
    for i in range(B):
        mask = np.zeros((h,w,frame.shape[-1]), dtype=np.float32)
        frame_now = frame[i,...].copy()
        for j in range(N):
            a,b = source_calc[i,j,:]
            c,d = target_calc[i,j,:]
            # print((b,a),(d,c))
            mask = cv2.line(mask, (b,a),(d,c), color[j].tolist(), 2)
            frame_now = cv2.circle(frame_now,(b,a),5,color[j].tolist(),-1)
        img = cv2.add(frame_now,mask)
        res.append(img)
    return res

    # cv2.imshow('frame',img)


def draw_trace_from_flow(flow, frame=None):
    assert flow.shape[3]==2
    b,h,w,c = flow.shape
    grid_x, grid_y = np.meshgrid(np.arange(h),np.arange(w))
    grid = np.concatenate((grid_x[...,np.newaxis], grid_y[...,np.newaxis]), axis=-1).transpose(1,0,2)
    return draw_track_trace(grid[np.newaxis,...], grid[np.newaxis,...]+flow, frame)